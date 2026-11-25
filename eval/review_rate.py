"""How many drafts would the gate send to REVIEW? Chooses the counting level, then reports.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Runs on CPU, no model: it reuses the stored validation predictions of the LoRA r16 run
(training/) and the official validation split. Rule fixed before running:
  1. On the 5,963 validation rows OUTSIDE the clean tier, compute the REVIEW rate of LoRA
     r16 with sentence-level counting. If it is above 25%, use document-level counting.
  2. Report per-flag and overall REVIEW rates on the clean tier (4,037 rows) with the
     chosen level, so the level is chosen and reported on disjoint rows.
GPT-4o references are scored the same way as a reading aid (they are not ground truth).

Writes only counts and rates (no dataset text) to eval/results/review_rate.json.

Usage: python eval/review_rate.py --predictions <val_predictions.jsonl of the LoRA r16 run>
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "training"), str(ROOT)]

from dataset import SOURCE_COLUMN, TARGET_COLUMN, evaluation_tiers, load_laysumm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from laysum.checks import LEVELS, check, strip_junk_tail  # noqa: E402
from laysum.writer import MAX_SOURCE_TOKENS, TASK_PROMPT, truncation_point  # noqa: E402

THRESHOLD = 0.25
CODES = ("INPUT_TRUNCATED", "NEGATION_LOST", "CERTAINTY_LOST")


def rates(sources, drafts, cuts, rows, level):
    n_review, per_code = 0, Counter()
    for i in rows:
        flags = check(sources[i], drafts[i], cuts[i], level)
        n_review += bool(flags)
        per_code.update({f.code for f in flags})
    n = len(rows)
    return {"n": n, "review": n_review, "review_rate": round(n_review / n, 4),
            "per_flag_rate": {c: round(per_code[c] / n, 4) for c in CODES}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True, help="val_predictions.jsonl of LoRA r16")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "review_rate.json"))
    ap.add_argument("--sample", type=int, default=0,
                    help="also draw this many REVIEW and this many PASS drafts for the audit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_texts", help="private .jsonl for the sampled texts (keep it "
                                           "outside the repository)")
    args = ap.parse_args()
    if args.sample and not args.sample_texts:
        ap.error("--sample needs --sample_texts")

    data = load_laysumm()
    train, val = data["train"], data["validation"]
    sources, references = list(val[SOURCE_COLUMN]), list(val[TARGET_COLUMN])
    tiers = evaluation_tiers(train, val)
    clean = tiers["clean"]
    outside = sorted(set(tiers["full"]) - set(clean))

    preds = {}
    for line in open(args.predictions):
        r = json.loads(line)
        preds[r["index"]] = strip_junk_tail(r["prediction"])  # same cleanup as the product
    assert sorted(preds) == list(range(len(sources))), "predictions must cover validation"
    drafts = [preds[i] for i in range(len(sources))]

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    cuts = [truncation_point(s, tokenizer) for s in sources]

    # Step 1: choose the level on rows outside the clean tier.
    choice = {lvl: rates(sources, drafts, cuts, outside, lvl) for lvl in LEVELS}
    level = "sentence" if choice["sentence"]["review_rate"] <= THRESHOLD else "document"

    # Step 2: report on the clean tier.
    report = {lvl: {"lora_r16": rates(sources, drafts, cuts, clean, lvl),
                    "reference": rates(sources, references, cuts, clean, lvl)}
              for lvl in LEVELS}
    out = {
        "rule": f"sentence level unless its REVIEW rate on rows outside the clean tier > {THRESHOLD}",
        "prompt": TASK_PROMPT, "max_source_tokens": MAX_SOURCE_TOKENS,
        "selection_outside_clean": choice,
        "chosen_level": level,
        "clean_tier": report,
        "truncated_reports": {"full": sum(c is not None for c in cuts),
                              "clean": sum(cuts[i] is not None for i in clean)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    r = report[level]["lora_r16"]
    print(f"chosen level: {level} (sentence-level REVIEW outside clean = "
          f"{choice['sentence']['review_rate']:.1%})")
    print(f"clean tier, LoRA r16: REVIEW {r['review_rate']:.1%} ({r['review']}/{r['n']}); "
          + ", ".join(f"{c} {v:.1%}" for c, v in r["per_flag_rate"].items()))
    ref = report[level]["reference"]
    print(f"clean tier, references: REVIEW {ref['review_rate']:.1%}")
    if args.sample:
        draw_sample(sources, drafts, cuts, clean, level, args)


def draw_sample(sources, drafts, cuts, clean, level, args):
    """Seeded audit sample: `--sample` REVIEW and `--sample` PASS drafts of the clean tier,
    shuffled together. Texts go to the private file; the repository gets indices only."""
    flags = {i: check(sources[i], drafts[i], cuts[i], level) for i in clean}
    review = sorted(i for i in clean if flags[i])
    passed = sorted(i for i in clean if not flags[i])
    rng = random.Random(args.seed)
    picked = ([(i, "REVIEW") for i in rng.sample(review, args.sample)]
              + [(i, "PASS") for i in rng.sample(passed, args.sample)])
    rng.shuffle(picked)
    with open(args.sample_texts, "w") as f:
        for item, (i, group) in enumerate(picked):
            f.write(json.dumps({"item": item, "index": i, "group": group, "source": sources[i],
                                "draft": drafts[i], "truncated_at": cuts[i],
                                "flags": [{"code": fl.code, "source_spans": fl.source_spans}
                                          for fl in flags[i]]}) + "\n")
    out = {"seed": args.seed, "level": level, "pool": {"REVIEW": len(review), "PASS": len(passed)},
           "items": [{"item": k, "index": i, "group": g} for k, (i, g) in enumerate(picked)]}
    path = ROOT / "eval" / "results" / "audit_sample.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"audit sample: {args.sample} REVIEW of {len(review)}, {args.sample} PASS of "
          f"{len(passed)}; indices in {path.relative_to(ROOT)}, texts in {args.sample_texts}")


if __name__ == "__main__":
    main()
