"""How many drafts would the gate send to REVIEW? Chooses the counting level, then reports.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Runs on CPU, no model: it reuses the stored validation predictions of the course project
and the official validation split. Rule fixed before running (z_Roadmap.md, Decisions):
  1. On the 5,963 validation rows OUTSIDE the clean tier, compute the REVIEW rate of LoRA
     r16 with sentence-level counting. If it is above 25%, use document-level counting.
  2. Report per-flag and overall REVIEW rates on the clean tier (4,037 rows) with the
     chosen level, so the level is chosen and reported on disjoint rows.
GPT-4o references are scored the same way as a reading aid (they are not ground truth).

Writes only counts and rates (no dataset text) to eval/results/review_rate.json.

Usage: python eval/review_rate.py --predictions <runs_from_rangpur/lora_r16_lr3e-3/val_predictions.jsonl>
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "training"), str(ROOT)]  # course code: training/ after Phase 1

from transformers import AutoTokenizer  # noqa: E402

from dataset import (MAX_SOURCE_LEN, SOURCE_COLUMN, TARGET_COLUMN, TASK_PROMPT,  # noqa: E402
                     build_source_text, evaluation_tiers, load_laysumm)
from laysum.checks import LEVELS, check, strip_junk_tail  # noqa: E402

THRESHOLD = 0.25
CODES = ("INPUT_TRUNCATED", "NEGATION_LOST", "CERTAINTY_LOST")


def truncation_points(reports, tokenizer, max_len=MAX_SOURCE_LEN):
    """Character offset in each report where the writer stops reading, or None.

    Matches the course data pipeline: tokenizer(prompt + report, max_length=512,
    truncation=True) keeps max_len - 1 text tokens and then appends </s>.
    """
    start = len(build_source_text("x")) - 1  # where the report begins in the input text
    assert build_source_text("x")[start:] == "x"
    points = []
    for report in reports:
        text = build_source_text(report)
        enc = tokenizer(text, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= max_len:
            points.append(None)
            continue
        first_dropped = enc["offset_mapping"][max_len - 1][0]
        stripped_lead = len(report) - len(report.lstrip())  # build_source_text strips
        points.append(first_dropped - start + stripped_lead)
    return points


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
    args = ap.parse_args()

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
    cuts = truncation_points(sources, tokenizer)

    # Step 1: choose the level on rows outside the clean tier.
    choice = {lvl: rates(sources, drafts, cuts, outside, lvl) for lvl in LEVELS}
    level = "sentence" if choice["sentence"]["review_rate"] <= THRESHOLD else "document"

    # Step 2: report on the clean tier.
    report = {lvl: {"lora_r16": rates(sources, drafts, cuts, clean, lvl),
                    "reference": rates(sources, references, cuts, clean, lvl)}
              for lvl in LEVELS}
    out = {
        "rule": f"sentence level unless its REVIEW rate on rows outside the clean tier > {THRESHOLD}",
        "prompt": TASK_PROMPT, "max_source_tokens": MAX_SOURCE_LEN,
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


if __name__ == "__main__":
    main()
