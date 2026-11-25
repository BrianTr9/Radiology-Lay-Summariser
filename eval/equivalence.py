"""Does the merged writer reproduce the numbers measured on the unmerged model?

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

The reported ROUGE and reading-level numbers were measured with the LoRA adapter unmerged,
batches of 32 with padding, on an A100 (training/). The demo merges the adapter and drafts one
report at a time. This script drafts a seeded sample of validation reports the way the demo
does and compares them with the stored predictions of the original run, row by row.

It reports: the share of identical drafts, ROUGE-Lsum of both sets against the references,
ROUGE-Lsum between the two sets, and the reading grade of both. It writes only counts and
row indices (no report text) to eval/results/equivalence.json. Run it on a GPU (Colab) with
the settings the demo uses; the file records the device and library versions.

Usage: python eval/equivalence.py --adapter <adapter folder or Hub id> \
           --predictions <runs_from_rangpur/lora_r16_lr3e-3/val_predictions.jsonl>
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "training"), str(ROOT)]

import torch  # noqa: E402
import transformers  # noqa: E402

from autopsy import reading_grade  # noqa: E402
from dataset import SOURCE_COLUMN, TARGET_COLUMN, evaluation_tiers, load_laysumm  # noqa: E402
from laysum import writer as writer_module  # noqa: E402
from laysum.writer import FlanT5Writer  # noqa: E402
from utils import mean_rouge, rouge_per_example  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, help="adapter folder or Hub id")
    ap.add_argument("--adapter-revision", default=None, help="Hub commit of the adapter; none for a folder")
    ap.add_argument("--predictions", required=True, help="val_predictions.jsonl of LoRA r16")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "equivalence.json"))
    args = ap.parse_args()

    data = load_laysumm()
    sources = list(data["validation"][SOURCE_COLUMN])
    references = list(data["validation"][TARGET_COLUMN])
    clean = set(evaluation_tiers(data["train"], data["validation"])["clean"])
    stored = {}
    for line in open(args.predictions):
        row = json.loads(line)
        stored[row["index"]] = row["prediction"]
    assert sorted(stored) == list(range(len(sources))), "predictions must cover validation"

    rows = sorted(random.Random(args.seed).sample(range(len(sources)), args.n))
    w = FlanT5Writer.from_pretrained(args.adapter, adapter_revision=args.adapter_revision)
    w.summarise(sources[rows[0]])  # warm-up, result discarded

    ours, seconds = {}, 0.0
    for i in rows:
        start = time.perf_counter()
        ours[i] = w.summarise(sources[i]).text
        if w.device == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - start

    ours_list = [ours[i] for i in rows]
    stored_list = [stored[i] for i in rows]
    refs = [references[i] for i in rows]
    srcs = [sources[i] for i in rows]
    r_ours, r_stored = rouge_per_example(ours_list, refs), rouge_per_example(stored_list, refs)
    r_between = rouge_per_example(ours_list, stored_list)
    different = [i for i in rows if ours[i] != stored[i]]
    in_clean = [k for k, i in enumerate(rows) if i in clean]

    def block(idx):
        n = len(idx)
        return {
            "n": n,
            "identical": sum(ours_list[k] == stored_list[k] for k in idx),
            "rouge_lsum_vs_reference_merged": round(mean_rouge(r_ours, idx)["rougeLsum"], 2),
            "rouge_lsum_vs_reference_original": round(mean_rouge(r_stored, idx)["rougeLsum"], 2),
            "rouge_lsum_between_merged_and_original": round(mean_rouge(r_between, idx)["rougeLsum"], 2),
            "reading_grade_merged": round(sum(reading_grade(ours_list[k]) for k in idx) / n, 2),
            "reading_grade_original": round(sum(reading_grade(stored_list[k]) for k in idx) / n, 2),
            "reading_grade_source": round(sum(reading_grade(srcs[k]) for k in idx) / n, 2),
        }

    out = {
        "n": args.n, "seed": args.seed, "indices": rows,
        "all_sampled": block(list(range(len(rows)))),
        "clean_tier_only": block(in_clean),
        "different_indices": different,
        "seconds_per_report": round(seconds / len(rows), 3),
        "settings": {"device": w.device, "model_version": w.version,
                     "num_beams": writer_module.NUM_BEAMS, "max_new_tokens": writer_module.MAX_NEW_TOKENS,
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    for key in ("all_sampled", "clean_tier_only"):
        b = out[key]
        b["identical_rate"] = round(b["identical"] / b["n"], 3)
        b["rouge_lsum_difference"] = round(b["rouge_lsum_vs_reference_merged"]
                                           - b["rouge_lsum_vs_reference_original"], 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    a = out["all_sampled"]
    print(f"{a['identical']}/{a['n']} identical drafts ({a['identical_rate']:.1%}); "
          f"ROUGE-Lsum vs reference: merged {a['rouge_lsum_vs_reference_merged']}, "
          f"original {a['rouge_lsum_vs_reference_original']} (diff {a['rouge_lsum_difference']:+}); "
          f"reading grade {a['reading_grade_merged']} vs {a['reading_grade_original']}; "
          f"{out['seconds_per_report']} s/report on {w.device}")


if __name__ == "__main__":
    main()
