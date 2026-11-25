"""Results of the audit of the review flags (see eval/audit_guide.md).

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Reads eval/results/audit_labels.json (row indices and labels only, no text) and writes
eval/results/audit.json: flag precision, the share of REVIEW and PASS drafts with an error,
the error types the gate lets through, all with Wilson 95% intervals, and an estimate for
the whole clean tier that weights the two strata by their size.

    python eval/audit.py                               # recompute from the labels
    python eval/audit.py --import_private <dir>        # first rebuild the labels file from
                                                       # the private labels_a/labels_b.jsonl
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
TYPES = ("ADDED", "NEGATION", "CERTAINTY", "CHANGED", "OMITTED", "DEGENERATE")
CODES = ("NEGATION_LOST", "CERTAINTY_LOST", "INPUT_TRUNCATED")


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"k": k, "n": n, "rate": round(p, 3),
            "ci95": [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]}


def import_private(directory: Path) -> None:
    """Join the private label files with the sample's indices; drop the free-text notes."""
    sample = json.loads((RESULTS / "audit_sample.json").read_text())
    a = {r["item"]: r for r in map(json.loads, open(directory / "labels_a.jsonl"))}
    b = {r["item"]: r for r in map(json.loads, open(directory / "labels_b.jsonl"))}
    items = []
    for s in sample["items"]:
        la = a[s["item"]]
        row = {**s, "error": la["error"], "types": la["types"], "not_plain": la["not_plain"]}
        if s["group"] == "REVIEW":
            row["flags"] = [{k: f[k] for k in ("code", "correct", "useful")}
                            for f in b[s["item"]]["flags"]]
        items.append(row)
    out = {"rater": "Claude (Anthropic), following eval/audit_guide.md; not a clinician",
           "seed": sample["seed"], "pool": sample["pool"], "items": items}
    (RESULTS / "audit_labels.json").write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--import_private", type=Path)
    ap.add_argument("--author_labels", type=Path,
                    help="private .jsonl of the author's blind re-labels of the agreement subset")
    args = ap.parse_args()
    if args.import_private:
        import_private(args.import_private)

    labels = json.loads((RESULTS / "audit_labels.json").read_text())
    items, pool = labels["items"], labels["pool"]
    groups = {g: [r for r in items if r["group"] == g] for g in ("REVIEW", "PASS")}

    flags = [f for r in groups["REVIEW"] for f in r["flags"]]
    precision = {"all": wilson(sum(f["correct"] for f in flags), len(flags))}
    for code in CODES:
        sub = [f for f in flags if f["code"] == code]
        precision[code] = wilson(sum(f["correct"] for f in sub), len(sub))
    useful = wilson(sum(f["useful"] for f in flags), len(flags))
    drafts_with_a_correct_flag = wilson(sum(any(f["correct"] for f in r["flags"])
                                            for r in groups["REVIEW"]), len(groups["REVIEW"]))

    def error_rates(rows):
        return {"significant": wilson(sum(r["error"] == "significant" for r in rows), len(rows)),
                "any": wilson(sum(r["error"] != "none" for r in rows), len(rows))}

    by_group = {g: error_rates(rows) for g, rows in groups.items()}
    pass_types = {sev: {t: sum(any(x["type"] == t and (sev == "any" or x["severity"] == sev)
                                   for x in r["types"]) for r in groups["PASS"])
                        for t in TYPES}
                  for sev in ("significant", "any")}

    # Whole clean tier: weight each stratum by its size in the pool.
    n_pool = pool["REVIEW"] + pool["PASS"]
    w_review = pool["REVIEW"] / n_pool
    p_r = by_group["REVIEW"]["significant"]["rate"]
    p_p = by_group["PASS"]["significant"]["rate"]
    tier_significant = w_review * p_r + (1 - w_review) * p_p
    caught = w_review * p_r / tier_significant if tier_significant else None

    def caught_at(pr, pp):
        return round(w_review * pr / (w_review * pr + (1 - w_review) * pp), 3)

    lo_r, hi_r = by_group["REVIEW"]["significant"]["ci95"]
    lo_p, hi_p = by_group["PASS"]["significant"]["ci95"]
    caught_range = [caught_at(lo_r, hi_p), caught_at(hi_r, lo_p)]

    agreement = author_agreement(items, args.author_labels) if args.author_labels else "pending"
    out = {
        "author_agreement": agreement,
        "rater": labels["rater"], "seed": labels["seed"], "pool": pool,
        "review_rate_clean_tier": round(w_review, 4),
        "flag_precision": precision,
        "flag_useful": useful,
        "review_drafts_with_a_correct_flag": drafts_with_a_correct_flag,
        "error_rate": by_group,
        "pass_error_types_drafts": pass_types,
        "not_plain": {g: wilson(sum(r["not_plain"] for r in rows), len(rows))
                      for g, rows in groups.items()},
        "clean_tier_estimate": {
            "significant_error_rate": round(tier_significant, 3),
            "share_of_significant_errors_sent_to_review": round(caught, 3) if caught else None,
            "share_range_from_the_two_cis": caught_range,
            "note": "strata weighted by pool size; the range combines the ends of the two "
                    "Wilson intervals, so it is wide when PASS errors are rare",
        },
        "types_total": dict(Counter(x["type"] for r in items for x in r["types"])),
    }
    (RESULTS / "audit.json").write_text(json.dumps(out, indent=2) + "\n")

    def pct(d):
        return f"{d['rate']:.0%} [{d['ci95'][0]:.0%}-{d['ci95'][1]:.0%}] ({d['k']}/{d['n']})"

    print(f"gate sends {w_review:.1%} of clean-tier drafts to REVIEW")
    print(f"flag precision {pct(precision['all'])}; useful {pct(useful)}")
    for code in CODES:
        print(f"  {code}: {pct(precision[code])}")
    for g in ("REVIEW", "PASS"):
        print(f"{g}: significant error {pct(by_group[g]['significant'])}, "
              f"any error {pct(by_group[g]['any'])}")
    print("PASS significant error types:", {t: n for t, n in pass_types["significant"].items() if n})
    print(f"clean tier: {tier_significant:.1%} of drafts have a significant error; "
          f"the gate sends {caught:.0%} of them to REVIEW (range {caught_range[0]:.0%}-"
          f"{caught_range[1]:.0%})")
    print("author agreement:", agreement)


def cohen_kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def author_agreement(items, path):
    """Agreement with the author's blind re-labels: {"item": int, "error": none|minor|significant}."""
    mine = {r["item"]: r["error"] for r in items}
    theirs = {r["item"]: r["error"] for r in map(json.loads, open(path))}
    keys = sorted(theirs)
    a, b = [mine[k] for k in keys], [theirs[k] for k in keys]
    sig_a, sig_b = [x == "significant" for x in a], [x == "significant" for x in b]
    return {"n": len(keys),
            "error_level_exact": round(sum(x == y for x, y in zip(a, b, strict=True)) / len(keys), 3),
            "error_level_kappa": cohen_kappa(a, b),
            "significant_agree": round(sum(x == y for x, y in zip(sig_a, sig_b, strict=True))
                                       / len(keys), 3),
            "significant_kappa": cohen_kappa(sig_a, sig_b)}


if __name__ == "__main__":
    main()
