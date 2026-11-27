"""Report in, draft and review decision out.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

    report -> writer -> cut the glued-on tail -> rule flags -> gate (PASS | REVIEW)

The gate is the one rule from the roadmap: any flag means REVIEW. It never logs or stores
the report or the draft.
"""
from itertools import pairwise
from typing import Any, Dict, List, Optional, Tuple

from .checks import Flag, check, strip_junk_tail
from .writer import Writer

MAX_REPORT_CHARS = 20_000  # about 5,000 tokens; the writer reads 512 and flags the rest
SHORT_REPORT_CHARS = 1_500  # below this a report gets the short GPU reservation


def gate(flags: List[Flag]) -> str:
    return "REVIEW" if flags else "PASS"


def flag_to_dict(flag: Flag) -> Dict[str, Any]:
    return {"code": flag.code, "message": flag.message,
            "source_spans": [list(span) for span in flag.source_spans]}


def validate_report(report: str) -> None:
    """Raise ValueError for an empty or over-long report."""
    if not report or not report.strip():
        raise ValueError("The report is empty.")
    if len(report) > MAX_REPORT_CHARS:
        raise ValueError(f"The report has {len(report):,} characters; the limit is "
                         f"{MAX_REPORT_CHARS:,}.")


def gpu_seconds(report: str) -> int:
    """GPU time for the app to reserve on ZeroGPU, which charges the reservation against the
    visitor's daily quota. Typical reports draft in under 2 s on the Space; long ones write
    long drafts. Kept here, not in app.py, so it can be tested without loading the model."""
    return 10 if len(report) < SHORT_REPORT_CHARS else 30


def run(report: str, writer: Writer, level: str = "sentence") -> Dict[str, Any]:
    """Draft a summary of `report` and decide whether it needs review.

    Returns {"summary", "decision", "flags", "meta"}; raises ValueError for an empty or
    over-long report.
    """
    validate_report(report)
    draft = writer.summarise(report)
    summary = strip_junk_tail(draft.text)
    flags = check(report, summary, draft.truncated_at, level)
    return {"summary": summary, "decision": gate(flags),
            "flags": [flag_to_dict(f) for f in flags],
            "meta": {"model_version": writer.version, "level": level, "n_flags": len(flags)}}


def highlight(report: str, flags: List[Dict[str, Any]]) -> List[Tuple[str, Optional[str]]]:
    """Split `report` into (text, label) pieces for display: label is the flag codes that
    point at that piece, joined with " + ", or None. Pieces cover the report in order."""
    cuts = {0, len(report)}
    for flag in flags:
        for start, end in flag["source_spans"]:
            cuts.update((start, end))
    edges = sorted(c for c in cuts if 0 <= c <= len(report))
    pieces: List[Tuple[str, Optional[str]]] = []
    for start, end in pairwise(edges):
        codes = [f["code"] for f in flags
                 if any(s <= start and end <= e for s, e in f["source_spans"])]
        pieces.append((report[start:end], " + ".join(codes) or None))
    return pieces
