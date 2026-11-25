"""Report in, draft and review decision out.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

    report -> writer -> cut the glued-on tail -> rule flags -> gate (PASS | REVIEW)

The gate is the one rule from the roadmap: any flag means REVIEW. It never logs or stores
the report or the draft.
"""
from typing import Any, Dict, List

from .checks import Flag, check, strip_junk_tail
from .writer import Writer

MAX_REPORT_CHARS = 20_000  # about 5,000 tokens; the writer reads 512 and flags the rest


def gate(flags: List[Flag]) -> str:
    return "REVIEW" if flags else "PASS"


def flag_to_dict(flag: Flag) -> Dict[str, Any]:
    return {"code": flag.code, "message": flag.message,
            "source_spans": [list(span) for span in flag.source_spans]}


def run(report: str, writer: Writer, level: str = "sentence") -> Dict[str, Any]:
    """Draft a summary of `report` and decide whether it needs review.

    Returns {"summary", "decision", "flags", "meta"}; raises ValueError for an empty or
    over-long report.
    """
    if not report or not report.strip():
        raise ValueError("The report is empty.")
    if len(report) > MAX_REPORT_CHARS:
        raise ValueError(f"The report has {len(report):,} characters; the limit is "
                         f"{MAX_REPORT_CHARS:,}.")
    draft = writer.summarise(report)
    summary = strip_junk_tail(draft.text)
    flags = check(report, summary, draft.truncated_at, level)
    return {"summary": summary, "decision": gate(flags),
            "flags": [flag_to_dict(f) for f in flags],
            "meta": {"model_version": writer.version, "level": level, "n_flags": len(flags)}}
