"""Rule-based checks between a radiology report and its plain-language draft.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Three flags, each tied to an error seen in the error audit of the writer (training/autopsy.py):
  INPUT_TRUNCATED  the writer did not read the end of the report;
  CERTAINTY_LOST   the draft has fewer hedged sentences than the report;
  NEGATION_LOST    the draft has fewer negated sentences than the report.
Any flag means the draft needs review. The checks need no model, so they are deterministic
and every flag can be explained by pointing at the report sentences it is about.

What the checks cannot see: content the draft adds that is not in the report. A
lay paraphrase can also hide a negation ("no consolidation" -> "the lungs are clear"), so a
flag is a reason to look, not proof of an error.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Span = Tuple[int, int]

# Word lists from the error audit (training/autopsy.py), plus a few lay words fixed
# before any rate was computed ("none", "nothing", "never"; "maybe", "perhaps", "seem(s)",
# "unclear", "uncertain", "not sure").
NEGATION = ("no", "not", "without", "absent", "negative", "neither", "nor", "free of", "n't",
            "none", "nothing", "never")
HEDGE = ("possible", "possibly", "probable", "probably", "may", "might", "could", "likely",
         "suspected", "suspicious", "suggest", "suggests", "suggestive",
         "cannot be excluded", "cannot be ruled out", "can't be ruled out", "not excluded",
         "maybe", "perhaps", "seem", "seems", "unclear", "uncertain", "not sure")

LEVELS = ("sentence", "document")


def _cue_pattern(words: Sequence[str]) -> "re.Pattern[str]":
    # Whole words only, except "n't", which is glued to the verb ("isn't").
    parts = [re.escape(w) if w == "n't" else r"(?<!\w)" + re.escape(w) + r"(?!\w)"
             for w in words]
    return re.compile("|".join(parts), re.IGNORECASE)


NEGATION_RE = _cue_pattern(NEGATION)
HEDGE_RE = _cue_pattern(HEDGE)

# A sentence ends at . ! or ? followed by whitespace or the end of the text, so "1.5 cm"
# is not split.
_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)


@dataclass(frozen=True)
class Flag:
    code: str
    message: str
    source_spans: Tuple[Span, ...]  # character spans in the report that the flag is about


def sentence_spans(text: str) -> List[Span]:
    """Character spans of the sentences in `text`."""
    return [m.span() for m in _SENTENCE_RE.finditer(text)]


def _sentences_with(pattern: "re.Pattern[str]", text: str) -> List[Span]:
    return [(s, e) for s, e in sentence_spans(text) if pattern.search(text[s:e])]


def _lost(n_source: int, n_summary: int, level: str) -> bool:
    if level == "sentence":
        return n_summary < n_source
    if level == "document":
        return n_source > 0 and n_summary == 0
    raise ValueError(f"level must be one of {LEVELS}, got {level!r}")


def check(source: str, summary: str, truncated_at: Optional[int] = None,
          level: str = "sentence") -> List[Flag]:
    """Flags for one (report, draft) pair; an empty list means PASS.

    truncated_at: character offset in `source` where the writer stopped reading, or None if
        it read the whole report. Only the part the writer read is checked for lost cues,
        so a cut-off sentence raises INPUT_TRUNCATED and nothing else.
    level: "sentence" flags when the draft has fewer cue sentences than the report;
        "document" flags only when the report has a cue and the draft has none.
    """
    flags: List[Flag] = []
    read = source
    if truncated_at is not None and truncated_at < len(source):
        read = source[:truncated_at]
        flags.append(Flag("INPUT_TRUNCATED",
                          "The report is longer than the writer reads; the end was not used.",
                          ((truncated_at, len(source)),)))
    for code, pattern, what in (("NEGATION_LOST", NEGATION_RE, "negated"),
                                ("CERTAINTY_LOST", HEDGE_RE, "hedged")):
        in_source = _sentences_with(pattern, read)
        n_summary = len(_sentences_with(pattern, summary))
        if _lost(len(in_source), n_summary, level):
            flags.append(Flag(code, f"The report has {len(in_source)} {what} sentence(s); "
                                    f"the draft has {n_summary}.", tuple(in_source)))
    return flags


def strip_junk_tail(summary: str) -> str:
    """Drop a second summary glued on after '": "' (440 training references have one, and
    the fine-tuned models learned it: 31 of 10,000 LoRA r16 validation outputs)."""
    return summary.split('": "', 1)[0].rstrip()
