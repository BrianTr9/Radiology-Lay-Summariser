"""Unit tests for laysum.checks. All texts are self-written, not taken from the dataset.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.
"""
import pytest

from laysum.checks import check, sentence_spans, strip_junk_tail


def codes(flags):
    return [f.code for f in flags]


def test_sentence_split_keeps_decimals():
    text = "A 1.5 cm nodule. No effusion! Is it new? Stable"
    assert [text[s:e] for s, e in sentence_spans(text)] == [
        "A 1.5 cm nodule.", "No effusion!", "Is it new?", "Stable"]


def test_faithful_draft_passes():
    source = "Opacity in the right lower lobe, likely pneumonia. No pleural effusion."
    summary = "There is a cloudy area in the lower right lung, probably pneumonia. There is no fluid."
    assert check(source, summary) == []


def test_dropped_likely_is_certainty_lost():
    source = "Infiltrate likely of infectious origin."
    flags = check(source, "There is an infection in the lung.")
    assert codes(flags) == ["CERTAINTY_LOST"]
    assert flags[0].source_spans == ((0, len(source)),)


def test_added_hedge_is_not_flagged():
    assert check("Cardiomegaly.", "The heart may be enlarged.") == []


def test_dropped_negation_is_negation_lost():
    source = "Cardiomegaly. No pneumothorax."
    flags = check(source, "The heart is enlarged and there is a collapsed lung.")
    assert codes(flags) == ["NEGATION_LOST"]
    s, e = flags[0].source_spans[0]
    assert source[s:e] == "No pneumothorax."


def test_contraction_counts_as_negation():
    assert check("No fracture.", "The bones aren't broken.") == []


def test_whole_words_only():
    # "nodule" and "know" contain "no"; "mayor" contains "may".
    assert check("A nodule. The mayor.", "There is a small spot.") == []


def test_merged_sentences_differ_by_level():
    source = "No effusion. No pneumothorax."
    summary = "There is no fluid and no collapsed lung."
    assert codes(check(source, summary, level="sentence")) == ["NEGATION_LOST"]
    assert check(source, summary, level="document") == []


def test_truncation_flag_and_unread_part_ignored():
    source = "Mild cardiomegaly. Mass likely malignant."
    cut = source.index("Mass")
    flags = check(source, "The heart is a little big.", truncated_at=cut)
    assert codes(flags) == ["INPUT_TRUNCATED"]
    assert flags[0].source_spans == ((cut, len(source)),)


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        check("No effusion.", "", level="paragraph")


def test_strip_junk_tail():
    assert strip_junk_tail('The lungs are clear.": "The lungs are clear.') == "The lungs are clear."
    assert strip_junk_tail("The lungs are clear.") == "The lungs are clear."
