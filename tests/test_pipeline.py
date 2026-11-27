"""Tests for laysum.pipeline and laysum.writer helpers. All report texts are self-written.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from laysum import pipeline, writer
from laysum.writer import Draft

ROOT = Path(__file__).resolve().parents[1]


class FakeWriter:
    version = "fake"

    def __init__(self, text, truncated_at=None):
        self.text, self.truncated_at, self.calls = text, truncated_at, []

    def summarise(self, report):
        self.calls.append(report)
        return Draft(self.text, self.truncated_at)


def test_faithful_draft_passes():
    result = pipeline.run("Cardiomegaly. No pneumothorax.",
                          FakeWriter("The heart is enlarged. There is no collapsed lung."))
    assert result["decision"] == "PASS"
    assert result["flags"] == []
    assert result["meta"] == {"model_version": "fake", "level": "sentence", "n_flags": 0}


def test_lost_negation_goes_to_review_with_spans():
    report = "Cardiomegaly. No pneumothorax."
    result = pipeline.run(report, FakeWriter("The heart is enlarged and the lung is collapsed."))
    assert result["decision"] == "REVIEW"
    (flag,) = result["flags"]
    assert flag["code"] == "NEGATION_LOST"
    s, e = flag["source_spans"][0]
    assert report[s:e] == "No pneumothorax."
    json.dumps(result)  # the result must be plain JSON for the API


def test_truncation_goes_to_review():
    report = "Mild cardiomegaly. " + "Stable finding. " * 3
    cut = len("Mild cardiomegaly. ")
    result = pipeline.run(report, FakeWriter("The heart is a little big.", truncated_at=cut))
    assert [f["code"] for f in result["flags"]] == ["INPUT_TRUNCATED"]
    assert result["decision"] == "REVIEW"


def test_junk_tail_is_cut_before_checking():
    # The glued-on second summary would hide the lost negation if it were not cut first.
    junk = 'The heart is enlarged.": "There is no collapsed lung.'
    result = pipeline.run("Cardiomegaly. No pneumothorax.", FakeWriter(junk))
    assert result["summary"] == "The heart is enlarged."
    assert [f["code"] for f in result["flags"]] == ["NEGATION_LOST"]


def test_document_level_is_passed_through():
    report = "No effusion. No pneumothorax."
    draft = FakeWriter("There is no fluid and no collapsed lung.")
    assert pipeline.run(report, draft, level="sentence")["decision"] == "REVIEW"
    assert pipeline.run(report, draft, level="document")["decision"] == "PASS"


@pytest.mark.parametrize("bad", ["", "   \n", None])
def test_empty_report_is_rejected_without_calling_the_writer(bad):
    fake = FakeWriter("x")
    with pytest.raises(ValueError, match="empty"):
        pipeline.run(bad, fake)
    assert fake.calls == []


def test_report_length_limit_is_inclusive():
    # Both sides of the boundary: a test far past it cannot tell > from >=.
    limit = pipeline.MAX_REPORT_CHARS
    fake = FakeWriter("x")
    pipeline.run("a" * limit, fake)
    assert len(fake.calls) == 1
    with pytest.raises(ValueError, match="limit"):
        pipeline.run("a" * (limit + 1), fake)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("length, seconds", [(1, 10), (1_499, 10), (1_500, 30), (20_000, 30)])
def test_gpu_reservation_switches_at_1500_characters(length, seconds):
    assert pipeline.gpu_seconds("a" * length) == seconds


def test_gate_is_any_flag():
    assert pipeline.gate([]) == "PASS"
    assert pipeline.gate([object()]) == "REVIEW"


class WordTokenizer:
    """One token per whitespace-separated word, with character offsets; enough for the cut."""

    def __call__(self, text, return_offsets_mapping=False):
        import re
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        ids = list(range(len(spans))) + [999]  # + </s>
        return {"input_ids": ids, "offset_mapping": spans + [(0, 0)]}


def test_truncation_point_with_a_word_tokenizer():
    tok = WordTokenizer()
    prompt_tokens = len(writer.TASK_PROMPT.split())
    report = "  " + " ".join(f"w{i}" for i in range(30)) + " "  # leading space is stripped
    assert writer.truncation_point(report, tok, max_tokens=10_000) is None
    max_tokens = prompt_tokens + 6  # the writer keeps 5 words of the report
    cut = writer.truncation_point(report, tok, max_tokens=max_tokens)
    assert report[cut:].startswith("w5")


def _training_module(name):
    path = ROOT / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"training_{name}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # torch or datasets missing
        pytest.skip(f"cannot import training/{name}.py: {exc}")
    return module


def test_inference_settings_match_the_training_code():
    dataset = _training_module("dataset")
    assert writer.TASK_PROMPT == dataset.TASK_PROMPT
    assert writer.MAX_SOURCE_TOKENS == dataset.MAX_SOURCE_LEN
    assert writer.MAX_NEW_TOKENS == dataset.MAX_TARGET_LEN
    assert writer.build_source_text("  x ") == dataset.build_source_text("  x ")


def test_base_revision_is_pinned_and_adapter_ships_with_the_repo():
    import re
    assert re.fullmatch(r"[0-9a-f]{40}", writer.BASE_REVISION)
    assert (writer.ADAPTER_DIR / "adapter_config.json").is_file()
    assert writer.adapter_sha256() == writer.ADAPTER_SHA256


def test_a_changed_adapter_is_refused(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"not the evaluated weights")
    with pytest.raises(ValueError, match="SHA-256"):
        writer.FlanT5Writer.from_pretrained(tmp_path)


def test_highlight_marks_flagged_sentences_and_keeps_the_text():
    report = "Cardiomegaly. No pneumothorax. Possible effusion."
    flags = [{"code": "NEGATION_LOST", "source_spans": [[14, 30]]},
             {"code": "CERTAINTY_LOST", "source_spans": [[31, 49]]},
             {"code": "INPUT_TRUNCATED", "source_spans": [[31, 49]]}]
    pieces = pipeline.highlight(report, flags)
    assert "".join(text for text, _ in pieces) == report
    assert ("No pneumothorax.", "NEGATION_LOST") in pieces
    assert ("Possible effusion.", "CERTAINTY_LOST + INPUT_TRUNCATED") in pieces
    assert pieces[0] == ("Cardiomegaly. ", None)


def test_highlight_without_flags_is_one_plain_piece():
    assert pipeline.highlight("No pneumothorax.", []) == [("No pneumothorax.", None)]


def test_examples_are_unique_and_the_app_ones_exist():
    from laysum.examples import APP_EXAMPLES, CASES, by_name

    assert len({c.name for c in CASES}) == len({c.report for c in CASES}) == len(CASES)
    for name in APP_EXAMPLES:
        assert by_name(name).decision in ("PASS", "REVIEW")
