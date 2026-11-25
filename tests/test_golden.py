"""Golden tests: the real writer and gate on self-written reports. Off by default.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Run with:  LAYSUM_GOLDEN=1 pytest tests/test_golden.py
Needs the pinned base model (downloaded once, about 1 GB) and about 2 GiB of RAM; runs on CPU
in about a minute. The expected decisions were recorded with the model version below on CPU
(fp32); a GPU in bf16 can word a draft differently, so these tests pin the CPU path.

Two cases go to REVIEW although the draft is faithful. They are here on purpose, as the
known limits of word-list flags: a lay paraphrase of a hedge the list does not know, and
three negated sentences merged into two. A flag is a reason to look, not proof of an error.
"""
import os

import pytest

from laysum.pipeline import run
from laysum.writer import BASE_REVISION, FlanT5Writer

pytestmark = pytest.mark.skipif(os.environ.get("LAYSUM_GOLDEN") != "1",
                                reason="set LAYSUM_GOLDEN=1 to run the real model")

MODEL_VERSION = f"google/flan-t5-base@{BASE_REVISION[:12]} + lora_r16 sha256:273ae336215a"

LONG_REPORT = " ".join([
    "Compared with the earlier film.",
    "A right-sided central line ends in the lower superior vena cava.",
    "The mediastinal outline is unchanged.",
    "Slightly increased interstitial markings at both bases.",
    "Mild wear-and-tear change of the thoracic vertebrae.",
] * 12) + " Small left pleural effusion."  # past the 512-token limit, so never read

# (name, report, expected decision, expected flag codes)
GOLDEN = [
    ("normal",
     "Both lungs are clear. Normal heart size. No pleural effusion or pneumothorax.",
     "PASS", []),
    ("finding_and_negation", "Cardiomegaly. No pneumothorax.", "PASS", []),
    ("hedge_kept",
     "Patchy opacity in the right lower lobe, likely pneumonia. No pleural effusion.",
     "PASS", []),
    ("devices",
     "The endotracheal tube ends 4 cm above the carina. A feeding tube passes below the "
     "diaphragm.",
     "PASS", []),
    ("post_operative",
     "Median sternotomy wires. Mild cardiomegaly. Pulmonary vascular congestion without "
     "frank edema.",
     "PASS", []),
    # Known false positive: the draft says "can't completely rule out", which the list lacks.
    ("hedge_paraphrased",
     "Small nodule in the left upper lobe; malignancy cannot be excluded. Follow-up CT is "
     "recommended.",
     "REVIEW", ["CERTAINTY_LOST"]),
    # Known false positive: the draft merges two negated findings into one sentence.
    ("negations_merged",
     "No consolidation. No effusion. No pneumothorax. No cardiomegaly.",
     "REVIEW", ["NEGATION_LOST"]),
    ("too_long", LONG_REPORT, "REVIEW", ["INPUT_TRUNCATED"]),
]


@pytest.fixture(scope="module")
def writer():
    return FlanT5Writer.from_pretrained(device="cpu")


def test_model_version_is_the_one_the_expectations_were_recorded_with(writer):
    assert writer.version == MODEL_VERSION


@pytest.mark.parametrize("name, report, decision, codes", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden_decision(writer, name, report, decision, codes):
    result = run(report, writer)
    assert result["summary"].strip()
    assert [f["code"] for f in result["flags"]] == codes, result["summary"]
    assert result["decision"] == decision


def test_truncation_points_at_the_unread_end(writer):
    result = run(LONG_REPORT, writer)
    (flag,) = [f for f in result["flags"] if f["code"] == "INPUT_TRUNCATED"]
    (start, end), = flag["source_spans"]
    assert end == len(LONG_REPORT)
    assert "Small left pleural effusion." in LONG_REPORT[start:end]
