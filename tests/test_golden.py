"""Golden tests: the real writer and gate on the self-written cases in laysum.examples.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Off by default. Run with:  LAYSUM_GOLDEN=1 pytest tests/test_golden.py
Needs the pinned base model (downloaded once, about 1 GB) and about 2 GiB of RAM; runs on CPU
in about a minute. The expected decisions were recorded with the model version below on CPU
(fp32); a GPU in bf16 can word a draft differently, so these tests pin the CPU path.
"""
import os

import pytest

from laysum.examples import CASES, LONG_REPORT
from laysum.pipeline import run
from laysum.writer import BASE_REVISION, FlanT5Writer

pytestmark = pytest.mark.skipif(os.environ.get("LAYSUM_GOLDEN") != "1",
                                reason="set LAYSUM_GOLDEN=1 to run the real model")

MODEL_VERSION = f"google/flan-t5-base@{BASE_REVISION[:12]} + lora_r16 sha256:273ae336215a"


@pytest.fixture(scope="module")
def writer():
    return FlanT5Writer.from_pretrained(device="cpu")


def test_model_version_is_the_one_the_expectations_were_recorded_with(writer):
    assert writer.version == MODEL_VERSION


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_golden_decision(writer, case):
    result = run(case.report, writer)
    assert result["summary"].strip()
    assert [f["code"] for f in result["flags"]] == list(case.codes), result["summary"]
    assert result["decision"] == case.decision


def test_truncation_points_at_the_unread_end(writer):
    result = run(LONG_REPORT, writer)
    (flag,) = [f for f in result["flags"] if f["code"] == "INPUT_TRUNCATED"]
    ((start, end),) = flag["source_spans"]
    assert end == len(LONG_REPORT)
    assert "Small left pleural effusion." in LONG_REPORT[start:end]
