"""Self-written chest X-ray reports shared by the app, the golden tests and the latency check.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

None of these texts comes from a dataset (the data terms forbid publishing any part of it);
they were checked against the training and validation data for shared 8-word phrases.
Each case has the decision the gate gives with the shipped model on CPU (tests/test_golden.py).

Two cases go to REVIEW although the draft is faithful. They are kept on purpose, as the known
limits of word-list flags: a lay paraphrase of a hedge the list does not know, and negated
findings merged into fewer sentences. A flag is a reason to look, not proof of an error.
"""
from typing import List, NamedTuple, Tuple


class Case(NamedTuple):
    name: str
    report: str
    decision: str
    codes: Tuple[str, ...]


LONG_REPORT = " ".join([
    "Compared with the earlier film.",
    "A right-sided central line ends in the lower superior vena cava.",
    "The mediastinal outline is unchanged.",
    "Slightly increased interstitial markings at both bases.",
    "Mild wear-and-tear change of the thoracic vertebrae.",
] * 12) + " Small left pleural effusion."  # past the 512-token limit, so never read

CASES: List[Case] = [
    Case("normal", "Both lungs are clear. Normal heart size. No pleural effusion or pneumothorax.",
         "PASS", ()),
    Case("finding_and_negation", "Cardiomegaly. No pneumothorax.", "PASS", ()),
    Case("hedge_kept",
         "Patchy opacity in the right lower lobe, likely pneumonia. No pleural effusion.",
         "PASS", ()),
    Case("devices",
         "The endotracheal tube ends 4 cm above the carina. A feeding tube passes below the "
         "diaphragm.",
         "PASS", ()),
    Case("post_operative",
         "Median sternotomy wires. Mild cardiomegaly. Pulmonary vascular congestion without "
         "frank edema.",
         "PASS", ()),
    # Known false positive: the draft says "can't completely rule out", which the list lacks.
    Case("hedge_paraphrased",
         "Small nodule in the left upper lobe; malignancy cannot be excluded. Follow-up CT is "
         "recommended.",
         "REVIEW", ("CERTAINTY_LOST",)),
    # Known false positive: the draft merges two negated findings into one sentence.
    Case("negations_merged", "No consolidation. No effusion. No pneumothorax. No cardiomegaly.",
         "REVIEW", ("NEGATION_LOST",)),
    Case("too_long", LONG_REPORT, "REVIEW", ("INPUT_TRUNCATED",)),
]

APP_EXAMPLES = ("hedge_kept", "hedge_paraphrased", "post_operative")


def by_name(name: str) -> Case:
    return next(case for case in CASES if case.name == name)
