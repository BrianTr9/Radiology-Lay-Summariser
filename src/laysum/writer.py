"""The writer: FLAN-T5-base with a merged LoRA adapter that drafts a plain-language summary.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Inference settings are the ones every reported number was measured with (training/ code):
the fixed task prompt before the report, the report cut to 512 tokens, beam search with
4 beams, at most 512 new tokens, bf16 autocast on GPU and fp32 on CPU.

torch, transformers and peft are imported when a writer is built, so the rest of the
package (checks, pipeline) and its tests need none of them.
"""
from dataclasses import dataclass
from typing import Optional, Protocol

BASE_MODEL = "google/flan-t5-base"
TASK_PROMPT = ("Rewrite the following radiology report in plain language that a patient can "
               "understand:")
MAX_SOURCE_TOKENS = 512
MAX_NEW_TOKENS = 512
NUM_BEAMS = 4


@dataclass(frozen=True)
class Draft:
    text: str
    truncated_at: Optional[int]  # character offset in the report where reading stopped, or None


class Writer(Protocol):
    version: str

    def summarise(self, report: str) -> Draft: ...


def build_source_text(report: str) -> str:
    return f"{TASK_PROMPT} {report.strip()}"


def truncation_point(report: str, tokenizer, max_tokens: int = MAX_SOURCE_TOKENS) -> Optional[int]:
    """Character offset in `report` where the writer stops reading, or None if it reads all.

    Matches tokenizer(prompt + report, max_length=max_tokens, truncation=True), which keeps
    max_tokens - 1 text tokens and then appends </s>.
    """
    enc = tokenizer(build_source_text(report), return_offsets_mapping=True)
    if len(enc["input_ids"]) <= max_tokens:
        return None
    report_start = len(TASK_PROMPT) + 1                       # where the report begins in the input
    stripped_lead = len(report) - len(report.lstrip())        # build_source_text strips the report
    return enc["offset_mapping"][max_tokens - 1][0] - report_start + stripped_lead


class FlanT5Writer:
    """FLAN-T5-base plus a LoRA adapter, merged into the base weights."""

    def __init__(self, model, tokenizer, version: str, device: str):
        self.model, self.tokenizer, self.version, self.device = model, tokenizer, version, device

    @classmethod
    def from_pretrained(cls, adapter: str, base: str = BASE_MODEL,
                        base_revision: Optional[str] = None,
                        adapter_revision: Optional[str] = None,
                        device: Optional[str] = None) -> "FlanT5Writer":
        """Load the base model, apply the adapter at `adapter` (a folder or a Hub id) and merge."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(base, revision=base_revision)
        model = AutoModelForSeq2SeqLM.from_pretrained(base, revision=base_revision)
        # transformers 5.x leaves the encoder and decoder embeddings randomly initialised when
        # the config sets tie_word_embeddings = False (FLAN-T5 does); sharing them explicitly
        # restores the trained matrix. Same fix as training/modules.py:_load_flan_t5.
        model.encoder.embed_tokens.weight = model.shared.weight
        model.decoder.embed_tokens.weight = model.shared.weight
        model = PeftModel.from_pretrained(model, adapter, revision=adapter_revision)
        model = model.merge_and_unload().to(device).eval()
        version = f"{base}@{base_revision or 'unpinned'} + {adapter}@{adapter_revision or 'unpinned'}"
        return cls(model, tokenizer, version, device)

    def summarise(self, report: str) -> Draft:
        import torch

        cut = truncation_point(report, self.tokenizer)
        batch = self.tokenizer(build_source_text(report), max_length=MAX_SOURCE_TOKENS,
                               truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16,
                                             enabled=self.device == "cuda"):
            out = self.model.generate(**batch, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS)
        text = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        return Draft(text=text, truncated_at=cut)
