"""The writer: FLAN-T5-base with a merged LoRA adapter that drafts a plain-language summary.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Inference settings are the ones every reported number was measured with (training/ code):
the fixed task prompt before the report, the report cut to 512 tokens, beam search with
4 beams, at most 512 new tokens, bf16 autocast on GPU and fp32 on CPU.

torch, transformers and peft are imported when a writer is built, so the rest of the
package (checks, pipeline) and its tests need none of them.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

# Every model reference lives here. The base model downloads from the Hugging Face Hub at a
# pinned commit (unchanged since 2023-07-17); the adapter ships with this repository and is
# checked against its SHA-256 when loaded.
BASE_MODEL = "google/flan-t5-base"
BASE_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
ADAPTER_DIR = Path(__file__).resolve().parents[2] / "models" / "lora_r16"
ADAPTER_SHA256 = "273ae336215af927b5f375bb5af0fa37a7748cd9579f9adcb5ef0082ac8f6822"
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


def adapter_sha256(adapter_dir: Path = ADAPTER_DIR) -> str:
    return hashlib.sha256((Path(adapter_dir) / "adapter_model.safetensors").read_bytes()).hexdigest()


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
    def from_pretrained(cls, adapter_dir: Path = ADAPTER_DIR, device: Optional[str] = None
                        ) -> "FlanT5Writer":
        """Load the pinned base model, apply the adapter in `adapter_dir` and merge it.

        Refuses an adapter whose weights do not match ADAPTER_SHA256, so every draft comes
        from the model that was evaluated.
        """
        digest = adapter_sha256(adapter_dir)
        if digest != ADAPTER_SHA256:
            raise ValueError(f"adapter in {adapter_dir} has SHA-256 {digest[:12]}..., "
                             f"expected {ADAPTER_SHA256[:12]}...")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        # transformers 5.x leaves the encoder and decoder embeddings randomly initialised when
        # the config sets tie_word_embeddings = False (FLAN-T5 does); sharing them explicitly
        # restores the trained matrix. Same fix as training/modules.py:_load_flan_t5.
        model.encoder.embed_tokens.weight = model.shared.weight
        model.decoder.embed_tokens.weight = model.shared.weight
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload().to(device).eval()
        version = f"{BASE_MODEL}@{BASE_REVISION[:12]} + lora_r16 sha256:{digest[:12]}"
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
