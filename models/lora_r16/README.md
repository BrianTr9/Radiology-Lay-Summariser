# FLAN-T5-base LoRA (r = 16) for plain-language chest X-ray report drafts

A LoRA adapter for `google/flan-t5-base` that rewrites a chest X-ray report as a short
plain-language draft. It is the writer of this repository's research prototype (see the
[main README](../../README.md)), where every draft is checked by rules and meant to be
reviewed by a clinician.

| File | Contents |
|---|---|
| `adapter_config.json` | LoRA settings (r = 16, alpha = 32, query and value matrices) |
| `adapter_model.safetensors` | 1,769,472 weights, 7.1 MB, SHA-256 `273ae336215a...` (checked when loaded) |

License: research use only, following the terms of the training data (see below); the
MIT license of this repository covers the code, not these weights.

> **Research use only.** This is not a medical device. Do not use it for diagnosis, patient
> care, or to send text to patients without review. Do not enter reports that contain
> personal information. The training data's terms prohibit sale and require approval for
> non-academic use, so do not use this adapter commercially.

## Use

The repository's `FlanT5Writer` loads the base model, applies this adapter and merges it,
with the settings every number below was measured with (fixed prompt, report cut to 512
tokens, 4 beams, at most 512 new tokens):

```python
from laysum.writer import FlanT5Writer

writer = FlanT5Writer.from_pretrained()   # base from the Hub at a pinned commit + this adapter
draft = writer.summarise("Small left-sided pneumothorax. No pleural effusion.")
print(draft.text)
```

Without that code, load this folder with `peft.PeftModel.from_pretrained` and prefix the
report with: *"Rewrite the following radiology report in plain language that a patient can
understand:"*. With transformers 5.x, share the encoder and decoder embeddings with
`model.shared` before loading the adapter (`model.encoder.embed_tokens.weight =
model.shared.weight`, same for the decoder); otherwise FLAN-T5's untied-embedding config
leaves them randomly initialised.

## Training

| | |
|---|---|
| Data | BioLaySumm 2025 LaymanRRG, open-source track: 150,454 training pairs from PadChest, BIMCV-COVID19 and Open-i; lay summaries written by GPT-4o (silver labels) |
| Method | LoRA on the attention query and value matrices, r = 16, alpha = 32, dropout 0.05; 1,769,472 trainable parameters (0.71% of 247.6M) |
| Optimisation | 3 epochs, learning rate 3e-3, batch 8 with 2 accumulation steps, warm-up 5%, weight decay 0.01, bf16 autocast, seed 0 |
| Selection | Last epoch, which also had the best score on a 1,000-report development set cut out of the training data |
| Hardware | One A100-PCIE-40GB, 2.4 hours, 15.1 GiB peak memory |
| Libraries | transformers 5.19.0, peft 0.21.2, torch 2.13.0 |

## Evaluation

ROUGE-Lsum on the official validation split (10,000 reports), beam search with 4 beams.
The official validation split shares 45% of its reports with the training data, so scores
are also reported on a subset without that overlap.

| ROUGE-Lsum | All validation (10,000) | Report not in training (5,494) | Clean: report and imaging study not in training (4,037) |
|---|---|---|---|
| Zero-shot FLAN-T5-base | 29.3 | 30.3 | 30.3 |
| **This adapter (LoRA r = 16)** | **71.3** | **67.4** | **67.2** |
| Full fine-tuning (all weights) | 73.6 | 70.4 | 70.2 |

Reading level (Flesch-Kincaid grade, approximate): source reports 14.4, this adapter 7.5,
GPT-4o references 7.5.

These numbers were measured on the unmerged adapter in batches. The repository's
`eval/equivalence.py` checks that the merged, one-report-at-a-time writer reproduces them.
The official test split has no public references, so nothing here is a test-set result and
none of it should be compared with the shared-task leaderboard.

## Limitations

- **References are not ground truth.** They were written by GPT-4o and sometimes differ
  from the source report. ROUGE counts shared words, so a high score can hide a clinical
  error: an inverted negation can score higher than a correct paraphrase.
- **Known failure modes.** The model can add content that is not in the report, change the
  wording of a finding, drop a negation, or drop a hedge such as "likely". It sometimes
  copies the source almost verbatim, and 31 of 10,000 validation drafts carry a second
  summary glued on after a `": "` separator (the repository removes it). Reports longer
  than 512 tokens are cut, so their end is never read.
- **Scope.** Chest X-ray reports in English; the Spanish PadChest and BIMCV-COVID19 reports
  were translated into English before the lay summaries were written. Reports outside this distribution are untested.
- **Review flags are partial.** The rule checks in the repository catch lost negation, lost
  hedges and truncated input; they cannot catch added content.

## Data terms and citation

The data comes from sources that allow research use only and forbid redistribution; no
report text is included here. If you use this adapter, cite the data sources and the task:

- Bustos et al., *PadChest: A large chest x-ray image dataset with multi-label annotated
  reports*, 2020 ([arXiv:1901.07441](https://arxiv.org/abs/1901.07441)).
- de la Iglesia Vayá et al., *BIMCV COVID-19+: a large annotated dataset of RX and CT images
  from COVID-19 patients*, 2020 ([arXiv:2006.01174](https://arxiv.org/abs/2006.01174)).
- Demner-Fushman et al., 2016, for the Open-i chest X-ray collection.
- Zhao et al., *X-ray made simple: lay radiology report generation and robust evaluation*
  ([arXiv:2406.17911](https://arxiv.org/abs/2406.17911)), which created the lay reports.
- Xiao et al., *Overview of the BioLaySumm 2025 Shared Task* (BioNLP 2025).

None of these authors or institutions endorses this model.
