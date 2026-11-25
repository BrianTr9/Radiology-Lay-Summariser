# Radiology Lay Summariser

A research prototype that drafts a plain-language summary of a chest X-ray report for a
clinician to review, and points out where the draft may have lost a negation, lost the
report's degree of certainty, or missed the end of a long report.

> **Status: work in progress.** The writer model is trained and evaluated; the review
> flags are written and measured; the demo app is not built yet.
>
> **Research use only.** Not a medical device and not for diagnosis or patient care.
> Do not paste reports that contain personal information.

## How it works

```
report ─► writer: FLAN-T5-base + LoRA (r = 16) ─► rule-based flags ─► PASS | REVIEW
```

- **Writer.** FLAN-T5-base (248M parameters) fine-tuned with LoRA on 150k report and
  lay-summary pairs; only 0.7% of the parameters are trained.
- **Flags.** `INPUT_TRUNCATED` (the report is longer than the 512 tokens the writer reads),
  `NEGATION_LOST` and `CERTAINTY_LOST` (the draft has fewer negated or hedged sentences
  than the report). Any flag sends the draft to REVIEW. The rules need no model, so each
  flag points at the exact report sentences it is about.
- **Not covered.** The flags cannot detect content the draft adds that is not in the
  report. A neural grounding checker was tested for this in a small test with criteria
  fixed in advance, and rejected: it could not tell a correct plain-language paraphrase
  from an error.

## Results so far

![ROUGE-Lsum by validation tier, training curves, memory and generation time](training/figures/results.png)

- The official validation split shares 45% of its reports with training, so the writer
  is also scored on a **clean tier** (4,037 rows whose report and imaging study never
  appear in training). On that tier LoRA reaches ROUGE-Lsum 67.2, against 30.3 zero-shot
  and 70.2 for full fine-tuning.
- On the clean tier the flags send **6.8%** of the LoRA drafts to REVIEW
  ([`eval/results/review_rate.json`](eval/results/review_rate.json)). Whether those flags
  are right is measured next, by reading a seeded sample by hand.

## Repository layout

| Path | Contents |
|---|---|
| `training/` | Data loading, LoRA and full fine-tuning, evaluation tiers, Slurm scripts |
| `src/laysum/` | The product code (`checks.py` today; writer and pipeline next) |
| `eval/` | Scripts that produce every number in this README |
| `tests/` | Unit tests; all example texts are self-written |

## Data

[BioLaySumm 2025 LaymanRRG, open-source track](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track):
reports from PadChest ([Bustos et al., 2020](https://arxiv.org/abs/1901.07441)),
BIMCV-COVID19+ ([de la Iglesia Vayá et al., 2020](https://arxiv.org/abs/2006.01174)) and
Open-i, with lay summaries written by GPT-4o. The source datasets forbid redistribution,
so **no report, summary or excerpt from the data is stored in this repository**; scripts
download the data from the Hugging Face Hub when run.

## License

Code: [MIT](LICENSE). Model weights and data follow the terms of their sources.

## AI assistance

Parts of the code and documentation were drafted with Claude Code and reviewed by the
author.
