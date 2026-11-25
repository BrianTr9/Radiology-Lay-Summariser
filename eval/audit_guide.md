# Audit guide: how the drafts are labelled

Written before any draft was read. The labels measure two things: whether the review flags
point at real problems, and how often a draft that passes still contains an error.

## Sample

`eval/review_rate.py --sample 100 --seed 0` draws, from the LoRA r16 drafts of the clean
validation tier (4,037 reports), 100 drafts the gate sends to REVIEW and 100 it lets PASS,
and shuffles them into one list. The texts go to a private file outside the repository (the
data terms forbid publishing them); the repository keeps only row indices and labels.

## Who labels

The labels were assigned by an AI model (Claude, Anthropic), reading each pair in full, one
at a time, following this guide. The rater is not a clinician. The author re-labels a
seeded subset of 40 pairs without seeing the model's labels, and the agreement is reported
with the results. Numbers from this audit are estimates of how a careful lay reader judges
the drafts, not a clinical evaluation.

## Pass A: errors (all 200 pairs, without seeing flags or group)

Compare the draft with the **source report**, not with the GPT-4o reference (the references
contain their own errors). Judge only meaning, not style. For each pair record the error
types found, each as **significant** (a patient would come away with a different clinical
picture) or **minor** (imprecise, but the picture is the same), or `none`.

| Type | The draft... | Example (self-written) |
|---|---|---|
| `ADDED` | states a finding, cause, diagnosis or reassurance the report does not | "opacity" becomes "pneumonia"; "nothing else is wrong" added |
| `NEGATION` | turns an absent finding into a present one, or a present one into absent | "no effusion" becomes "fluid around the lung" |
| `CERTAINTY` | states a hedged finding as certain, or a certain one as possible | "possible nodule" becomes "there is a nodule" |
| `CHANGED` | gets a finding's side, place, size, severity, change over time or device position wrong | "left" becomes "right"; "improved" becomes "worse" |
| `OMITTED` | leaves out an abnormal finding or a device that the report mentions | "small effusion" not in the draft |

| `DEGENERATE` | repeats itself in a loop or turns into text that says nothing about the report (amendment, see below) | one sentence repeated until the length limit |

**Amendment, made after reading items 0 to 19 and before any later item:** two drafts looped
(one repeated a correct paragraph, one repeated an empty sentence and lost the report's
conclusion). The table had no type for this, so `DEGENERATE` was added: **minor** when the
report's content is all there and only repeated, **significant** when the loop replaces it.

Not errors:
- plain-language wording that keeps the meaning ("pleural effusion" as "fluid around the
  lungs", "cardiomegaly" as "enlarged heart");
- leaving out technique, comparison or other administrative lines;
- leaving out normal or negative statements, as long as nothing contradicts them;
- jargon copied unchanged (a readability problem, noted as `NOT_PLAIN` but not an error).

Leaving out an abnormal finding counts as `OMITTED` even when it sits past the point where
the writer stopped reading: the patient still does not learn about it.

The decision rule for an item: **significant error** if any type is significant; **minor
error** if only minor types; **no error** otherwise. The headline numbers use significant
errors; minor ones are reported beside them.

## Pass B: is each flag right? (REVIEW pairs only, after pass A)

For each flag the gate raised, read the report sentences it points at and decide:

| Flag | Correct when |
|---|---|
| `NEGATION_LOST` | at least one negated finding in those sentences is missing from the draft or turned positive |
| `CERTAINTY_LOST` | at least one hedged finding in those sentences is stated as certain, or missing from the draft |
| `INPUT_TRUNCATED` | the unread end of the report holds an abnormal finding or device the draft does not mention |

Otherwise the flag is a **false alarm** (for example, negations merged into fewer sentences
or a hedge reworded in words the list does not know). A flag is also marked **useful** when,
even if its named problem is absent, the sentences it points at contain an error from pass A.

## What is reported

- flag precision: correct flags / all flags, per flag type and overall, with Wilson 95% CIs;
- the share of REVIEW drafts and of PASS drafts with a significant error, with Wilson CIs;
- the error types among PASS drafts (what the gate lets through);
- agreement between the model's labels and the author's on the re-labelled subset.
