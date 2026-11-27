"""Gradio app: paste a chest X-ray report, get a plain-language draft for clinician review and
a PASS/REVIEW decision with the report sentences behind each flag.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Run:  python app.py   (then open http://localhost:7860)
API:  POST to the "summarize" endpoint, e.g. with gradio_client:
        Client("http://localhost:7860").predict("No pneumothorax.", api_name="/summarize")
      returns {"summary", "decision", "flags", "meta"}.

The writer is loaded once at import. On a Hugging Face ZeroGPU Space only the drafting step
runs on the GPU (@spaces.GPU); anywhere else the decorator does nothing and the writer uses
the GPU if there is one, else the CPU. The app logs the time, decision and number of flags of
each request, never the report or the draft.
"""
import logging
import sys
import time
from pathlib import Path

# Spaces install requirements.txt but not this package, so put src/ on the path here.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402
import spaces  # noqa: E402

from laysum.examples import APP_EXAMPLES, by_name  # noqa: E402
from laysum.pipeline import MAX_REPORT_CHARS, gpu_seconds, highlight, run, validate_report  # noqa: E402
from laysum.writer import FlanT5Writer  # noqa: E402

log = logging.getLogger("laysum.app")

# ZeroGPU wants the model on "cuda" at import (CUDA is emulated outside @spaces.GPU);
# elsewhere the writer picks the GPU if there is one, else the CPU.
WRITER = FlanT5Writer.from_pretrained(device="cuda" if spaces.config.Config.zero_gpu else None)

LABELS = {"NEGATION_LOST": "negation lost", "CERTAINTY_LOST": "uncertainty lost",
          "INPUT_TRUNCATED": "not read"}
COLOURS = {"negation lost": "#eb6834", "uncertainty lost": "#2a78d6", "not read": "#8a8984"}

DISCLAIMER = (
    "**Research prototype. Not a medical device; not for diagnosis or patient care.** "
    "Drafts are written by a fine-tuned FLAN-T5 model and must be checked by a clinician. "
    "Do not paste reports that contain personal information."
)
NOT_COVERED = (
    "**What the flags cannot see:** content the draft adds that is not in the report, or a "
    "finding it states differently. A PASS means no flag fired, not that the draft is correct."
)
EXAMPLES = [[by_name(name).report] for name in APP_EXAMPLES]  # self-written
# Gradio's HighlightedText sets word-break: break-all and a tall line height, which split words
# mid-way ("edem" / "a.") and space the report lines far apart; wrap at word boundaries instead.
CSS = "#report-view .textfield { word-break: normal; overflow-wrap: anywhere; line-height: 1.6; }"


@spaces.GPU(duration=gpu_seconds)
def _draft(report: str) -> dict:
    return run(report, WRITER)


def summarize(report: str) -> dict:
    """Draft a plain-language summary of a chest X-ray report and decide PASS or REVIEW.

    Args:
        report: the radiology report text (at most 20,000 characters).

    Returns:
        {"summary": str, "decision": "PASS" | "REVIEW",
         "flags": [{"code", "message", "source_spans"}], "meta": {...}}
    """
    try:
        validate_report(report)
    except ValueError as error:
        raise gr.Error(str(error)) from None
    start = time.perf_counter()
    result = _draft(report)
    log.info("decision=%s flags=%d seconds=%.2f", result["decision"],
             result["meta"]["n_flags"], time.perf_counter() - start)
    return result


def _decision_markdown(result: dict) -> str:
    if result["decision"] == "PASS":
        return "### PASS\nNo flag fired. Still read the draft against the report."
    lines = [f"- **{LABELS[f['code']]}**: {f['message']}" for f in result["flags"]]
    return "### REVIEW\nCheck the highlighted report sentences:\n" + "\n".join(lines)


def review(report: str):
    result = summarize(report)
    pieces = [(text, " + ".join(LABELS[c] for c in code.split(" + ")) if code else None)
              for text, code in highlight(report, result["flags"])]
    return _decision_markdown(result), result["summary"], pieces


with gr.Blocks(title="Radiology Lay Summariser", analytics_enabled=False) as demo:
    gr.Markdown("# Radiology Lay Summariser\nA plain-language draft of a chest X-ray report, "
                "with flags for the clinician who reviews it.")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            report_box = gr.Textbox(label="Chest X-ray report", lines=10,
                                    max_length=MAX_REPORT_CHARS,
                                    placeholder="Paste a de-identified report in English.")
            run_button = gr.Button("Draft summary", variant="primary")
            gr.Examples(EXAMPLES, inputs=report_box, label="Self-written examples")
        with gr.Column():
            decision_box = gr.Markdown()
            draft_box = gr.Textbox(label="Draft for the patient (to be checked)", lines=6,
                                   interactive=False)
            report_view = gr.HighlightedText(label="Report, with the sentences behind each flag",
                                             color_map=COLOURS, show_legend=True,
                                             elem_id="report-view")
            gr.Markdown(NOT_COVERED)
    run_button.click(review, inputs=report_box, outputs=[decision_box, draft_box, report_view],
                     api_visibility="private")
    gr.api(summarize, api_name="summarize")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    demo.launch(footer_links=["api"], css=CSS)
