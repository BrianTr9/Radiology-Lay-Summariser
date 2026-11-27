"""Deploy the app to the Hugging Face Space, wait until it runs, and call its API once.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Run by the `deploy` job in .github/workflows/ci.yml after the tests and the Docker check pass
on main; also runs by hand: HF_TOKEN=<write token> python .github/deploy_space.py

Uploads only the files the Space needs (git push to a Space rejects large binaries such as
the adapter). The Space rebuilds on each new commit; the script waits until the Space runs
that commit, then drafts one self-written report through the public API.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from gradio_client import Client
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laysum.examples import by_name  # noqa: E402

SMOKE_CASE = "finding_and_negation"  # checked for its expected decision, not just a valid one
SPACE = "BrianTr9/radiology-lay-summariser"
FILES = ["app.py", "requirements.txt", "README.md", "LICENSE", "src/laysum/*.py",
         "models/lora_r16/*", "docs/review_screen.png", "training/figures/results.png"]
TIMEOUT_S = 20 * 60


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set; nothing deployed.")
        return 1
    api = HfApi(token=token)
    sha = os.environ.get("GITHUB_SHA") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    commit = api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=str(ROOT),
                               allow_patterns=FILES, commit_message=f"Deploy {sha[:7]}")
    target = commit.oid
    print(f"uploaded {sha[:7]} as Space commit {target[:7]}")

    start = time.time()
    while time.time() - start < TIMEOUT_S:
        runtime = api.get_space_runtime(SPACE)
        running_sha = runtime.raw.get("sha")
        print(f"{time.time() - start:5.0f}s  {runtime.stage}  running {str(running_sha)[:7]}")
        if "ERROR" in runtime.stage:
            print(runtime.raw.get("errorMessage"))
            return 1
        if runtime.stage == "RUNNING" and running_sha == target:
            break
        time.sleep(20)
    else:
        print("the Space did not reach RUNNING on the new commit in time")
        return 1

    case = by_name(SMOKE_CASE)
    result = Client(SPACE, token=token, verbose=False).predict(case.report,
                                                               api_name="/summarize")
    got = (result["decision"], tuple(f["code"] for f in result["flags"]))
    print(*got, result["meta"]["model_version"])
    if not result["summary"] or got != (case.decision, case.codes):
        print(f"expected {case.decision} {case.codes} for the {SMOKE_CASE!r} case")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
