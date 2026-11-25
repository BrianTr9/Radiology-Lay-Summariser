"""Publish the LoRA adapter and its model card to the Hugging Face Hub.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Stages adapter_config.json, adapter_model.safetensors and docs/model_card.md (as README.md)
in a temporary folder and, only with --upload, creates the repo and uploads it. It prints the
commit hash to put in src/laysum/writer.py as ADAPTER_REVISION. Needs a write token:
`huggingface-cli login` or the HF_TOKEN environment variable. Nothing is uploaded without
--upload.

Usage: python scripts/publish_adapter.py --adapter <runs_from_rangpur/lora_r16_lr3e-3/best> [--upload]
"""
import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laysum.writer import ADAPTER_REPO  # noqa: E402

FILES = ("adapter_config.json", "adapter_model.safetensors")
PERSONAL = re.compile(r"/Users/|/home/|s4984662|@uq\.edu|hf_[A-Za-z0-9]{20,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, help="folder with the trained adapter")
    ap.add_argument("--repo", default=ADAPTER_REPO)
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--upload", action="store_true", help="really upload (default: dry run)")
    args = ap.parse_args()

    src = Path(args.adapter)
    card = ROOT / "docs" / "model_card.md"
    for f in FILES:
        assert (src / f).is_file(), f"missing {src / f}"
    for f in (src / "adapter_config.json", card):
        hit = PERSONAL.search(f.read_text())
        assert not hit, f"{f.name} contains {hit.group(0)!r}; remove it before publishing"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in FILES:
            shutil.copy(src / f, stage / f)
        shutil.copy(card, stage / "README.md")
        print(f"repo: {args.repo} ({'private' if args.private else 'public'})")
        for f in sorted(stage.iterdir()):
            print(f"  {f.name}  {f.stat().st_size:,} bytes")
        if not args.upload:
            print("dry run: nothing uploaded. Add --upload to publish.")
            return
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, private=args.private, exist_ok=True)
        info = api.upload_folder(folder_path=str(stage), repo_id=args.repo,
                                 commit_message="Add LoRA r16 adapter and model card")
        print(f"uploaded. Set ADAPTER_REVISION = \"{info.oid}\" in src/laysum/writer.py")


if __name__ == "__main__":
    main()
