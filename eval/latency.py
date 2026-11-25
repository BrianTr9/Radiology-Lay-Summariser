"""End-to-end latency of the running app's "summarize" API: p50 and p95 per report.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
AI assistance: drafted with Claude Code; reviewed by the author.

Start the app first (for the reported number, the Docker image on CPU):
    docker run --rm -p 7860:7860 --cpus 4 laysum
    python eval/latency.py --note "docker --cpus 4"

It sends the self-written cases of laysum.examples one at a time (one warm-up round, then
--rounds rounds) and times each call from the client, so the time includes the API and the
queue as well as drafting. The long case is reported apart: it is cut at 512 tokens and
writes the longest draft. Writes eval/results/latency.json with the machine it ran on.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradio_client import Client  # noqa: E402

from laysum.examples import CASES  # noqa: E402


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


def summary(seconds):
    return {"n": len(seconds), "p50": round(percentile(seconds, 50), 2),
            "p95": round(percentile(seconds, 95), 2), "mean": round(statistics.mean(seconds), 2)}


def host():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count()}
    try:
        info["docker"] = subprocess.run(
            ["docker", "info", "--format", "{{.NCPU}} CPUs, {{.MemTotal}} bytes"],
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["docker"] = None
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--note", default="", help="how the app was run, e.g. 'docker --cpus 4'")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "latency.json"))
    args = ap.parse_args()

    client = Client(args.url, verbose=False)
    times = {case.name: [] for case in CASES}
    version = None
    for round_ in range(args.rounds + 1):
        for case in CASES:
            start = time.perf_counter()
            result = client.predict(case.report, api_name="/summarize")
            elapsed = time.perf_counter() - start
            if round_:  # round 0 is the warm-up
                times[case.name].append(elapsed)
            version = result["meta"]["model_version"]

    short = [t for name, ts in times.items() if name != "too_long" for t in ts]
    out = {"note": args.note, "url_is_local": "127.0.0.1" in args.url or "localhost" in args.url,
           "model_version": version, "rounds": args.rounds,
           "reports_up_to_512_tokens": summary(short), "report_over_512_tokens": summary(times["too_long"]),
           "per_case_median": {name: round(statistics.median(ts), 2) for name, ts in times.items()},
           "host": host()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    s = out["reports_up_to_512_tokens"]
    print(f"p50 {s['p50']} s, p95 {s['p95']} s over {s['n']} calls "
          f"(long report: p50 {out['report_over_512_tokens']['p50']} s)")


if __name__ == "__main__":
    main()
