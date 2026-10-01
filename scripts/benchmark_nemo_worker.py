#!/usr/bin/env python3
"""Compare the C++ worker with upstream's complete streaming request at concurrency one."""
import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--native-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 99:
        parser.error("repetitions must be in [1,99]")
    report = {"status": "running", "threads": 2, "concurrency": 1, "batching": False,
              "artifacts": {str(path): sha(path) for path in (args.worker, args.native_cli, args.model, args.wav)},
              "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for context in (1, 2, 7, 14):
        worker_command = [str(args.worker), "--model-dir", str(args.model), "--wav", str(args.wav),
                          "--device", "cpu",
                          "--num-threads", "2", "--chunk-samples", str(context * 1280),
                          "--wav-repeat", str(args.repetitions + 1)]
        native_command = [str(args.native_cli), "bench", "asr", str(args.wav), "-m", str(args.model),
                          "--mode", "stream", "--device", "cpu", "--asr.backend.threads", "2",
                          "--asr.streaming.rnnt_right_context", str(context - 1), "--chunk-ms", str(context * 80),
                          "-c", "1", "-n", str(args.repetitions), "--warmup", "1", "--json"]
        # Alternate order between modes; this is a screening test, not a confidence interval.
        commands = [worker_command, native_command] if context in (1, 7) else [native_command, worker_command]
        outputs = {}
        for command in commands:
            outputs[command[0]] = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180).stdout
        commits = [event for line in outputs[str(args.worker)].splitlines()
                   if (event := json.loads(line))["event"] == "commit"]
        if len(commits) != args.repetitions + 1 or len({event["text"] for event in commits}) != 1:
            raise SystemExit("Repeated-session output changed during benchmarking")
        worker_seconds = [event["data"]["elapsed_seconds"] for event in commits[1:]]
        native = json.loads(outputs[str(args.native_cli)])
        native_seconds = native["runs"][0]["latency_ms"]["p50"] / 1000
        median = statistics.median(worker_seconds)
        row = {"chunk_frames": context, "worker_seconds": worker_seconds,
               "worker_median_seconds": median, "upstream_median_seconds": native_seconds,
               "worker_overhead_percent": (median / native_seconds - 1) * 100,
               "worker_cold_metrics": commits[0]["data"], "upstream": native,
               "worker_command": worker_command, "upstream_command": native_command}
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{context * 80} ms: worker={median:.3f}s native={native_seconds:.3f}s delta={row['worker_overhead_percent']:+.1f}%", flush=True)
    report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
