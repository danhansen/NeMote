#!/usr/bin/env python3
"""Interleave identical-audio baseline/candidate worker measurements."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--ort-dylib", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ, ORT_DYLIB_PATH=str(args.ort_dylib),
               PARAKEET_LOAD_TRACE="1")
    results = {"baseline": [], "candidate": []}
    transcripts = set()
    partials = {}
    for run in range(args.runs):
        for label in (("baseline", "candidate") if run % 2 == 0 else ("candidate", "baseline")):
            worker = getattr(args, label)
            command = [str(worker.resolve()), "--model-dir", str(args.model_dir),
                       "--wav", str(args.wav), "--num-threads", "2"]
            if label == "candidate" and args.cache_dir is not None:
                command += ["--ort-optimized-model-cache-dir", str(args.cache_dir)]
            process = subprocess.run(command, env=env, check=True,
                                     text=True, capture_output=True)
            events = [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]
            commit = next(event for event in reversed(events) if event.get("event") == "commit")
            load = next(event for event in events if event.get("event") == "model_loaded")
            text_sequence = [event["text"] for event in events if event.get("event") == "partial"]
            transcripts.add(commit["text"])
            if partials:
                assert text_sequence == next(iter(partials.values())), "partial text sequence changed"
            partials[label] = text_sequence
            item = {
                "run": run, "decode_seconds": commit["data"]["decode_seconds"],
                "load_seconds": load["data"]["load_seconds"],
                "rtf": commit["data"]["real_audio_real_time_factor"],
                "cache_hit": "optimized cache hit encoder" in process.stderr,
            }
            results[label].append(item)
            print(label, json.dumps(item), flush=True)
    assert len(transcripts) == 1, "baseline and candidate transcripts differ"
    summary = {
        label: {field: statistics.median(item[field] for item in rows)
                for field in ("decode_seconds", "load_seconds", "rtf")}
        for label, rows in results.items()
    }
    result = {
        "model_dir": str(args.model_dir), "wav": str(args.wav),
        "ort_dylib": str(args.ort_dylib), "transcript": next(iter(transcripts)),
        "partials_identical": True, "runs": results, "medians": summary,
        "decode_delta_percent": (
            summary["candidate"]["decode_seconds"] / summary["baseline"]["decode_seconds"] - 1) * 100,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["medians"]), flush=True)
    print(f"decode delta: {result['decode_delta_percent']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
