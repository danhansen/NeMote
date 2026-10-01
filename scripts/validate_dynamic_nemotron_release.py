#!/usr/bin/env python3
"""Bind locally measured release gates to the exact generic model artifacts.

The supplied reports come from numerical verification and interleaved worker
benchmarks. This is a small regression gate, not a general accuracy claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def validate(args):
    config = load(args.source_dir / "config.json")
    frames = config["dynamic_streaming"]["supported_chunk_frames"]
    parity = load(args.encoder_parity)
    if parity.get("passed") is not True or parity["atol"] > 1e-4 or parity["rtol"] > 1e-4:
        raise ValueError("native encoder parity gate did not pass")
    expected_cases = {(mode, cache) for mode in frames for cache in (0, 5, config["dynamic_streaming"]["cache_len"])}
    if {(row["chunk_frames"], row["valid_cache"]) for row in parity["records"]} != expected_cases:
        raise ValueError("native fixtures do not cover all supported modes/cache occupancies")
    projected = None
    pointwise = None
    if args.profile == "compact" and config.get("pointwise_linear_rewrite_count", 0):
        if config["pointwise_linear_rewrite_count"] != config["num_encoder_layers"] * 2:
            raise ValueError("incomplete pointwise projection rewrite")
        if getattr(args, "pointwise_parity", None) is None:
            raise ValueError("pointwise lowering requires numerical parity evidence")
        pointwise = load(args.pointwise_parity)
        if pointwise.get("passed") is not True or pointwise["atol"] > 1e-3 or pointwise["rtol"] > 1e-4:
            raise ValueError("pointwise lowering numerical gate did not pass")
        if {(row["chunk_frames"], row["valid_cache"]) for row in pointwise["records"]} != expected_cases:
            raise ValueError("pointwise parity does not cover all supported modes/cache occupancies")
    if args.profile == "fast":
        if args.projected_parity is None:
            raise ValueError("FP32 requires final projected-cache parity evidence")
        projected = load(args.projected_parity)
        # Splitting one GEMM into cached/current GEMMs changes FP32 accumulation.
        # Keep the explicit budget visible; audio transcript parity is also required.
        if projected.get("passed") is not True or projected["atol"] > 1e-3 or projected["rtol"] > 1e-4:
            raise ValueError("projected-cache numerical gate did not pass")
        if Path(projected["encoder"]).resolve() != (args.source_dir / "encoder.onnx").resolve():
            raise ValueError("projected parity report describes another encoder")
    modes = load(args.modes_report)
    if modes.get("passed") is not True or {
        row["chunk_frames"] for row in modes["results"] if row["label"] == args.profile
    } != set(frames):
        raise ValueError("worker smoke test does not cover all supported modes")
    corpus = load(args.corpus_report)
    baseline = {row["clip"]: row for row in corpus["results"] if row["label"] == args.profile + "-baseline"}
    candidate = {row["clip"]: row for row in corpus["results"] if row["label"] == args.profile + "-generic"}
    if len(baseline) < 5 or baseline.keys() != candidate.keys():
        raise ValueError("at least five matching baseline/candidate speech clips are required")
    for clip in baseline:
        normalize = lambda text: re.findall(r"[a-z0-9']+", text.lower())
        if normalize(baseline[clip]["text"]) != normalize(candidate[clip]["text"]):
            raise ValueError(f"transcript regression on {clip}")
    benchmark = load(args.benchmark_report)
    if benchmark.get("status") != "complete":
        raise ValueError("benchmark incomplete")
    if benchmark.get("settings", {}).get("interleave") is not True:
        raise ValueError("performance runs must be interleaved")
    summaries = {row["label"]: row for row in benchmark["summaries"]}
    base, current = summaries["baseline"], summaries["generic"]
    if min(base["runs"], current["runs"]) < 5:
        raise ValueError("at least five interleaved performance runs are required")
    paired = {}
    for row in benchmark.get("runs", []):
        if row["label"] in ("baseline", "generic"):
            key = row["run_index"]
            pair = paired.setdefault(key, {})
            if row["label"] in pair:
                raise ValueError("duplicate performance run")
            pair[row["label"]] = row["metrics"]["decode_seconds"]
    if len(paired) < 5 or any(set(pair) != {"baseline", "generic"} for pair in paired.values()):
        raise ValueError("at least five complete paired performance runs are required")
    if any(not math.isfinite(value) or value <= 0 for pair in paired.values() for value in pair.values()):
        raise ValueError("performance decode times must be positive")
    # A ratio of independent medians can mask regressions under clock drift.
    ratios = [pair["generic"] / pair["baseline"] for pair in paired.values()]
    regression = (statistics.median(ratios) - 1) * 100
    if regression > 5:
        raise ValueError(f"decode regression {regression:.2f}% exceeds the user's 5% noise band")
    startup = load(args.startup_report)
    startup_summaries = {row["label"]: row for row in startup["summaries"]}
    if startup_summaries["generic"]["max_peak_rss_kib"] > startup_summaries["baseline"]["max_peak_rss_kib"] * 1.05:
        raise ValueError("peak memory regression exceeds 5%")
    names = ["encoder.onnx", "decoder_joint.onnx", "tokenizer.model", "config.json",
             *config.get("shared_weight_files", [])]
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("unsafe artifact path")
    return {"format": 1, "passed": True, "profile": args.profile,
            "supported_chunk_frames": frames, "performance_noise_percent": 5,
            "decode_regression_percent": regression,
            "paired_decode_ratios": ratios,
            "files": {name: {"sha256": sha256(args.source_dir / name),
                              "bytes": (args.source_dir / name).stat().st_size} for name in names},
            "native_encoder_parity": parity,
            "projected_encoder_parity": projected,
            "pointwise_encoder_parity": pointwise,
            "corpus_summary": corpus["summary"],
            "performance_summary": benchmark["summaries"],
            "startup_and_memory_summary": startup["summaries"],
            "runtime_modes": modes["results"],
            "limitations": ["Small English LibriSpeech regression set; not a population WER estimate.",
                            "First use performs local ORT graph optimization, not a NeMo export.",
                            "Cold-start cost is recorded, not covered by the steady-state decode noise band."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("fast", "compact"), required=True)
    for name in ("encoder-parity", "corpus-report", "benchmark-report", "startup-report", "modes-report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--projected-parity", type=Path)
    parser.add_argument("--pointwise-parity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"release gates passed; decode change {report['decode_regression_percent']:+.2f}%")


if __name__ == "__main__":
    main()
