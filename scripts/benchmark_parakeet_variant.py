#!/usr/bin/env python3
"""Run repeated long-WAV benchmarks for one or more Parakeet/Nemotron model dirs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ORT_DYLIB = Path(
    ".venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.27.0"
)


def parse_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            events.append(json.loads(line))
    return events


def final_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    commits = [event for event in events if event.get("event") == "commit"]
    if commits:
        return commits[-1]
    text_events = [event for event in events if event.get("text")]
    return text_events[-1] if text_events else (events[-1] if events else {})


def first_event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") == name:
            return event
    return None


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_power_metadata() -> dict[str, Any]:
    cpu0 = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    battery = Path("/sys/class/power_supply/BAT0")
    ac = Path("/sys/class/power_supply/AC")
    gnome_profile, gnome_profile_error = read_gnome_power_profile()
    return {
        "ac_online": read_text(ac / "online"),
        "battery_capacity_percent": read_text(battery / "capacity"),
        "battery_status": read_text(battery / "status"),
        "gnome_power_profile": gnome_profile,
        "gnome_power_profile_error": gnome_profile_error,
        "cpu0_scaling_governor": read_text(cpu0 / "scaling_governor"),
        "cpu0_scaling_cur_freq_khz": read_text(cpu0 / "scaling_cur_freq"),
        "cpu0_scaling_max_freq_khz": read_text(cpu0 / "scaling_max_freq"),
        "intel_pstate_no_turbo": read_text(Path("/sys/devices/system/cpu/intel_pstate/no_turbo")),
        "platform_profile": read_text(Path("/sys/firmware/acpi/platform_profile")),
    }


def read_gnome_power_profile() -> tuple[str | None, str | None]:
    try:
        proc = subprocess.run(
            [
                "busctl",
                "--system",
                "get-property",
                "net.hadess.PowerProfiles",
                "/net/hadess/PowerProfiles",
                "net.hadess.PowerProfiles",
                "ActiveProfile",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    output = proc.stdout.strip()
    if output.startswith('s "') and output.endswith('"'):
        return output[3:-1], None
    if output.startswith("s "):
        return output[2:].strip().strip('"'), None
    return output or None, None


def set_gnome_power_profile(profile: str) -> None:
    subprocess.run(
        [
            "busctl",
            "--system",
            "set-property",
            "net.hadess.PowerProfiles",
            "/net/hadess/PowerProfiles",
            "net.hadess.PowerProfiles",
            "ActiveProfile",
            "s",
            profile,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5.0,
    )


def hold_gnome_power_profile(profile: str) -> tuple[int | None, str | None]:
    command = [
        "busctl",
        "--system",
        "call",
        "net.hadess.PowerProfiles",
        "/net/hadess/PowerProfiles",
        "net.hadess.PowerProfiles",
        "HoldProfile",
        "sss",
        profile,
        "Wordpipe benchmark",
        "wordpipe",
    ]
    try:
        proc = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
        )
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        return None, message
    output = proc.stdout.strip()
    if not output.startswith("u "):
        return None, f"Unexpected HoldProfile response: {output!r}"
    return int(output[2:].strip()), None


def release_gnome_power_profile(cookie: int) -> None:
    subprocess.run(
        [
            "busctl",
            "--system",
            "call",
            "net.hadess.PowerProfiles",
            "/net/hadess/PowerProfiles",
            "net.hadess.PowerProfiles",
            "ReleaseProfile",
            "u",
            str(cookie),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5.0,
    )


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(":", 1)
        parts = raw_value.strip().split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0])
    return values


def read_memory_metadata() -> dict[str, Any]:
    meminfo = read_meminfo()
    return {
        "mem_available_kb": meminfo.get("MemAvailable"),
        "mem_free_kb": meminfo.get("MemFree"),
        "swap_total_kb": meminfo.get("SwapTotal"),
        "swap_free_kb": meminfo.get("SwapFree"),
        "swap_used_kb": (
            meminfo["SwapTotal"] - meminfo["SwapFree"]
            if "SwapTotal" in meminfo and "SwapFree" in meminfo
            else None
        ),
    }


def check_memory_guard(args: argparse.Namespace) -> None:
    if args.min_mem_available_gb <= 0:
        return
    mem_available_kb = read_memory_metadata().get("mem_available_kb")
    required_kb = int(args.min_mem_available_gb * 1024 * 1024)
    if mem_available_kb is None:
        raise RuntimeError("Cannot read MemAvailable for benchmark memory guard")
    if mem_available_kb < required_kb:
        raise RuntimeError(
            f"Refusing benchmark run: MemAvailable={mem_available_kb / 1024 / 1024:.1f} GiB "
            f"is below --min-mem-available-gb={args.min_mem_available_gb:.1f}"
        )


def check_power_guard(args: argparse.Namespace) -> None:
    if not args.require_ac and not args.require_power_profile and not args.set_power_profile:
        return
    if args.set_power_profile:
        set_gnome_power_profile(args.set_power_profile)
    power = read_power_metadata()
    if args.require_ac and power.get("ac_online") != "1":
        raise RuntimeError(
            "Refusing benchmark run: AC power is not online "
            f"(ac_online={power.get('ac_online')!r}, battery={power.get('battery_capacity_percent')!r}%)."
        )
    expected_profile = args.require_power_profile or args.set_power_profile
    if expected_profile:
        profile = power.get("gnome_power_profile")
        if profile != expected_profile:
            raise RuntimeError(
                f"Refusing benchmark run: GNOME power profile is {profile!r}, "
                f"expected {expected_profile!r}. "
                f"error={power.get('gnome_power_profile_error')!r}"
            )


def child_resource_limiter(args: argparse.Namespace):
    if args.child_memory_limit_gb <= 0:
        return None

    limit_bytes = int(args.child_memory_limit_gb * 1024 * 1024 * 1024)

    def limit_child() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    return limit_child


def run_once(args: argparse.Namespace, label: str, model_dir: Path, run_index: int) -> dict[str, Any]:
    check_power_guard(args)
    check_memory_guard(args)
    env = os.environ.copy()
    if "ORT_DYLIB_PATH" not in env and args.ort_dylib.exists():
        env["ORT_DYLIB_PATH"] = str(args.ort_dylib.resolve())
    env.setdefault("OMP_NUM_THREADS", str(args.num_threads))
    env.setdefault("MKL_NUM_THREADS", str(args.num_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(args.num_threads))

    command = [
        str(args.worker),
        "--model-dir",
        str(model_dir),
        "--wav",
        str(args.wav),
        "--num-threads",
        str(args.num_threads),
        "--flush-chunks",
        str(args.flush_chunks),
        "--graph-optimization",
        args.graph_optimization,
    ]
    if args.ort_memory_pattern != "auto":
        command.extend(["--ort-memory-pattern", args.ort_memory_pattern])
    if args.ort_parallel_execution:
        command.append("--ort-parallel-execution")
    if args.ort_cpu_arena != "auto":
        command.extend(["--ort-cpu-arena", args.ort_cpu_arena])
    if args.ort_optimized_model_cache_dir:
        command.extend(
            [
                "--ort-optimized-model-cache-dir",
                str(args.ort_optimized_model_cache_dir),
            ]
        )
    if getattr(args, "chunk_samples", None) is not None:
        command.extend(["--chunk-samples", str(args.chunk_samples)])
    if getattr(args, "measure_peak_memory", False):
        command = ["/usr/bin/time", "-f", "WORDPIPE_PEAK_RSS_KIB=%M", *command]
    started = time.perf_counter()
    power_before = read_power_metadata()
    memory_before = read_memory_metadata()
    proc = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=args.timeout_seconds,
        preexec_fn=child_resource_limiter(args),
    )
    wall_seconds = time.perf_counter() - started
    power_after = read_power_metadata()
    memory_after = read_memory_metadata()
    events = parse_events(proc.stdout)
    event = final_event(events)
    load_event = first_event(events, "model_loaded") or {}
    load_metrics = dict(load_event.get("data") or {})
    metrics = dict(event.get("data") or {})
    return {
        "label": label,
        "model_dir": str(model_dir),
        "run_index": run_index,
        "wall_seconds": wall_seconds,
        "load_seconds": load_metrics.get("load_seconds"),
        "peak_rss_kib": next((int(line.split("=", 1)[1]) for line in proc.stderr.splitlines()
                              if line.startswith("WORDPIPE_PEAK_RSS_KIB=")), None),
        "text": str(event.get("text") or ""),
        "metrics": metrics,
        "power_before": power_before,
        "power_after": power_after,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }


def median_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = []
    for row in rows:
        value = row["metrics"].get(metric)
        if value is not None:
            values.append(float(value))
    return statistics.median(values) if values else None


def summarize(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    load_values = [float(row["load_seconds"]) for row in rows if row.get("load_seconds") is not None]
    peak_values = [row["peak_rss_kib"] for row in rows if row.get("peak_rss_kib") is not None]
    return {
        "label": label,
        "runs": len(rows),
        "median_load_seconds": statistics.median(load_values) if load_values else None,
        "median_peak_rss_kib": statistics.median(peak_values) if peak_values else None,
        "max_peak_rss_kib": max(peak_values) if peak_values else None,
        "median_real_time_factor": median_metric(rows, "real_time_factor"),
        "median_real_audio_real_time_factor": median_metric(rows, "real_audio_real_time_factor"),
        "median_decode_seconds": median_metric(rows, "decode_seconds"),
        "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
        "texts": [row["text"] for row in rows],
    }


def make_output(
    args: argparse.Namespace,
    *,
    power_at_start: dict[str, Any],
    memory_at_start: dict[str, Any],
    summaries: list[dict[str, Any]],
    results: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "settings": {
            "wav": str(args.wav),
            "runs": args.runs,
            "interleave": args.interleave,
            "num_threads": args.num_threads,
            "flush_chunks": args.flush_chunks,
            "chunk_samples": getattr(args, "chunk_samples", None),
            "measure_peak_memory": getattr(args, "measure_peak_memory", False),
            "graph_optimization": args.graph_optimization,
            "min_mem_available_gb": args.min_mem_available_gb,
            "child_memory_limit_gb": args.child_memory_limit_gb,
            "require_ac": args.require_ac,
            "require_power_profile": args.require_power_profile,
            "set_power_profile": args.set_power_profile,
            "restore_power_profile": args.restore_power_profile,
            "previous_gnome_power_profile": args.previous_gnome_power_profile,
            "power_profile_hold_cookie": args.power_profile_hold_cookie,
            "power_profile_hold_error": args.power_profile_hold_error,
            "ort_memory_pattern": args.ort_memory_pattern,
            "ort_parallel_execution": args.ort_parallel_execution,
            "ort_cpu_arena": args.ort_cpu_arena,
            "ort_optimized_model_cache_dir": (
                str(args.ort_optimized_model_cache_dir)
                if args.ort_optimized_model_cache_dir
                else None
            ),
        },
        "power_at_start": power_at_start,
        "power_at_end": read_power_metadata(),
        "memory_at_start": memory_at_start,
        "memory_at_end": read_memory_metadata(),
        "summaries": summaries,
        "runs": results,
    }


def write_output(
    args: argparse.Namespace,
    *,
    power_at_start: dict[str, Any],
    memory_at_start: dict[str, Any],
    summaries: list[dict[str, Any]],
    results: list[dict[str, Any]],
    status: str,
) -> None:
    output = make_output(
        args,
        power_at_start=power_at_start,
        memory_at_start=memory_at_start,
        summaries=summaries,
        results=results,
        status=status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


def print_run_result(label: str, run_index: int, row: dict[str, Any]) -> None:
    metrics = row["metrics"]
    print(
        f"[bench] {label} run {run_index}: "
        f"load={row.get('load_seconds')} "
        f"rtf={metrics.get('real_time_factor')} "
        f"real_audio_rtf={metrics.get('real_audio_real_time_factor')} "
        f"decode={metrics.get('decode_seconds')} wall={row['wall_seconds']:.3f}",
        flush=True,
    )


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="+", help="Model dir, optionally label=/path.")
    parser.add_argument("--wav", type=Path, default=Path("build/allocation-ablation/librispeech-long.wav"))
    parser.add_argument("--worker", type=Path, default=Path("target/release/wordpipe-parakeet-worker"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Run pass 1 for every model, then pass 2, instead of finishing one model at a time.",
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--flush-chunks", type=int, default=3)
    parser.add_argument("--chunk-samples", type=int,
                        help="Select a checkpoint-supported chunk size (16 samples per ms at 16 kHz).")
    parser.add_argument("--measure-peak-memory", action="store_true",
                        help="Measure per-worker peak RSS, including cold optimization, using GNU /usr/bin/time.")
    parser.add_argument("--graph-optimization", default="all")
    parser.add_argument("--ort-memory-pattern", choices=("auto", "enable", "disable"), default="auto")
    parser.add_argument("--ort-parallel-execution", action="store_true")
    parser.add_argument("--ort-cpu-arena", choices=("auto", "enable", "disable"), default="auto")
    parser.add_argument(
        "--ort-optimized-model-cache-dir",
        type=Path,
        help="Directory for ORT optimized model cache artifacts used by the worker.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--ort-dylib", type=Path, default=DEFAULT_ORT_DYLIB)
    parser.add_argument("--output", type=Path, default=Path("build/parakeet-variant-bench/summary.json"))
    parser.add_argument(
        "--min-mem-available-gb",
        type=float,
        default=0.0,
        help="Refuse to start a run when /proc/meminfo MemAvailable is below this threshold.",
    )
    parser.add_argument(
        "--child-memory-limit-gb",
        type=float,
        default=0.0,
        help="Set RLIMIT_AS for each worker subprocess. 0 disables the limit.",
    )
    parser.add_argument(
        "--require-ac",
        action="store_true",
        help="Refuse to start each run unless /sys/class/power_supply/AC/online is 1.",
    )
    parser.add_argument(
        "--require-power-profile",
        help="Refuse to start each run unless GNOME power-profiles-daemon ActiveProfile matches this value.",
    )
    parser.add_argument(
        "--set-power-profile",
        help="Set GNOME power-profiles-daemon ActiveProfile for the benchmark; hold it when the daemon supports that profile.",
    )
    parser.add_argument(
        "--no-restore-power-profile",
        dest="restore_power_profile",
        action="store_false",
        help="Leave --set-power-profile active after the benchmark instead of restoring the previous profile.",
    )
    parser.set_defaults(
        restore_power_profile=True,
        previous_gnome_power_profile=None,
        power_profile_hold_cookie=None,
        power_profile_hold_error=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.wav.exists():
        raise SystemExit(f"Missing WAV: {args.wav}")
    if not args.worker.exists():
        raise SystemExit(f"Missing worker: {args.worker}")

    if args.set_power_profile:
        profile, error = read_gnome_power_profile()
        if error:
            raise SystemExit(f"Could not read current GNOME power profile: {error}")
        args.previous_gnome_power_profile = profile
        args.power_profile_hold_cookie, args.power_profile_hold_error = hold_gnome_power_profile(
            args.set_power_profile
        )
        set_gnome_power_profile(args.set_power_profile)

    try:
        power_at_start = read_power_metadata()
        memory_at_start = read_memory_metadata()
        results = []
        summaries = []
        models = list(map(parse_model_arg, args.model))
        for label, model_dir in models:
            if not model_dir.exists():
                raise SystemExit(f"Missing model dir for {label}: {model_dir}")

        if args.interleave:
            rows_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label, _ in models}
            for run_index in range(1, args.runs + 1):
                # Counterbalance ordering to avoid consistently favoring the
                # second model as clocks, temperature, and filesystem caches drift.
                order = models if run_index % 2 else list(reversed(models))
                for label, model_dir in order:
                    print(f"[bench] {label} run {run_index}/{args.runs}", flush=True)
                    row = run_once(args, label, model_dir, run_index)
                    rows_by_label[label].append(row)
                    results.append(row)
                    print_run_result(label, run_index, row)
                    summaries = [
                        summarize(label, rows)
                        for label, rows in rows_by_label.items()
                        if rows
                    ]
                    write_output(
                        args,
                        power_at_start=power_at_start,
                        memory_at_start=memory_at_start,
                        summaries=summaries,
                        results=results,
                        status="partial",
                    )
            summaries = []
            for label, _ in models:
                summary = summarize(label, rows_by_label[label])
                summaries.append(summary)
                print(f"[bench] {label} median {json.dumps(summary, sort_keys=True)}", flush=True)
        else:
            for label, model_dir in models:
                rows = []
                for run_index in range(1, args.runs + 1):
                    print(f"[bench] {label} run {run_index}/{args.runs}", flush=True)
                    row = run_once(args, label, model_dir, run_index)
                    rows.append(row)
                    print_run_result(label, run_index, row)
                    write_output(
                        args,
                        power_at_start=power_at_start,
                        memory_at_start=memory_at_start,
                        summaries=summaries,
                        results=results + rows,
                        status="partial",
                    )
                results.extend(rows)
                summary = summarize(label, rows)
                summaries.append(summary)
                print(f"[bench] {label} median {json.dumps(summary, sort_keys=True)}", flush=True)
                write_output(
                    args,
                    power_at_start=power_at_start,
                    memory_at_start=memory_at_start,
                    summaries=summaries,
                    results=results,
                    status="partial",
                )

        write_output(
            args,
            power_at_start=power_at_start,
            memory_at_start=memory_at_start,
            summaries=summaries,
            results=results,
            status="complete",
        )
        print(f"[bench] wrote {args.output}")
    finally:
        if args.power_profile_hold_cookie is not None:
            release_gnome_power_profile(args.power_profile_hold_cookie)
        if (
            args.set_power_profile
            and args.restore_power_profile
            and args.previous_gnome_power_profile
            and args.previous_gnome_power_profile != args.set_power_profile
        ):
            set_gnome_power_profile(args.previous_gnome_power_profile)


if __name__ == "__main__":
    main()
