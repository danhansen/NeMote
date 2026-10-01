from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import DEFAULT_CONFIG, WordpipeConfig, load_config
from .probe import ProbeResult, run_probe
from .audio import parse_audio_device


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _non_negative_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def _cmd_probe(args: argparse.Namespace) -> int:
    result = run_probe()
    if args.json:
        _print_json(result.to_dict())
        return 0 if result.usable else 2

    print(result.render_text())
    return 0 if result.usable else 2


def _cmd_asr_worker(args: argparse.Namespace) -> int:
    from .asr_worker import AsrWorkerConfig, run_stdio_worker

    config = AsrWorkerConfig(
        model_dir=Path(args.model_dir),
        provider=args.provider,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        input_device=parse_audio_device(args.input_device),
        partial_interval_seconds=args.partial_interval_seconds,
        audio_chunk_seconds=args.audio_chunk_seconds,
        queue_seconds=args.queue_seconds,
        stats_interval_seconds=args.stats_interval_seconds,
        enable_endpoint_detection=args.endpoint,
        endpoint_rule1_min_trailing_silence=args.endpoint_rule1_min_trailing_silence,
        endpoint_rule2_min_trailing_silence=args.endpoint_rule2_min_trailing_silence,
        endpoint_rule3_min_utterance_length=args.endpoint_rule3_min_utterance_length,
    )
    return run_stdio_worker(config)


def _cmd_model_info(args: argparse.Namespace) -> int:
    from .asr_worker import render_model_info

    print(render_model_info(Path(args.model_dir)))
    return 0


def _cmd_transcribe_file(args: argparse.Namespace) -> int:
    from .asr_worker import AsrWorkerConfig, transcribe_wav_file

    config = AsrWorkerConfig(
        model_dir=Path(args.model_dir),
        provider=args.provider,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        endpoint_rule1_min_trailing_silence=args.endpoint_rule1_min_trailing_silence,
        endpoint_rule2_min_trailing_silence=args.endpoint_rule2_min_trailing_silence,
        endpoint_rule3_min_utterance_length=args.endpoint_rule3_min_utterance_length,
    )
    text, metrics = transcribe_wav_file(
        config,
        Path(args.wav),
        flush_chunks=args.flush_chunks,
    )
    print(text)
    if args.metrics:
        print(json.dumps(metrics, indent=2, sort_keys=True), file=sys.stderr)
    return 0


def _cmd_listen_test(args: argparse.Namespace) -> int:
    from .listen_test import ListenTestConfig, run_listen_test

    file_config = _load_cli_config(args)
    config = ListenTestConfig(
        model_dir=_resolve_model_dir(args, file_config),
        asr_runtime=args.asr_runtime,
        asr_worker_path=Path(args.asr_worker_path).expanduser()
        if args.asr_worker_path
        else None,
        provider=args.provider,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        input_device=parse_audio_device(args.input_device),
        partial_interval_seconds=args.partial_interval_seconds,
        audio_chunk_seconds=args.audio_chunk_seconds,
        queue_seconds=args.queue_seconds,
        stats_interval_seconds=args.stats_interval_seconds,
        enable_endpoint_detection=args.endpoint,
        endpoint_rule1_min_trailing_silence=args.endpoint_rule1_min_trailing_silence,
        endpoint_rule2_min_trailing_silence=args.endpoint_rule2_min_trailing_silence,
        endpoint_rule3_min_utterance_length=args.endpoint_rule3_min_utterance_length,
        duration_seconds=args.duration,
        json_output=args.json,
        full_hypotheses=args.full_hypotheses,
    )
    return run_listen_test(config)


def _cmd_stream_file_test(args: argparse.Namespace) -> int:
    from .listen_test import _format_event

    file_config = _load_cli_config(args)
    model_dir = _resolve_model_dir(args, file_config)
    if args.asr_runtime == "parakeet":
        from .daemon import _resolve_parakeet_worker, parakeet_worker_env

        command = [
            str(
                _resolve_parakeet_worker(
                    Path(args.asr_worker_path).expanduser()
                    if args.asr_worker_path
                    else None
                )
            ),
            "--model-dir",
            str(model_dir),
            "--num-threads",
            str(args.num_threads),
            "--sample-rate",
            str(args.sample_rate),
            "--stats-interval-seconds",
            str(args.stats_interval_seconds),
            "--chunk-samples",
            str(max(1, int(args.sample_rate * args.chunk_seconds))),
            "--flush-chunks",
            str(args.flush_chunks),
            "--wav",
            str(Path(args.wav)),
        ]
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=parakeet_worker_env(),
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        stderr_lines: list[str] = []
        stderr_thread = threading.Thread(
            target=_collect_stderr_lines,
            args=(proc.stderr, stderr_lines),
            name="wordpipe-stream-file-stderr",
            daemon=True,
        )
        stderr_thread.start()
        for line in proc.stdout:
            item = json.loads(line)
            if args.json:
                print(json.dumps(item, sort_keys=True), flush=True)
            else:
                print(_format_event(item), flush=True)
        return_code = proc.wait()
        stderr_thread.join(timeout=1)
        if return_code != 0:
            if stderr_lines:
                print("".join(stderr_lines), file=sys.stderr, end="")
            return return_code or 1
        return 0

    from .asr_worker import AsrWorkerConfig, stream_wav_file_events

    config = AsrWorkerConfig(
        model_dir=model_dir,
        provider=args.provider,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        partial_interval_seconds=args.partial_interval_seconds,
        audio_chunk_seconds=args.audio_chunk_seconds,
        queue_seconds=args.queue_seconds,
        stats_interval_seconds=args.stats_interval_seconds,
        endpoint_rule1_min_trailing_silence=args.endpoint_rule1_min_trailing_silence,
        endpoint_rule2_min_trailing_silence=args.endpoint_rule2_min_trailing_silence,
        endpoint_rule3_min_utterance_length=args.endpoint_rule3_min_utterance_length,
    )
    for item in stream_wav_file_events(
        config,
        Path(args.wav),
        chunk_seconds=args.chunk_seconds,
        flush_chunks=args.flush_chunks,
        reset_on_endpoint=args.reset_on_endpoint,
    ):
        if args.json:
            print(json.dumps(item, sort_keys=True), flush=True)
        else:
            print(_format_event(item), flush=True)
    return 0


def _collect_stderr_lines(stream, lines: list[str]) -> None:  # type: ignore[no-untyped-def]
    for line in stream:
        lines.append(line)


def _cmd_audio_devices(args: argparse.Namespace) -> int:
    from .audio import render_input_devices, render_parakeet_input_devices

    if args.backend == "parakeet":
        print(
            render_parakeet_input_devices(
                Path(args.asr_worker_path).expanduser() if args.asr_worker_path else None
            )
        )
    else:
        print(render_input_devices())
    return 0


def _cmd_record_test(args: argparse.Namespace) -> int:
    from .audio import record_wav

    record_wav(
        Path(args.output),
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
        device=parse_audio_device(args.input_device),
    )
    print(args.output)
    return 0


def _cmd_download_model(args: argparse.Namespace) -> int:
    from .models import download_model, make_download_plan

    plan = make_download_plan(
        output_dir=Path(args.output_dir),
        repo_id=args.repo_id,
        include_test_wavs=args.include_test_wavs,
    )
    model_dir = download_model(plan, force=args.force)
    print(model_dir)
    return 0


def _cmd_model_install(args: argparse.Namespace) -> int:
    requested_family = getattr(args, "model_family", "multilingual")
    families = ("multilingual", "english") if requested_family == "all" else (requested_family,)
    if len(families) > 1 and (args.source or getattr(args, "prebuilt_repo", None)):
        raise SystemExit(
            "--source and --prebuilt-repo install one checkpoint family; "
            "pass --model-family multilingual or --model-family english"
        )
    for family in families:
        _cmd_model_install_family(args, family)
    return 0


def _cmd_model_install_family(args: argparse.Namespace, family: str) -> int:
    from .models import (
        DEFAULT_ENGLISH_NEMO_SOURCE_REPO,
        build_model_profile,
        download_prebuilt_profile,
        download_streaming_profile,
        default_nemo_source_path,
        download_nemo_source,
        ensure_profile_completion_marker,
        install_built_profile,
        install_prebuilt_profile,
        model_runtime_dir_valid,
        profile_runtime_dir,
        profile_spec,
        source_may_be_built_profile_archive,
        source_is_built_profile,
        streaming_model_root,
        profile_streaming_latency,
        profile_supports_streaming_latency,
        profile_is_dynamic,
    )

    file_config = _load_cli_config(args)
    profile = args.profile or file_config.model_profile
    model_root = Path(args.model_root).expanduser() if args.model_root else file_config.model_root
    if model_root is None:
        raise SystemExit("model_root is required")
    latency_ms = getattr(args, "streaming_latency_ms", 560)
    if profile == "nemo-q8":
        from .nemo_models import install_nemo_model
        if latency_ms not in (80, 160, 560, 1120):
            raise SystemExit("NeMo supports 80, 160, 560, and 1120 ms chunks")
        if args.build_from_nemo or args.prebuilt_repo:
            raise SystemExit("NeMo Q8 uses the official GGUF; no local export or repo override is needed")
        if args.dry_run:
            print(profile_runtime_dir(model_root, profile, family))
            return 0
        print(install_nemo_model(model_root, family,
            source=Path(args.source).expanduser() if args.source else None,
            force=args.force or args.force_source,
            progress=lambda message: print(message, file=sys.stderr, flush=True)))
        return 0
    base_model_root = model_root
    model_root = (base_model_root if getattr(args, "dynamic_streaming", False)
                  else streaming_model_root(model_root, latency_ms))
    base_runtime = profile_runtime_dir(base_model_root, profile, family)
    if profile_is_dynamic(base_runtime, family):
        model_root = base_model_root

    def report(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    build_from_nemo = getattr(args, "build_from_nemo", bool(args.source))
    prebuilt_repo = getattr(args, "prebuilt_repo", None)
    source_candidate = Path(args.source).expanduser() if args.source else None
    runtime_dir = profile_runtime_dir(model_root, profile, family)
    if (
        not args.force
        and not args.force_source
        and not args.dry_run
        and source_candidate is None
        and model_runtime_dir_valid(runtime_dir)
    ):
        if not profile_supports_streaming_latency(runtime_dir, latency_ms):
            raise SystemExit("installed model streaming latency does not match the requested mode")
        runtime_dir = ensure_profile_completion_marker(model_root, profile, family)
        report(f"Using installed model profile: {runtime_dir}")
        print(runtime_dir)
        return 0
    if source_candidate is not None and source_candidate.exists() and (
        source_is_built_profile(source_candidate) or source_may_be_built_profile_archive(source_candidate)
    ):
        if source_candidate.is_dir() and not profile_supports_streaming_latency(source_candidate, latency_ms):
            raise SystemExit("source model streaming latency does not match the requested mode")
        if source_candidate.is_dir() and profile_is_dynamic(source_candidate, family):
            model_root = base_model_root
        runtime_dir = install_built_profile(
            source=source_candidate,
            model_root=model_root,
            profile=profile,
            family=family,
            force=args.force,
            streaming_latency_ms=latency_ms,
        )
        print(runtime_dir)
        return 0

    if not build_from_nemo:
        if args.source:
            raise SystemExit(
                "--source must be a local built Wordpipe profile directory/archive unless "
                "--build-from-nemo is used"
            )
        elif args.dry_run:
            selected_repo = prebuilt_repo or profile_spec(profile).prebuilt_repo_for_family(family)
            source_path = model_root / "downloads" / selected_repo.replace("/", "--") / profile
        elif latency_ms != 560:
            source_path = download_streaming_profile(
                profile=profile, model_root=base_model_root, family=family,
                latency_ms=latency_ms, repo_id=prebuilt_repo, force=args.force_source,
                progress=report,
            )
        else:
            source_path = download_prebuilt_profile(
                profile=profile,
                model_root=model_root,
                family=family,
                repo_id=prebuilt_repo,
                force=args.force_source,
                progress=report,
            )
        if args.dry_run:
            print(source_path)
            return 0
        if not profile_supports_streaming_latency(source_path, latency_ms):
            raise SystemExit("downloaded model streaming latency does not match the requested mode")
        upgrading_for_new_mode = False
        if profile_is_dynamic(source_path, family):
            model_root = base_model_root
            upgrading_for_new_mode = latency_ms != 560 and not profile_supports_streaming_latency(
                profile_runtime_dir(base_model_root, profile, family), latency_ms,
            )
            if upgrading_for_new_mode:
                report("Installing a generic encoder supporting the checkpoint's chunk sizes")
        runtime_dir = install_prebuilt_profile(
            source=source_path,
            model_root=model_root,
            profile=profile,
            family=family,
            python=Path(args.python).expanduser(),
            force=args.force or upgrading_for_new_mode,
            progress=report,
        )
        print(runtime_dir)
        return 0

    source_value = args.source or (
        DEFAULT_ENGLISH_NEMO_SOURCE_REPO if family == "english" else file_config.nemo_source
    )
    source_candidate = Path(source_value).expanduser()
    source_output = Path(args.source_output).expanduser() if args.source_output else None
    if args.dry_run:
        source_path = (
            source_candidate
            if source_candidate.exists()
            else source_output or default_nemo_source_path(model_root, family)
        )
    else:
        download_kwargs = {
            "force": args.force_source,
            "progress": report,
        }
        if family == "english":
            download_kwargs["family"] = family
        source_path = download_nemo_source(
            source_value,
            source_output or default_nemo_source_path(model_root, family),
            **download_kwargs,
        )
    runtime_dir = build_model_profile(
        source=source_path,
        model_root=model_root,
        profile=profile,
        family=family,
        python=Path(args.python).expanduser(),
        force=args.force,
        dry_run=args.dry_run,
        keep_build_dir=args.keep_build_dir,
        progress=report,
        **({"streaming_latency_ms": latency_ms} if latency_ms != 560 else {}),
        **({"dynamic_streaming": True} if getattr(args, "dynamic_streaming", False) else {}),
    )
    print(runtime_dir)
    return 0


def _cmd_model_profiles(args: argparse.Namespace) -> int:
    from .models import MODEL_PROFILES, profile_installed, profile_runtime_dir

    file_config = _load_cli_config(args)
    model_root = Path(args.model_root).expanduser() if args.model_root else file_config.model_root
    rows = []
    presets = (
        ("nemo-q8", "NeMo Q8 (multilingual)", "multilingual"),
        ("nemo-q8-english", "NeMo Q8 (English only)", "english"),
        ("compact", "compact", "multilingual"),
        ("compact-english", "compact (English only, lower WER)", "english"),
        ("fast", "fast", "multilingual"),
        ("fast-english", "fast (English only, lower WER)", "english"),
    )
    for name, title, family in presets:
        profile = name.removesuffix("-english")
        spec = MODEL_PROFILES[profile]
        runtime_dir = profile_runtime_dir(model_root, profile, family) if model_root else None
        installed = bool(model_root and profile_installed(model_root, profile, family))
        rows.append(
            {
                "name": name,
                "title": title,
                "backend": "nemo-speech" if profile == "nemo-q8" else "parakeet",
                "model_profile": profile,
                "model_family": family,
                "description": spec.description,
                "build_profile": spec.build_profile,
                "prebuilt_repo": spec.prebuilt_repo_for_family(family),
                "runtime_dir": str(runtime_dir) if runtime_dir is not None else None,
                "installed": installed,
            }
        )
    if args.json:
        _print_json(rows)
    else:
        for row in rows:
            state = "installed" if row["installed"] else "not installed"
            print(f"{row['name']}: {state}")
            print(f"  {row['description']}")
            print(f"  runtime: {row['runtime_dir']}")
    return 0


def _cmd_type_text(args: argparse.Namespace) -> int:
    from .insertion import DryRunKeyboardBackend, PortalKeyboardBackend

    backend = DryRunKeyboardBackend() if args.dry_run else PortalKeyboardBackend()
    backend.open()
    try:
        backend.insert_text(args.text)
    finally:
        backend.close()

    if args.dry_run:
        for event in backend.events:
            print(event)
    return 0


def _shortcut_spec_from_args(args: argparse.Namespace):
    from .shortcuts import local_shortcut_spec

    root = Path(args.root).expanduser() if args.root else Path(__file__).resolve().parents[2]
    return local_shortcut_spec(root, binding=args.binding)


def _cmd_shortcut_status(args: argparse.Namespace) -> int:
    from .shortcuts import read_shortcut_status

    status = read_shortcut_status(_shortcut_spec_from_args(args))
    if args.json:
        _print_json(
            {
                "present": status.present,
                "matches": status.matches,
                "name": status.name,
                "command": status.command,
                "binding": status.binding,
                "expected": {
                    "name": status.spec.name,
                    "command": status.spec.command,
                    "binding": status.spec.binding,
                    "path": status.spec.path,
                },
                "configured_paths": list(status.configured_paths),
            }
        )
    else:
        print(status.summary)
        if not status.matches:
            print(f"expected: {status.spec.binding} -> {status.spec.command}")
    return 0 if status.matches else 2


def _cmd_shortcut_install(args: argparse.Namespace) -> int:
    from .shortcuts import install_shortcut

    status = install_shortcut(_shortcut_spec_from_args(args))
    print(status.summary)
    return 0 if status.matches else 2


def _cmd_shortcut_cleanup(_args: argparse.Namespace) -> int:
    from .shortcuts import LOCAL_SHORTCUT_PATH, remove_shortcut_paths

    removed = remove_shortcut_paths((LOCAL_SHORTCUT_PATH,))
    if removed:
        print("removed legacy shortcut paths: " + ", ".join(removed))
    else:
        print("no legacy shortcut paths found")
    return 0


def _load_cli_config(args: argparse.Namespace) -> WordpipeConfig:
    path = Path(args.config).expanduser() if getattr(args, "config", None) else None
    return load_config(path)


def _resolve_model_dir(args: argparse.Namespace, config: WordpipeConfig) -> Path:
    from .models import profile_installed, profile_runtime_dir

    raw = getattr(args, "model_dir", None)
    model_dir = Path(raw).expanduser() if raw else config.model_dir
    if model_dir is not None:
        return model_dir
    profile = getattr(args, "model_profile", None) or config.model_profile
    raw_model_root = getattr(args, "model_root", None)
    model_root = Path(raw_model_root).expanduser() if raw_model_root else config.model_root
    if model_root is not None and profile_installed(model_root, profile):
        return profile_runtime_dir(model_root, profile)
    raise SystemExit(
        "--model-dir is required when config.toml does not set model_dir and "
        f"profile {profile!r} is not installed. Run `wordpipe model-install "
        f"--profile {profile}` first."
    )


def _daemon_config_from_args(
    args: argparse.Namespace,
    file_config: WordpipeConfig,
    *,
    log_metrics_default: bool = False,
    insert_partial_default: bool = False,
):
    from .daemon import DaemonConfig

    return DaemonConfig(
        model_dir=_resolve_model_dir(args, file_config),
        asr_runtime=getattr(args, "asr_runtime", None) or file_config.asr_runtime,
        asr_worker_path=Path(args.asr_worker_path).expanduser()
        if getattr(args, "asr_worker_path", None)
        else file_config.asr_worker_path,
        dry_run_insertion=getattr(args, "dry_run_insertion", False)
        or file_config.dry_run_insertion,
        provider=getattr(args, "provider", None) or file_config.provider,
        num_threads=args.num_threads
        if getattr(args, "num_threads", None) is not None
        else file_config.num_threads,
        sample_rate=args.sample_rate
        if getattr(args, "sample_rate", None) is not None
        else file_config.sample_rate,
        input_device=parse_audio_device(args.input_device)
        if getattr(args, "input_device", None) is not None
        else file_config.input_device,
        partial_interval_seconds=args.partial_interval_seconds
        if getattr(args, "partial_interval_seconds", None) is not None
        else file_config.partial_interval_seconds,
        audio_chunk_seconds=args.audio_chunk_seconds
        if getattr(args, "audio_chunk_seconds", None) is not None
        else file_config.audio_chunk_seconds,
        queue_seconds=args.queue_seconds
        if getattr(args, "queue_seconds", None) is not None
        else file_config.queue_seconds,
        stats_interval_seconds=args.stats_interval_seconds
        if getattr(args, "stats_interval_seconds", None) is not None
        else file_config.stats_interval_seconds,
        enable_endpoint_detection=getattr(args, "endpoint", False)
        or file_config.enable_endpoint_detection,
        endpoint_rule1_min_trailing_silence=args.endpoint_rule1_min_trailing_silence
        if getattr(args, "endpoint_rule1_min_trailing_silence", None) is not None
        else file_config.endpoint_rule1_min_trailing_silence,
        endpoint_rule2_min_trailing_silence=args.endpoint_rule2_min_trailing_silence
        if getattr(args, "endpoint_rule2_min_trailing_silence", None) is not None
        else file_config.endpoint_rule2_min_trailing_silence,
        endpoint_rule3_min_utterance_length=args.endpoint_rule3_min_utterance_length
        if getattr(args, "endpoint_rule3_min_utterance_length", None) is not None
        else file_config.endpoint_rule3_min_utterance_length,
        spoken_punctuation=file_config.spoken_punctuation
        and not getattr(args, "no_spoken_punctuation", False),
        log_metrics=getattr(args, "log_metrics", False)
        or file_config.log_metrics
        or log_metrics_default,
        insert_partial_text=(
            file_config.insert_partial_text
            or insert_partial_default
            or getattr(args, "insert_partials", False)
        )
        and not getattr(args, "final_commit_only", False),
        stream_insert_delay_seconds=args.stream_insert_delay_seconds
        if getattr(args, "stream_insert_delay_seconds", None) is not None
        else file_config.stream_insert_delay_seconds,
    )


def _cmd_daemon(args: argparse.Namespace) -> int:
    from .daemon import run_daemon
    from .transcript import make_transcript_sink

    file_config = _load_cli_config(args)
    config = _daemon_config_from_args(args, file_config)
    return run_daemon(config, make_transcript_sink(args.overlay or file_config.overlay))


def _cmd_hotkey_daemon(args: argparse.Namespace) -> int:
    from .daemon import run_hotkey_daemon
    from .transcript import make_transcript_sink

    file_config = _load_cli_config(args)
    config = _daemon_config_from_args(args, file_config)
    return run_hotkey_daemon(
        config,
        mode=args.mode or file_config.mode,
        shortcut=args.shortcut or file_config.shortcut,
        manual_hotkey=args.manual_hotkey,
        transcript=make_transcript_sink(args.overlay or file_config.overlay),
    )


def _cmd_voice_keyboard(args: argparse.Namespace) -> int:
    from .daemon import run_hotkey_daemon, run_signal_hotkey_daemon
    from .transcript import make_transcript_sink

    file_config = _load_cli_config(args)
    config = _daemon_config_from_args(
        args,
        file_config,
        log_metrics_default=True,
        insert_partial_default=True,
    )
    overlay = args.overlay or file_config.overlay or "gtk"
    if args.signal_hotkey:
        return run_signal_hotkey_daemon(
            config,
            transcript=make_transcript_sink(overlay),
            pid_file=Path(args.pid_file).expanduser() if args.pid_file else None,
        )
    return run_hotkey_daemon(
        config,
        mode=args.mode or file_config.mode,
        shortcut=args.shortcut or file_config.shortcut,
        manual_hotkey=args.manual_hotkey,
        transcript=make_transcript_sink(overlay),
    )


def _cmd_voice_keyboard_toggle(args: argparse.Namespace) -> int:
    import signal

    from .daemon import default_voice_keyboard_pid_file

    pid_file = Path(args.pid_file).expanduser() if args.pid_file else default_voice_keyboard_pid_file()
    if _signal_voice_keyboard(pid_file, signal.SIGUSR1):
        return 0
    if not args.start_if_needed:
        raise _voice_keyboard_not_running_error(pid_file)
    _start_voice_keyboard_daemon(pid_file, args)
    if _signal_voice_keyboard(pid_file, signal.SIGUSR1):
        return 0
    raise _voice_keyboard_not_running_error(pid_file)


def _signal_voice_keyboard(pid_file: Path, signum: int) -> bool:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return False
    except ValueError:
        _remove_stale_pid_file(pid_file)
        return False
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        _remove_stale_pid_file(pid_file)
        return False
    except PermissionError as exc:
        raise RuntimeError(f"voice keyboard pid {pid} exists but cannot be signaled: {exc}") from exc
    return True


def _start_voice_keyboard_daemon(pid_file: Path, args: argparse.Namespace) -> None:
    log_file = (
        Path(args.daemon_log_file).expanduser()
        if getattr(args, "daemon_log_file", None)
        else _default_voice_keyboard_log_file()
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "wordpipe",
        "voice-keyboard",
        "--signal-hotkey",
        "--pid-file",
        str(pid_file),
    ]
    if getattr(args, "config", None):
        command.extend(["--config", str(Path(args.config).expanduser())])
    with log_file.open("a", encoding="utf-8") as log:
        print(
            f"\n--- wordpipe voice-keyboard start {time.strftime('%Y-%m-%d %H:%M:%S')} ---",
            file=log,
            flush=True,
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + float(args.start_timeout)
    while time.monotonic() < deadline:
        if pid_file.exists() and _pid_file_points_to_live_process(pid_file):
            return
        if process.poll() is not None:
            raise RuntimeError(
                "voice keyboard daemon exited before becoming ready; "
                f"command: {' '.join(command)}; log: {log_file}"
            )
        time.sleep(0.05)
    _terminate_starting_daemon(process)
    raise RuntimeError(
        f"voice keyboard daemon did not become ready within {args.start_timeout}s; "
        f"pid file: {pid_file}; log: {log_file}"
    )


def _terminate_starting_daemon(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _pid_file_points_to_live_process(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        _remove_stale_pid_file(pid_file)
        return False
    except PermissionError:
        return True
    return True


def _voice_keyboard_not_running_error(pid_file: Path) -> RuntimeError:
    return RuntimeError(
        f"voice keyboard is not running; missing or stale pid file {pid_file}. "
        "Start it with `wordpipe voice-keyboard --signal-hotkey`."
    )


def _default_voice_keyboard_log_file() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / "wordpipe" / "voice-keyboard.log"
    return Path.home() / ".cache" / "wordpipe" / "voice-keyboard.log"


def _remove_stale_pid_file(pid_file: Path) -> None:
    try:
        pid_file.unlink()
    except FileNotFoundError:
        return


def _cmd_config_example(_args: argparse.Namespace) -> int:
    print(DEFAULT_CONFIG, end="")
    return 0


def _add_asr_tuning_args(parser: argparse.ArgumentParser, *, worker_defaults: bool) -> None:
    parser.add_argument(
        "--partial-interval-seconds",
        type=_positive_float_arg,
        default=0.10 if worker_defaults else None,
        help="Minimum time between partial transcript updates.",
    )
    parser.add_argument(
        "--audio-chunk-seconds",
        type=_positive_float_arg,
        default=0.03 if worker_defaults else None,
        help="Microphone audio chunk size sent to the streaming recognizer.",
    )
    parser.add_argument(
        "--queue-seconds",
        type=_positive_float_arg,
        default=10.0 if worker_defaults else None,
        help="Maximum queued microphone audio before chunks are dropped.",
    )
    parser.add_argument(
        "--stats-interval-seconds",
        type=_positive_float_arg,
        default=1.0 if worker_defaults else None,
        help="Interval for diagnostic stats events.",
    )
    parser.add_argument(
        "--endpoint-rule1-min-trailing-silence",
        type=_positive_float_arg,
        default=0.55 if worker_defaults else None,
        help="Trailing silence before committing a phrase with speech.",
    )
    parser.add_argument(
        "--endpoint-rule2-min-trailing-silence",
        type=_positive_float_arg,
        default=0.35 if worker_defaults else None,
        help="Trailing silence endpoint rule for empty/no-speech streams.",
    )
    parser.add_argument(
        "--endpoint-rule3-min-utterance-length",
        type=_positive_float_arg,
        default=20.0 if worker_defaults else None,
        help="Maximum utterance length before endpointing.",
    )


def _add_runtime_args(parser: argparse.ArgumentParser, *, default: str | None) -> None:
    parser.add_argument(
        "--asr-runtime",
        choices=("parakeet", "sherpa"),
        default=default,
        help="ASR worker runtime. Parakeet is the new Rust runtime; sherpa is the legacy worker.",
    )
    parser.add_argument(
        "--asr-worker-path",
        help="Path to wordpipe-parakeet-worker. Defaults to target/debug or PATH.",
    )


def _add_model_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-profile",
        choices=("fast", "compact"),
        help="Built Wordpipe model profile to load when --model-dir is not set.",
    )
    parser.add_argument(
        "--model-root",
        help="Directory containing built Wordpipe model profiles.",
    )


def _add_insertion_mode_args(parser: argparse.ArgumentParser, *, partials_default: bool) -> None:
    if partials_default:
        parser.add_argument(
            "--final-commit-only",
            action="store_true",
            help="Wait until dictation stops before inserting recognized text.",
        )
    else:
        parser.add_argument(
            "--insert-partials",
            action="store_true",
            help="Insert appended partial text as ASR produces it instead of waiting for stop.",
        )
    parser.add_argument(
        "--stream-insert-delay-seconds",
        type=_non_negative_float_arg,
        help="Optional delay between word-like pieces inside one streamed partial burst.",
    )


def _add_shortcut_args(parser: argparse.ArgumentParser) -> None:
    from .shortcuts import DEFAULT_SHORTCUT_BINDING

    parser.add_argument(
        "--binding",
        default=DEFAULT_SHORTCUT_BINDING,
        help="GNOME accelerator string to bind.",
    )
    parser.add_argument(
        "--root",
        help="Project root used for the local development shortcut. Defaults to this source checkout.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wordpipe",
        description="Wayland-first GNOME dictation with streaming Parakeet/Nemotron ASR.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser(
        "probe",
        help="Check GNOME, portal, and Python runtime capabilities.",
    )
    probe.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    probe.set_defaults(func=_cmd_probe)

    asr = subparsers.add_parser(
        "asr-worker",
        help="Run the streaming ASR worker protocol on stdin/stdout.",
    )
    asr.add_argument(
        "--model-dir",
        required=True,
        help="Path to legacy sherpa-onnx Nemotron streaming model directory.",
    )
    asr.add_argument("--provider", default="cpu", help="ONNX Runtime provider.")
    asr.add_argument("--num-threads", type=_positive_int_arg, default=2)
    asr.add_argument("--sample-rate", type=_positive_int_arg, default=16000)
    asr.add_argument("--input-device", help="sounddevice input device index or name.")
    asr.add_argument(
        "--endpoint",
        action="store_true",
        help="Enable endpoint detection/reset. Disabled by default for raw ASR streaming.",
    )
    _add_asr_tuning_args(asr, worker_defaults=True)
    asr.set_defaults(func=_cmd_asr_worker)

    model_info = subparsers.add_parser(
        "model-info",
        help="Inspect a sherpa-onnx model directory and report the factory layout.",
    )
    model_info.add_argument(
        "--model-dir",
        required=True,
        help="Path to sherpa-onnx streaming model directory.",
    )
    model_info.set_defaults(func=_cmd_model_info)

    transcribe_file = subparsers.add_parser(
        "transcribe-file",
        help="Run streaming ASR over a 16 kHz mono PCM WAV file.",
    )
    transcribe_file.add_argument("--model-dir", required=True)
    transcribe_file.add_argument("--wav", required=True)
    transcribe_file.add_argument("--provider", default="cpu", help="ONNX Runtime provider.")
    transcribe_file.add_argument("--num-threads", type=_positive_int_arg, default=2)
    transcribe_file.add_argument("--sample-rate", type=_positive_int_arg, default=16000)
    transcribe_file.add_argument("--metrics", action="store_true", help="Print timing metrics.")
    transcribe_file.add_argument(
        "--flush-chunks",
        type=_non_negative_int_arg,
        default=0,
        help="Synthetic silence chunks to feed after the WAV before final result.",
    )
    transcribe_file.add_argument(
        "--endpoint-rule1-min-trailing-silence",
        type=_positive_float_arg,
        default=0.55,
        help="Trailing silence before committing a phrase with speech.",
    )
    transcribe_file.add_argument(
        "--endpoint-rule2-min-trailing-silence",
        type=_positive_float_arg,
        default=0.35,
        help="Trailing silence endpoint rule for empty/no-speech streams.",
    )
    transcribe_file.add_argument(
        "--endpoint-rule3-min-utterance-length",
        type=_positive_float_arg,
        default=20.0,
        help="Maximum utterance length before endpointing.",
    )
    transcribe_file.set_defaults(func=_cmd_transcribe_file)

    listen_test = subparsers.add_parser(
        "listen-test",
        help="Open the microphone and print live partials/commits with RTF metrics.",
    )
    listen_test.add_argument("--model-dir")
    _add_model_selection_args(listen_test)
    listen_test.add_argument("--config", help="Path to config.toml.")
    _add_runtime_args(listen_test, default="parakeet")
    listen_test.add_argument("--provider", default="cpu", help="ONNX Runtime provider.")
    listen_test.add_argument("--num-threads", type=_positive_int_arg, default=2)
    listen_test.add_argument("--sample-rate", type=_positive_int_arg, default=16000)
    listen_test.add_argument("--input-device", help="Input device selector; Parakeet also accepts cpal:N.")
    listen_test.add_argument(
        "--endpoint",
        action="store_true",
        help="Enable endpoint detection/reset. Disabled by default for raw ASR testing.",
    )
    listen_test.add_argument(
        "--duration",
        type=_positive_float_arg,
        help="Stop after this many seconds. Default is to run until interrupted.",
    )
    listen_test.add_argument("--json", action="store_true", help="Print raw JSON events.")
    listen_test.add_argument(
        "--full-hypotheses",
        action="store_true",
        help="Print full hypotheses instead of only newly appended suffixes.",
    )
    _add_asr_tuning_args(listen_test, worker_defaults=True)
    listen_test.set_defaults(func=_cmd_listen_test)

    stream_file_test = subparsers.add_parser(
        "stream-file-test",
        help="Feed a WAV through the streaming recognizer and print partial/stats events.",
    )
    stream_file_test.add_argument("--model-dir")
    _add_model_selection_args(stream_file_test)
    stream_file_test.add_argument("--config", help="Path to config.toml.")
    stream_file_test.add_argument("--wav", required=True)
    stream_file_test.add_argument("--provider", default="cpu", help="ONNX Runtime provider.")
    stream_file_test.add_argument("--num-threads", type=_positive_int_arg, default=2)
    stream_file_test.add_argument("--sample-rate", type=_positive_int_arg, default=16000)
    _add_runtime_args(stream_file_test, default="parakeet")
    stream_file_test.add_argument("--chunk-seconds", type=_positive_float_arg, default=0.56)
    stream_file_test.add_argument(
        "--flush-chunks",
        type=_non_negative_int_arg,
        default=3,
        help="Synthetic silence chunks to feed after the WAV before final commit.",
    )
    stream_file_test.add_argument(
        "--reset-on-endpoint",
        action="store_true",
        help="Reset the recognizer whenever endpoint detection fires.",
    )
    stream_file_test.add_argument("--json", action="store_true", help="Print raw JSON events.")
    _add_asr_tuning_args(stream_file_test, worker_defaults=True)
    stream_file_test.set_defaults(func=_cmd_stream_file_test)

    audio_devices = subparsers.add_parser(
        "audio-devices",
        help="List available input devices.",
    )
    audio_devices.add_argument(
        "--backend",
        choices=("sounddevice", "parakeet"),
        default="sounddevice",
        help="Audio backend to query. Use parakeet to list the Rust/CPAL worker devices.",
    )
    audio_devices.add_argument(
        "--asr-worker-path",
        help="Path to wordpipe-parakeet-worker when --backend parakeet is used.",
    )
    audio_devices.set_defaults(func=_cmd_audio_devices)

    record_test = subparsers.add_parser(
        "record-test",
        help="Record raw microphone audio to a 16 kHz mono WAV for diagnostics.",
    )
    record_test.add_argument("--output", default="/tmp/wordpipe-record-test.wav")
    record_test.add_argument("--duration", type=_positive_float_arg, default=5.0)
    record_test.add_argument("--sample-rate", type=_positive_int_arg, default=16000)
    record_test.add_argument("--input-device", help="sounddevice input device index or name.")
    record_test.set_defaults(func=_cmd_record_test)

    download_model = subparsers.add_parser(
        "download-model",
        help="Download the default Nemotron int8 streaming model from Hugging Face.",
    )
    download_model.add_argument(
        "--repo-id",
        default="csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        help="Hugging Face model repo ID.",
    )
    download_model.add_argument(
        "--output-dir",
        default="models",
        help="Directory that will contain the downloaded model directory.",
    )
    download_model.add_argument(
        "--include-test-wavs",
        action="store_true",
        help="Also download the small upstream test WAV files.",
    )
    download_model.add_argument("--force", action="store_true", help="Redownload existing files.")
    download_model.set_defaults(func=_cmd_download_model)

    model_profiles = subparsers.add_parser(
        "model-profiles",
        help="List selectable Wordpipe model profiles and installation state.",
    )
    model_profiles.add_argument("--config", help="Path to config.toml.")
    model_profiles.add_argument("--model-root", help="Directory containing Wordpipe model profiles.")
    model_profiles.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    model_profiles.set_defaults(func=_cmd_model_profiles)

    model_install = subparsers.add_parser(
        "model-install",
        help="Install a selectable Wordpipe profile from prebuilt Hugging Face ONNX files.",
    )
    model_install.add_argument("--config", help="Path to config.toml.")
    model_install.add_argument(
        "--streaming-latency-ms", type=int, default=560,
        help="Streaming lookahead mode. Generic exports share one model across both sizes.",
    )
    model_install.add_argument("--dynamic-streaming", action="store_true",
                              help="With --build-from-nemo, export one generic encoder for both modes.")
    model_install.add_argument(
        "--profile",
        choices=("fast", "compact", "nemo-q8"),
        help="Profile to build. Defaults to config.toml model_profile.",
    )
    model_install.add_argument(
        "--model-family",
        choices=("all", "multilingual", "english"),
        default="multilingual",
        help="Checkpoint family to install. Defaults to multilingual; pass english for English-only.",
    )
    model_install.add_argument(
        "--model-root",
        help="Directory containing Wordpipe model profiles. Defaults to config.toml model_root.",
    )
    model_install.add_argument(
        "--source",
        help=(
            "Built Wordpipe profile directory/archive. With --build-from-nemo this may also "
            "be a local .nemo path or Hugging Face repo id."
        ),
    )
    model_install.add_argument(
        "--prebuilt-repo",
        help="Override the Hugging Face repo containing this profile's raw ONNX files.",
    )
    model_install.add_argument(
        "--build-from-nemo",
        action="store_true",
        help="Run the full NeMo export/optimization pipeline instead of downloading a prebuilt profile.",
    )
    model_install.add_argument(
        "--source-output",
        help="Where to store a downloaded .nemo source checkpoint.",
    )
    model_install.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for the NeMo export/build pipeline.",
    )
    model_install.add_argument("--force", action="store_true", help="Overwrite existing profile output.")
    model_install.add_argument(
        "--force-source",
        action="store_true",
        help="Redownload the source .nemo even when it already exists.",
    )
    model_install.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the build command without running the export pipeline.",
    )
    model_install.add_argument(
        "--keep-build-dir",
        action="store_true",
        help="Keep intermediate files under model_root/build after a successful export.",
    )
    model_install.set_defaults(func=_cmd_model_install)

    type_text = subparsers.add_parser(
        "type-text",
        help="Insert text using the keyboard insertion backend.",
    )
    type_text.add_argument("text", help="Text to insert.")
    type_text.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated keysyms instead of opening a portal session.",
    )
    type_text.set_defaults(func=_cmd_type_text)

    shortcut_status = subparsers.add_parser(
        "shortcut-status",
        help="Check the GNOME custom shortcut used to toggle dictation.",
    )
    _add_shortcut_args(shortcut_status)
    shortcut_status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    shortcut_status.set_defaults(func=_cmd_shortcut_status)

    shortcut_install = subparsers.add_parser(
        "shortcut-install",
        help="Install or repair the GNOME custom shortcut used to toggle dictation.",
    )
    _add_shortcut_args(shortcut_install)
    shortcut_install.set_defaults(func=_cmd_shortcut_install)

    shortcut_cleanup = subparsers.add_parser(
        "shortcut-cleanup",
        help="Remove legacy GNOME custom shortcuts now handled by the Shell extension.",
    )
    shortcut_cleanup.set_defaults(func=_cmd_shortcut_cleanup)

    config_example = subparsers.add_parser(
        "config-example",
        help="Print an example XDG config.toml.",
    )
    config_example.set_defaults(func=_cmd_config_example)

    voice_keyboard = subparsers.add_parser(
        "voice-keyboard",
        help="Run Wordpipe as a global-hotkey voice keyboard for the focused text field.",
    )
    voice_keyboard.add_argument(
        "--model-dir",
        help="Path to Parakeet/Nemotron model directory, or legacy sherpa model when --asr-runtime sherpa.",
    )
    _add_model_selection_args(voice_keyboard)
    voice_keyboard.add_argument("--config", help="Path to config.toml.")
    voice_keyboard.add_argument(
        "--mode",
        choices=("hold", "toggle"),
        help="Shortcut behavior. Hold starts on activation and stops on deactivation.",
    )
    voice_keyboard.add_argument(
        "--shortcut",
        help="Preferred GlobalShortcuts trigger string.",
    )
    voice_keyboard.add_argument(
        "--manual-hotkey",
        action="store_true",
        help="Read manual commands from stdin instead of opening GlobalShortcuts.",
    )
    voice_keyboard.add_argument(
        "--signal-hotkey",
        action="store_true",
        help="Use SIGUSR1/pid-file toggling instead of the GlobalShortcuts portal.",
    )
    voice_keyboard.add_argument(
        "--pid-file",
        help="Pid file used by --signal-hotkey and voice-keyboard-toggle.",
    )
    voice_keyboard.add_argument(
        "--dry-run-insertion",
        action="store_true",
        help="Print keyboard events instead of opening a portal keyboard session.",
    )
    voice_keyboard.add_argument("--provider", help="ONNX Runtime provider.")
    _add_runtime_args(voice_keyboard, default=None)
    voice_keyboard.add_argument("--num-threads", type=_positive_int_arg)
    voice_keyboard.add_argument("--sample-rate", type=_positive_int_arg)
    voice_keyboard.add_argument("--input-device", help="Input device selector; Parakeet also accepts cpal:N.")
    _add_asr_tuning_args(voice_keyboard, worker_defaults=False)
    voice_keyboard.add_argument(
        "--no-spoken-punctuation",
        action="store_true",
        help="Insert raw ASR text instead of converting spoken punctuation commands.",
    )
    _add_insertion_mode_args(voice_keyboard, partials_default=True)
    voice_keyboard.add_argument(
        "--endpoint",
        action="store_true",
        help="Enable endpoint detection/reset. Disabled by default for raw ASR streaming.",
    )
    voice_keyboard.add_argument("--log-metrics", action="store_true", help="Show ASR timing metrics.")
    voice_keyboard.add_argument(
        "--overlay",
        choices=("stderr", "gtk"),
        help="Where partial transcript/status text is shown. Defaults to config overlay.",
    )
    voice_keyboard.set_defaults(func=_cmd_voice_keyboard)

    voice_keyboard_toggle = subparsers.add_parser(
        "voice-keyboard-toggle",
        help="Toggle a running --signal-hotkey voice keyboard daemon.",
    )
    voice_keyboard_toggle.add_argument(
        "--pid-file",
        help="Pid file written by voice-keyboard --signal-hotkey.",
    )
    voice_keyboard_toggle.add_argument(
        "--config",
        help="Path to config.toml for --start-if-needed.",
    )
    voice_keyboard_toggle.add_argument(
        "--start-if-needed",
        action="store_true",
        help="Start a resident --signal-hotkey daemon when the pid file is missing or stale.",
    )
    voice_keyboard_toggle.add_argument(
        "--start-timeout",
        type=_positive_float_arg,
        default=30.0,
        help="Seconds to wait for a newly started daemon to become ready.",
    )
    voice_keyboard_toggle.add_argument(
        "--daemon-log-file",
        help="Log file for a daemon started by --start-if-needed.",
    )
    voice_keyboard_toggle.set_defaults(func=_cmd_voice_keyboard_toggle)

    daemon = subparsers.add_parser(
        "daemon",
        help="Run the MVP dictation loop: ASR subprocess plus text insertion.",
    )
    daemon.add_argument(
        "--model-dir",
        help="Path to Parakeet/Nemotron model directory, or legacy sherpa model when --asr-runtime sherpa.",
    )
    _add_model_selection_args(daemon)
    daemon.add_argument("--config", help="Path to config.toml.")
    daemon.add_argument(
        "--dry-run-insertion",
        action="store_true",
        help="Print keyboard events instead of opening a portal keyboard session.",
    )
    daemon.add_argument("--provider", help="ONNX Runtime provider.")
    _add_runtime_args(daemon, default=None)
    daemon.add_argument("--num-threads", type=_positive_int_arg)
    daemon.add_argument("--sample-rate", type=_positive_int_arg)
    daemon.add_argument("--input-device", help="Input device selector; Parakeet also accepts cpal:N.")
    _add_asr_tuning_args(daemon, worker_defaults=False)
    daemon.add_argument(
        "--no-spoken-punctuation",
        action="store_true",
        help="Insert raw ASR text instead of converting spoken punctuation commands.",
    )
    _add_insertion_mode_args(daemon, partials_default=False)
    daemon.add_argument(
        "--endpoint",
        action="store_true",
        help="Enable endpoint detection/reset. Disabled by default for raw ASR streaming.",
    )
    daemon.add_argument("--log-metrics", action="store_true", help="Log ASR timing metrics.")
    daemon.add_argument(
        "--overlay",
        choices=("stderr", "gtk"),
        help="Where partial transcript/status text is shown.",
    )
    daemon.set_defaults(func=_cmd_daemon)

    hotkey_daemon = subparsers.add_parser(
        "hotkey-daemon",
        help="Run dictation controlled by a GNOME GlobalShortcuts portal hotkey.",
    )
    hotkey_daemon.add_argument(
        "--model-dir",
        help="Path to Parakeet/Nemotron model directory, or legacy sherpa model when --asr-runtime sherpa.",
    )
    _add_model_selection_args(hotkey_daemon)
    hotkey_daemon.add_argument("--config", help="Path to config.toml.")
    hotkey_daemon.add_argument(
        "--mode",
        choices=("hold", "toggle"),
        help="Shortcut behavior. Hold starts on activation and stops on deactivation.",
    )
    hotkey_daemon.add_argument(
        "--shortcut",
        help="Preferred GlobalShortcuts trigger string.",
    )
    hotkey_daemon.add_argument(
        "--manual-hotkey",
        action="store_true",
        help="Read manual commands from stdin instead of opening GlobalShortcuts.",
    )
    hotkey_daemon.add_argument(
        "--dry-run-insertion",
        action="store_true",
        help="Print keyboard events instead of opening a portal keyboard session.",
    )
    hotkey_daemon.add_argument("--provider", help="ONNX Runtime provider.")
    _add_runtime_args(hotkey_daemon, default=None)
    hotkey_daemon.add_argument("--num-threads", type=_positive_int_arg)
    hotkey_daemon.add_argument("--sample-rate", type=_positive_int_arg)
    hotkey_daemon.add_argument("--input-device", help="Input device selector; Parakeet also accepts cpal:N.")
    _add_asr_tuning_args(hotkey_daemon, worker_defaults=False)
    hotkey_daemon.add_argument(
        "--no-spoken-punctuation",
        action="store_true",
        help="Insert raw ASR text instead of converting spoken punctuation commands.",
    )
    _add_insertion_mode_args(hotkey_daemon, partials_default=False)
    hotkey_daemon.add_argument(
        "--endpoint",
        action="store_true",
        help="Enable endpoint detection/reset. Disabled by default for raw ASR streaming.",
    )
    hotkey_daemon.add_argument("--log-metrics", action="store_true", help="Log ASR timing metrics.")
    hotkey_daemon.add_argument(
        "--overlay",
        choices=("stderr", "gtk"),
        help="Where partial transcript/status text is shown.",
    )
    hotkey_daemon.set_defaults(func=_cmd_hotkey_daemon)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"wordpipe error: {exc}", file=sys.stderr)
        return 1


__all__ = ["ProbeResult", "build_parser", "main"]
