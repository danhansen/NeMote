#!/usr/bin/env python3
"""Run under dbus-run-session; never connect this probe to the desktop bus."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from gi.repository import Gio, GLib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--dynamic-streaming", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="wordpipe-lifecycle-") as temporary:
        root = Path(temporary)
        model_root = root / "models"
        for latency in ((560,) if args.dynamic_streaming else (560, 1120)):
            directory = (model_root if latency == 560 else model_root / "1120ms") / "nemotron-wordpipe-en-fast-fp32-projected"
            directory.mkdir(parents=True)
            for name in ("encoder.onnx", "decoder_joint.onnx", "tokenizer.model"):
                (directory / name).write_bytes(b"lifecycle-only fixture")
            frames = latency // 80
            payload = {
                "model_family": "english", "right_context": frames - 1,
                "projected_cache": True, "dynamic_quint8_quantization": False,
                "fixed_streaming_shapes": {
                    "input_frames": frames * 8 + 9, "output_frames": frames,
                    "num_layers": 24, "cache_len": 70, "hidden_dim": 1024,
                    "conv_context": 8,
                },
            }
            if args.dynamic_streaming:
                payload.pop("fixed_streaming_shapes")
                payload.update({
                    "num_encoder_layers": 24, "hidden_dim": 1024, "conv_context": 8,
                    "cache_shapes": {"cache_last_channel": [24, 1, 70, 1024],
                                     "cache_last_time": [24, 1, 1024, 8]},
                    "dynamic_streaming": {
                        "format": 1, "shape_derived_attention_context": True,
                        "subsampling_factor": 8, "mel_frames_overhead": 9, "cache_len": 70,
                        "default_chunk_frames": 7, "supported_chunk_frames": [1, 2, 7, 14],
                    },
                })
            (directory / "config.json").write_text(json.dumps(payload))
        worker = root / "worker"
        shutil.copy2(Path(__file__).resolve().parents[1] / "tests/fixtures/mock_runtime_worker.py", worker)
        worker.chmod(0o755)
        installer = root / "installer"
        shutil.copy2(Path(__file__).resolve().parents[1] / "tests/fixtures/mock_model_installer.py", installer)
        installer.chmod(0o755)
        config = root / "service.json"
        config.write_text(json.dumps({"model_root": str(model_root), "model_family": "english",
                                     "model_profile": "fast", "worker_path": str(worker),
                                     "model_installer_path": str(installer)}))
        process = subprocess.Popen([str(args.service.resolve()), "--config", str(config)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)

        def call(method, parameters=None):
            return connection.call_sync(
                "dev.wordpipe.Service", "/dev/wordpipe/Service", "dev.wordpipe.Service1",
                method, parameters, None, Gio.DBusCallFlags.NO_AUTO_START, 1000, None).unpack()

        def wait_state(predicate):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read())
                try:
                    state = call("GetState")[0]
                except GLib.Error:
                    time.sleep(0.01)
                    continue
                if predicate(state):
                    return state
                time.sleep(0.01)
            raise AssertionError("service failed to reach expected state")

        try:
            wait_state(lambda state: True)
            call("RegisterShellClient")
            call("Start")
            state = wait_state(lambda state: state["model_loaded"] and state["listening"])
            try:
                call("SetRuntimeOptions", GLib.Variant("(a{sv})", ({
                    "streaming_latency_ms": GLib.Variant("u", 1120)},)))
                raise AssertionError("active dictation allowed a runtime rebuild")
            except GLib.Error as error:
                assert "stop dictation" in str(error)
            assert call("GetConfig")[0]["streaming_latency_ms"] == 560
            call("Stop")
            state = wait_state(lambda state: not state["listening"] and not state["stopping"])
            previous_pid = state["last_metrics"]["worker_pid"]
            try:
                call("SetRuntimeOptions", GLib.Variant("(a{sv})", ({
                    "streaming_latency_ms": GLib.Variant("u", 1120),
                    "language": GLib.Variant("s", "invalid-language")},)))
                raise AssertionError("invalid runtime options were accepted")
            except GLib.Error:
                pass
            assert call("GetConfig")[0]["streaming_latency_ms"] == 560
            if args.dynamic_streaming:
                assert call("GetConfig")[0]["supported_streaming_latencies_ms"] == [80, 160, 560, 1120]
                try:
                    call("SetRuntimeOptions", GLib.Variant("(a{sv})", ({
                        "streaming_latency_ms": GLib.Variant("u", 240)},)))
                    raise AssertionError("unsupported context was accepted")
                except GLib.Error:
                    pass
            latencies = (80, 160, 1120, 560) if args.dynamic_streaming else (1120, 560, 1120, 560)
            for latency in latencies:
                call("SetRuntimeOptions", GLib.Variant("(a{sv})", ({
                    "streaming_latency_ms": GLib.Variant("u", latency)},)))
                state = wait_state(lambda state: state["model_loaded"] and
                                   state["last_metrics"].get("chunk_samples") == latency * 16)
                current_pid = state["last_metrics"]["worker_pid"]
                assert current_pid != previous_pid
                try:
                    os.kill(previous_pid, 0)
                    raise AssertionError("old worker survived runtime replacement")
                except ProcessLookupError:
                    pass
                time.sleep(0.05)
                assert call("GetState")[0]["model_loaded"], "stale EOF cleared new worker"
                assert call("GetConfig")[0]["streaming_latency_ms"] == latency
                assert state["last_metrics"]["cache_dir"] == str(model_root / "runtime-cache")
                previous_pid = current_pid
            print(f"Verified mode replacement {latencies}, stale EOF guards, persistence, and active-session guard")
            call("InstallModel", GLib.Variant("(s)", ("fast-english",)))
            installing = wait_state(lambda state: state["installing"] and
                "child_pid" in state["last_install_progress"])
            installer_pid = installing["last_install_progress"]["installer_pid"]
            child_pid = installing["last_install_progress"]["child_pid"]
            call("Shutdown")
            assert process.wait(timeout=5) == 0, "service did not exit successfully"

            def running(pid):
                try:
                    return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
                except FileNotFoundError:
                    return False

            assert not running(previous_pid), "worker survived service shutdown"
            assert not running(installer_pid), "installer survived service shutdown"
            assert not running(child_pid), "installer descendant survived service shutdown"
            print("Verified successful shutdown and cancellation of the owned installer process tree")

            process = subprocess.Popen([str(args.service.resolve()), "--config", str(config)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            wait_state(lambda state: True)
            call("RegisterShellClient")
            connection.close_sync(None)
            assert process.wait(timeout=5) == 0, "Shell disconnect left the service running"
            print("Verified automatic service exit when the registered Shell connection disappears")
        finally:
            if process.poll() is None:
                try:
                    call("Shutdown")
                    process.wait(timeout=5)
                except (GLib.Error, subprocess.TimeoutExpired):
                    process.terminate()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
