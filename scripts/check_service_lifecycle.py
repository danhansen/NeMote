#!/usr/bin/env python3
"""Test the production D-Bus service on a private bus, without microphone/network."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from gi.repository import Gio, GLib


def main():
    if os.environ.get("NEMOTE_TEST_BUS") != "1":
        raise SystemExit("Run with dbus-run-session -- env NEMOTE_TEST_BUS=1; never use the desktop bus")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--legacy-config", action="store_true", help="Exercise automatic Wordpipe settings migration")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nemote-service-test-") as directory:
        root = Path(directory)
        models = root / "models"
        for family, size in (("en-", 699872960), ("", 742090464)):
            path = models / f"nemotron-nemo-{family}q8/model.gguf"
            path.parent.mkdir(parents=True)
            with path.open("wb") as output:
                output.write(b"GGUF")
                output.truncate(size)  # Sparse fixture, no actual weights or disk cost.
        worker = root / "worker"
        fixture_root = Path(__file__).resolve().parents[1] / "tests/fixtures"
        shutil.copy2(fixture_root / "mock_runtime_worker.py", worker)
        worker.chmod(0o755)
        installer = root / "installer"
        shutil.copy2(fixture_root / "mock_model_installer.py", installer)
        installer.chmod(0o755)
        pnc_installer = root / "nemote-companion-install"
        shutil.copy2(fixture_root / "mock_pnc_installer.py", pnc_installer)
        pnc_installer.chmod(0o755)
        config = root / "config/nemote/service.json" if args.legacy_config else root / "service.json"
        saved_config = root / "config/wordpipe/service.json" if args.legacy_config else config
        saved_config.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps({"backend": "parakeet", "model_profile": "fast", "model_family": "english",
            "model_root": str(models), "worker_path": str(worker), "model_installer_path": str(installer),
            "input_device": "old-cpal-index", "insert_partials": True})
        saved_config.write_text(original)
        log = (root / "service.log").open("w+")
        command = [str(args.service.resolve())]
        if not args.legacy_config:
            command += ["--config", str(config)]
        env = dict(os.environ, XDG_CONFIG_HOME=str(root / "config"), XDG_DATA_HOME=str(root / "data"))
        process = subprocess.Popen(command, env=env, stderr=log)
        connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        signals = []
        subscription = connection.signal_subscribe("dev.nemote.Service", "dev.nemote.Service1", None,
            "/dev/nemote/Service", None, Gio.DBusSignalFlags.NONE,
            lambda bus, sender, path, interface, name, values: signals.append((name, values.unpack())))

        def pump():
            context = GLib.MainContext.default()
            while context.pending():
                context.iteration(False)

        def call(name, params=None):
            return connection.call_sync("dev.nemote.Service", "/dev/nemote/Service", "dev.nemote.Service1",
                name, params, None, Gio.DBusCallFlags.NO_AUTO_START, 2000, None).unpack()

        def wait(predicate):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                pump()
                if process.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read())
                try:
                    state = call("GetState")[0]
                    if predicate(state):
                        return state
                except GLib.Error:
                    pass
                time.sleep(0.01)
            raise AssertionError("Service did not reach expected state")

        def runtime(options):
            call("SetRuntimeOptions", GLib.Variant("(a{sv})", (options,)))

        def dead(pid):
            try:
                return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] == "Z"
            except FileNotFoundError:
                return True

        try:
            wait(lambda state: True)
            view = call("GetConfig")[0]
            assert "show_overlay" not in view
            call("SetInsertionOptions", GLib.Variant("(a{sv})", ({"show_overlay": GLib.Variant("b", True)},)))
            assert call("GetConfig")[0] == view, "Retired overlay option must be a compatibility no-op"
            assert view["backend"] == "nemo-speech" and view["model_profile"] == "nemo-q8"
            assert view["input_device"] == "" and view["supported_streaming_latencies_ms"] == [80, 160, 560, 1120]
            assert view["pnc"] is False and view["pnc_model_path"] == ""
            assert [item["title"] for item in call("ListModelProfiles")[0]] == ["English (lower WER)", "Multilingual"]
            assert [item["id"] for item in call("ListPncModels")[0]] == ["pnc-bert", "pnc-distilbert"]
            # Install independently of ASR and without enabling PnC or changing its selection.
            empty_models = root / "empty-models"
            runtime({"model_root": GLib.Variant("s", str(empty_models))})
            call("InstallModel", GLib.Variant("(s)", ("pnc-bert",)))
            wait(lambda state: state["installing"])
            wait(lambda state: not state["installing"])
            assert call("ListPncModels")[0][0]["installed"] is True
            assert call("GetState")[0]["selected_model_installed"] is False
            assert call("GetConfig")[0]["pnc"] is False
            runtime({"model_root": GLib.Variant("s", str(models))})
            devices = call("ListInputDevices")[0]
            assert devices[0]["name"] == "Fixture microphone"
            for chunk in (80, 160, 560, 1120):
                runtime({"streaming_latency_ms": GLib.Variant("u", chunk)})
                call("Start")
                state = wait(lambda state: state["listening"] and state["model_loaded"] and state["partial_text"] == "I have $20")
                pid = state["last_metrics"].get("worker_pid")
                assert state["last_metrics"]["chunk_samples"] == chunk * 16
                try:
                    runtime({"itn": GLib.Variant("b", True)})
                    raise AssertionError("Active runtime mutation was accepted")
                except GLib.Error:
                    pass
                call("Stop")
                wait(lambda state: not state["stopping"] and not state["listening"] and state["last_commit_text"] == "I have $20")
                pump()
                commits = [index for index, (name, values) in enumerate(signals) if name == "Commit" and values[0] == state["session_id"]]
                stops = [index for index, (name, values) in enumerate(signals) if name == "SessionStopped" and values[0] == state["session_id"]]
                assert commits and stops and commits[-1] < stops[-1], "EOF commit must precede stop"
                call("Start")
                state = wait(lambda state: state["listening"] and state["partial_text"] == "I have $20")
                assert state["last_metrics"]["worker_pid"] == pid, "Warm session rebuilt the worker"
                call("Stop")
                wait(lambda state: not state["stopping"])
            before = call("GetConfig")[0]
            try:
                runtime({"streaming_latency_ms": GLib.Variant("u", 240)})
                raise AssertionError("Invalid chunk size accepted")
            except GLib.Error:
                assert call("GetConfig")[0] == before
            runtime({"itn": GLib.Variant("b", True), "endpoint_mode": GLib.Variant("s", "preview")})
            assert json.loads(config.read_text())["itn"] is True
            before = call("GetConfig")[0]
            try:
                runtime({"pnc": GLib.Variant("b", True), "pnc_model_path": GLib.Variant("s", str(root / "missing.gguf"))})
                raise AssertionError("Missing PnC override was accepted")
            except GLib.Error:
                assert call("GetConfig")[0] == before
            runtime({"pnc": GLib.Variant("b", True), "pnc_variant": GLib.Variant("s", "distilbert")})
            assert call("GetState")[0]["selected_pnc_model_installed"] is False
            try:
                call("Start")
                raise AssertionError("Missing PnC model triggered worker startup")
            except GLib.Error as error:
                assert "preferences" in str(error)
            assert call("GetState")[0]["listening"] is False
            assert call("GetState")[0]["loading_model"] is False
            # Failed explicit installation is recoverable and progress reaches the client.
            failure = models / ".fail-pnc-install"
            failure.touch()
            call("InstallModel", GLib.Variant("(s)", ("pnc-distilbert",)))
            wait(lambda state: state["installing"])
            try:
                call("Start")
                raise AssertionError("Dictation started during PnC installation")
            except GLib.Error:
                pass
            state = wait(lambda state: not state["installing"])
            assert state["last_install_progress"]["phase"] == "error"
            assert "Fixture download failed" in state["last_error"]
            failure.unlink()
            call("InstallModel", GLib.Variant("(s)", ("pnc-distilbert",)))
            wait(lambda state: state["installing"])
            state = wait(lambda state: not state["installing"])
            assert state["last_install_progress"]["phase"] == "complete"
            assert state["selected_pnc_model_installed"] is True
            assert call("ListPncModels")[0][1]["installed"] is True
            assert any(name == "InstallProgress" and values[0] == "pnc-distilbert" and
                       values[1].get("completed_bytes") == 10 for name, values in signals)
            # Older per-ASR downloads are reused in place.
            pnc_model = models / "pnc/pnc-distilbert-en.q8_0.gguf"
            legacy_model = models / "nemotron-nemo-en-q8" / pnc_model.name
            pnc_model.rename(legacy_model)
            assert call("ListPncModels")[0][1]["installed"] is True
            call("Start")
            state = wait(lambda state: state.get("loading_message") == "PnC fixture setup")
            assert state["loading_model"] is True
            state = wait(lambda state: state["listening"] and state["model_loaded"] and state["last_metrics"].get("pnc") is True)
            assert state["loading_message"] == ""
            assert state["last_metrics"]["pnc_model_path"] == str(legacy_model)
            assert state["last_metrics"]["pnc_variant"] == "distilbert"
            call("Stop")
            wait(lambda state: not state["listening"] and not state["stopping"])
            # Only the SDK parses model contents; this private-bus fixture
            # checks persistence, forwarding and worker restarts, not weights.
            runtime({"pnc_model_path": GLib.Variant("s", str(worker)), "pnc": GLib.Variant("b", True)})
            saved = json.loads(config.read_text())
            assert saved["pnc"] is True and saved["pnc_model_path"] == str(worker)
            call("Start")
            state = wait(lambda state: state["listening"] and state["model_loaded"] and state["last_metrics"].get("pnc_model_path") == str(worker))
            assert state["last_metrics"]["pnc_model_path"] == str(worker)
            call("Stop")
            wait(lambda state: not state["listening"] and not state["stopping"])
            runtime({"pnc": GLib.Variant("b", False)})
            if args.legacy_config:
                assert saved_config.read_text() == original, "Legacy settings were overwritten"
            call("SetModelProfile", GLib.Variant("(s)", ("nemo-q8",)))
            assert call("GetConfig")[0]["model_family"] == "multilingual"
            call("SetModelProfile", GLib.Variant("(s)", ("nemo-q8-english",)))
            for kind in ("error-worker", "crash-worker"):
                broken = root / kind
                shutil.copy2(worker, broken)
                runtime({"worker_path": GLib.Variant("s", str(broken))})
                call("Start")
                wait(lambda state: not state["listening"] and not state["stopping"] and bool(state["last_error"]))
                runtime({"worker_path": GLib.Variant("s", str(worker))})
                call("Start")
                wait(lambda state: state["listening"] and state["model_loaded"])
                call("Stop")
                wait(lambda state: not state["listening"] and not state["stopping"])
            call("InstallModel", GLib.Variant("(s)", ("nemo-q8-english",)))
            state = wait(lambda state: "child_pid" in state["last_install_progress"])
            install_pid = state["last_install_progress"]["installer_pid"]
            child_pid = state["last_install_progress"]["child_pid"]
            # Only a registered Shell connection owns the service lifetime.
            shell = Gio.DBusConnection.new_for_address_sync(os.environ["DBUS_SESSION_BUS_ADDRESS"],
                Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION, None, None)
            shell.call_sync("dev.nemote.Service", "/dev/nemote/Service", "dev.nemote.Service1", "RegisterShellClient",
                None, None, Gio.DBusCallFlags.NO_AUTO_START, 2000, None)
            shell.close_sync(None)
            assert process.wait(timeout=5) == 0
            deadline = time.monotonic() + 3
            while not (dead(install_pid) and dead(child_pid)) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert dead(install_pid) and dead(child_pid), "Installer descendants survived shutdown"
            print("D-Bus migration, settings, device enumeration, all chunks, resets, EOF ordering, persistence and shutdown passed")
        finally:
            connection.signal_unsubscribe(subscription)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            log.close()


if __name__ == "__main__":
    main()
