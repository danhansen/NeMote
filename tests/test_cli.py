from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wordpipe.cli import (
    _cmd_listen_test,
    _cmd_model_install,
    _cmd_stream_file_test,
    _cmd_voice_keyboard_toggle,
    _start_voice_keyboard_daemon,
    _resolve_model_dir,
    build_parser,
    main,
)
from wordpipe.config import WordpipeConfig
from wordpipe.models import DEFAULT_NEMO_SOURCE_FILENAME, PROFILE_COMPLETION_MARKER, profile_runtime_dir


def _args(
    *,
    model_dir: str | None = None,
    model_profile: str | None = None,
    model_root: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        model_dir=model_dir,
        model_profile=model_profile,
        model_root=model_root,
    )


def _install_marker(model_root: Path, profile: str) -> Path:
    runtime_dir = profile_runtime_dir(model_root, profile)
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
    (runtime_dir / "encoder.onnx").write_text("encoder", encoding="utf-8")
    (runtime_dir / "decoder_joint.onnx").write_text("decoder", encoding="utf-8")
    return runtime_dir


def _toggle_args(
    pid_file: Path,
    *,
    start_if_needed: bool = False,
    config: str | None = None,
    start_timeout: float = 30.0,
    daemon_log_file: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        pid_file=str(pid_file),
        start_if_needed=start_if_needed,
        config=config,
        start_timeout=start_timeout,
        daemon_log_file=daemon_log_file,
    )


class CliModelResolutionTests(unittest.TestCase):
    def test_model_install_defaults_to_one_multilingual_family(self) -> None:
        args = build_parser().parse_args(["model-install", "--profile", "compact"])

        self.assertEqual(args.model_family, "multilingual")

    def test_model_install_accepts_explicit_english_family(self) -> None:
        args = build_parser().parse_args(
            ["model-install", "--profile", "compact", "--model-family", "english"]
        )

        self.assertEqual(args.model_family, "english")

    def test_voice_keyboard_parser_accepts_profile_and_shortcut(self) -> None:
        args = build_parser().parse_args(
            [
                "voice-keyboard",
                "--model-profile",
                "compact",
                "--shortcut",
                "CTRL+ALT+D",
                "--overlay",
                "gtk",
                "--final-commit-only",
            ]
        )

        self.assertEqual(args.command, "voice-keyboard")
        self.assertEqual(args.model_profile, "compact")
        self.assertEqual(args.shortcut, "CTRL+ALT+D")
        self.assertEqual(args.overlay, "gtk")
        self.assertTrue(args.final_commit_only)

    def test_daemon_parser_accepts_insert_partials(self) -> None:
        args = build_parser().parse_args(
            [
                "daemon",
                "--model-dir",
                "/models/parakeet",
                "--insert-partials",
                "--stream-insert-delay-seconds",
                "0.03",
            ]
        )

        self.assertEqual(args.command, "daemon")
        self.assertTrue(args.insert_partials)
        self.assertEqual(args.stream_insert_delay_seconds, 0.03)

    def test_voice_keyboard_toggle_parser_accepts_start_if_needed(self) -> None:
        args = build_parser().parse_args(
            [
                "voice-keyboard-toggle",
                "--start-if-needed",
                "--start-timeout",
                "12.5",
                "--config",
                "/tmp/wordpipe.toml",
                "--daemon-log-file",
                "/tmp/wordpipe.log",
            ]
        )

        self.assertEqual(args.command, "voice-keyboard-toggle")
        self.assertTrue(args.start_if_needed)
        self.assertEqual(args.start_timeout, 12.5)
        self.assertEqual(args.config, "/tmp/wordpipe.toml")
        self.assertEqual(args.daemon_log_file, "/tmp/wordpipe.log")

    def test_diagnostic_parsers_accept_model_profiles_without_model_dir(self) -> None:
        listen_args = build_parser().parse_args(
            ["listen-test", "--model-profile", "compact", "--duration", "1"]
        )
        stream_args = build_parser().parse_args(
            [
                "stream-file-test",
                "--model-profile",
                "fast",
                "--wav",
                "/tmp/input.wav",
            ]
        )

        self.assertEqual(listen_args.model_profile, "compact")
        self.assertIsNone(listen_args.model_dir)
        self.assertEqual(stream_args.model_profile, "fast")
        self.assertIsNone(stream_args.model_dir)

    def test_audio_devices_parser_accepts_parakeet_backend(self) -> None:
        args = build_parser().parse_args(
            ["audio-devices", "--backend", "parakeet", "--asr-worker-path", "/tmp/worker"]
        )

        self.assertEqual(args.command, "audio-devices")
        self.assertEqual(args.backend, "parakeet")
        self.assertEqual(args.asr_worker_path, "/tmp/worker")

    def test_shortcut_parsers_accept_binding_root_and_json(self) -> None:
        status_args = build_parser().parse_args(
            ["shortcut-status", "--binding", "<Super>d", "--json"]
        )
        install_args = build_parser().parse_args(
            ["shortcut-install", "--root", "/repo"]
        )

        self.assertEqual(status_args.command, "shortcut-status")
        self.assertEqual(status_args.binding, "<Super>d")
        self.assertTrue(status_args.json)
        self.assertEqual(install_args.command, "shortcut-install")
        self.assertEqual(install_args.root, "/repo")

    def test_parser_rejects_non_positive_runtime_values(self) -> None:
        cases = [
            ["voice-keyboard", "--num-threads", "0"],
            ["voice-keyboard", "--sample-rate", "0"],
            ["voice-keyboard", "--queue-seconds", "0"],
            ["voice-keyboard", "--stream-insert-delay-seconds", "-0.1"],
            ["voice-keyboard", "--stream-insert-delay-seconds", "nan"],
            ["voice-keyboard", "--endpoint-rule1-min-trailing-silence", "0"],
            ["daemon", "--endpoint-rule2-min-trailing-silence", "-1"],
            ["listen-test", "--model-dir", "/models/parakeet", "--duration", "0"],
            [
                "listen-test",
                "--model-dir",
                "/models/parakeet",
                "--endpoint-rule3-min-utterance-length",
                "nan",
            ],
            [
                "stream-file-test",
                "--model-dir",
                "/models/parakeet",
                "--wav",
                "/tmp/in.wav",
                "--chunk-seconds",
                "0",
            ],
            [
                "stream-file-test",
                "--model-dir",
                "/models/parakeet",
                "--wav",
                "/tmp/in.wav",
                "--flush-chunks",
                "-1",
            ],
            ["voice-keyboard-toggle", "--start-timeout", "0"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        build_parser().parse_args(argv)

    def test_voice_keyboard_toggle_sends_sigusr1_to_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "voice-keyboard.pid"
            pid_file.write_text("12345\n", encoding="utf-8")
            args = _toggle_args(pid_file)

            with mock.patch("os.kill") as kill:
                self.assertEqual(_cmd_voice_keyboard_toggle(args), 0)

        self.assertEqual(kill.call_args.args[0], 12345)

    def test_voice_keyboard_toggle_starts_daemon_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "voice-keyboard.pid"
            args = _toggle_args(pid_file, start_if_needed=True)

            def mark_ready(path: Path, _args: argparse.Namespace) -> None:
                path.write_text("23456\n", encoding="utf-8")

            with (
                mock.patch("wordpipe.cli._start_voice_keyboard_daemon", side_effect=mark_ready) as start,
                mock.patch("os.kill") as kill,
            ):
                self.assertEqual(_cmd_voice_keyboard_toggle(args), 0)

        start.assert_called_once()
        self.assertEqual(kill.call_args.args[0], 23456)

    def test_start_voice_keyboard_daemon_logs_child_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_file = root / "voice-keyboard.pid"
            log_file = root / "voice-keyboard.log"
            args = _toggle_args(
                pid_file,
                start_if_needed=True,
                start_timeout=0.2,
                daemon_log_file=str(log_file),
            )

            process = mock.Mock()
            process.poll.return_value = None

            def popen(_command, **kwargs):
                kwargs["stdout"].write("daemon output\n")
                kwargs["stdout"].flush()
                pid_file.write_text("23456\n", encoding="utf-8")
                return process

            with (
                mock.patch("wordpipe.cli.subprocess.Popen", side_effect=popen) as popen_mock,
                mock.patch("wordpipe.cli.os.kill"),
            ):
                _start_voice_keyboard_daemon(pid_file, args)

            self.assertIn("wordpipe voice-keyboard start", log_file.read_text(encoding="utf-8"))
            self.assertIn("daemon output", log_file.read_text(encoding="utf-8"))
            self.assertEqual(popen_mock.call_args.kwargs["stderr"], subprocess.STDOUT)

    def test_start_voice_keyboard_daemon_terminates_child_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_file = root / "voice-keyboard.pid"
            log_file = root / "voice-keyboard.log"
            args = _toggle_args(
                pid_file,
                start_if_needed=True,
                start_timeout=0,
                daemon_log_file=str(log_file),
            )
            process = mock.Mock()
            process.poll.return_value = None

            with mock.patch("wordpipe.cli.subprocess.Popen", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                    _start_voice_keyboard_daemon(pid_file, args)

        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        process.wait.assert_called_once_with(timeout=2)

    def test_start_voice_keyboard_daemon_kills_child_if_terminate_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_file = root / "voice-keyboard.pid"
            log_file = root / "voice-keyboard.log"
            args = _toggle_args(
                pid_file,
                start_if_needed=True,
                start_timeout=0,
                daemon_log_file=str(log_file),
            )
            process = mock.Mock()
            process.poll.return_value = None
            process.wait.side_effect = [subprocess.TimeoutExpired("wordpipe", 2), None]

            with mock.patch("wordpipe.cli.subprocess.Popen", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                    _start_voice_keyboard_daemon(pid_file, args)

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)

    def test_voice_keyboard_toggle_removes_stale_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "voice-keyboard.pid"
            pid_file.write_text("12345\n", encoding="utf-8")
            args = _toggle_args(pid_file)

            with mock.patch("os.kill", side_effect=ProcessLookupError):
                with self.assertRaises(RuntimeError) as raised:
                    _cmd_voice_keyboard_toggle(args)

            self.assertIn("voice keyboard is not running", str(raised.exception))
            self.assertFalse(pid_file.exists())

    def test_voice_keyboard_toggle_removes_malformed_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "voice-keyboard.pid"
            pid_file.write_text("not-a-pid\n", encoding="utf-8")
            args = _toggle_args(pid_file)

            with self.assertRaises(RuntimeError) as raised:
                _cmd_voice_keyboard_toggle(args)

            self.assertIn("voice keyboard is not running", str(raised.exception))
            self.assertFalse(pid_file.exists())

    def test_stream_file_test_drains_parakeet_stderr_while_streaming_stdout(self) -> None:
        args = argparse.Namespace(
            asr_runtime="parakeet",
            asr_worker_path="/tmp/worker",
            model_dir="/models/parakeet",
            num_threads=2,
            sample_rate=16000,
            stats_interval_seconds=1.0,
            chunk_seconds=0.56,
            flush_chunks=3,
            wav="/tmp/input.wav",
            json=False,
        )
        process = mock.Mock()
        process.stdout = io.StringIO('{"event":"commit","text":"hello","data":{}}\n')
        process.stderr = io.StringIO("worker failed\n")
        process.wait.return_value = 7

        with (
            mock.patch("wordpipe.daemon._resolve_parakeet_worker", return_value="/tmp/worker"),
            mock.patch("wordpipe.daemon.parakeet_worker_env", return_value={}),
            mock.patch("wordpipe.cli.subprocess.Popen", return_value=process),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            code = _cmd_stream_file_test(args)

        self.assertEqual(code, 7)
        self.assertIn("worker failed", stderr.getvalue())
        process.communicate.assert_not_called()

    def test_runtime_error_prints_without_traceback(self) -> None:
        parser = mock.Mock()
        parser.parse_args.return_value = argparse.Namespace(
            func=mock.Mock(side_effect=RuntimeError("setup failed"))
        )

        with mock.patch("wordpipe.cli.build_parser", return_value=parser), contextlib.redirect_stderr(
            io.StringIO()
        ) as stderr:
            code = main(["probe"])

        self.assertEqual(code, 1)
        self.assertIn("wordpipe error: setup failed", stderr.getvalue())

    def test_invalid_config_prints_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text('model_profile = "huge"\n', encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = main(["model-profiles", "--config", str(config)])

        self.assertEqual(code, 1)
        self.assertIn(f"wordpipe error: invalid config {config}", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_explicit_model_dir_wins(self) -> None:
        config = WordpipeConfig(model_dir=None, model_profile="fast")

        self.assertEqual(
            _resolve_model_dir(_args(model_dir="/models/manual"), config),
            Path("/models/manual"),
        )

    def test_config_profile_resolves_installed_runtime_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _install_marker(root, "fast")
            config = WordpipeConfig(model_dir=None, model_profile="fast", model_root=root)

            self.assertEqual(_resolve_model_dir(_args(), config), expected)

    def test_cli_profile_override_resolves_other_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _install_marker(root, "compact")
            config = WordpipeConfig(model_dir=None, model_profile="fast", model_root=root)

            self.assertEqual(
                _resolve_model_dir(_args(model_profile="compact"), config),
                expected,
            )

    def test_missing_selected_profile_points_to_install_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = WordpipeConfig(model_dir=None, model_profile="fast", model_root=Path(tmp))

            with self.assertRaises(SystemExit) as raised:
                _resolve_model_dir(_args(model_profile="compact"), config)

        self.assertIn("wordpipe model-install --profile compact", str(raised.exception))

    def test_listen_test_resolves_installed_profile_without_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _install_marker(root, "compact")
            args = argparse.Namespace(
                config=None,
                model_dir=None,
                model_profile="compact",
                model_root=str(root),
                asr_runtime="parakeet",
                asr_worker_path=None,
                provider="cpu",
                num_threads=2,
                sample_rate=16000,
                input_device=None,
                partial_interval_seconds=0.1,
                audio_chunk_seconds=0.03,
                queue_seconds=10.0,
                stats_interval_seconds=1.0,
                endpoint=False,
                endpoint_rule1_min_trailing_silence=0.55,
                endpoint_rule2_min_trailing_silence=0.35,
                endpoint_rule3_min_utterance_length=20.0,
                duration=None,
                json=False,
                full_hypotheses=False,
            )

            with mock.patch("wordpipe.listen_test.run_listen_test", return_value=0) as run:
                self.assertEqual(_cmd_listen_test(args), 0)

        self.assertEqual(run.call_args.args[0].model_dir, expected)

    def test_model_install_download_cache_follows_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_source = root / "sources" / DEFAULT_NEMO_SOURCE_FILENAME
            args = argparse.Namespace(
                config=None,
                profile="compact",
                model_root=str(root),
                source="nvidia/example",
                source_output=None,
                python="/venv/bin/python",
                force=False,
                force_source=False,
                dry_run=False,
                keep_build_dir=False,
            )

            with (
                mock.patch("wordpipe.models.download_nemo_source", return_value=expected_source) as download,
                mock.patch(
                    "wordpipe.models.build_model_profile",
                    return_value=Path("/models/runtime"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(_cmd_model_install(args), 0)

        download.assert_called_once_with(
            "nvidia/example",
            expected_source,
            force=False,
            progress=mock.ANY,
        )

    def test_model_install_uses_existing_runtime_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = _install_marker(root, "compact")
            args = argparse.Namespace(
                config=None,
                profile="compact",
                model_root=str(root),
                source=None,
                source_output=None,
                python="/venv/bin/python",
                force=False,
                force_source=False,
                dry_run=False,
                keep_build_dir=False,
            )

            with (
                mock.patch("wordpipe.models.download_prebuilt_profile") as download,
                mock.patch("wordpipe.models.install_prebuilt_profile") as install,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(_cmd_model_install(args), 0)
            self.assertTrue((runtime_dir / PROFILE_COMPLETION_MARKER).exists())

        download.assert_not_called()
        install.assert_not_called()
        self.assertEqual(stdout.getvalue().strip(), str(runtime_dir))

    def test_model_install_dry_run_does_not_download_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                config=None,
                profile="compact",
                model_root=str(root),
                source="nvidia/example",
                source_output=None,
                python="/venv/bin/python",
                force=False,
                force_source=False,
                dry_run=True,
                keep_build_dir=False,
            )

            with (
                mock.patch("wordpipe.models.download_nemo_source") as download,
                mock.patch("wordpipe.models.build_model_profile", return_value=Path("/models/runtime")) as build,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(_cmd_model_install(args), 0)

        download.assert_not_called()
        build.assert_called_once()
        self.assertTrue(build.call_args.kwargs["dry_run"])
        self.assertEqual(build.call_args.kwargs["source"], root / "sources" / DEFAULT_NEMO_SOURCE_FILENAME)

    def test_model_install_imports_built_profile_source_without_huggingface_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source-profile"
            source.mkdir()
            (source / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
            (source / "encoder.ort").write_text("encoder", encoding="utf-8")
            (source / "decoder_joint.ort").write_text("decoder", encoding="utf-8")
            args = argparse.Namespace(
                config=None,
                profile="compact",
                model_root=str(root / "models"),
                source=str(source),
                source_output=None,
                python="/venv/bin/python",
                force=False,
                force_source=False,
                dry_run=False,
                keep_build_dir=False,
            )

            with (
                mock.patch("wordpipe.models.download_nemo_source") as download,
                mock.patch("wordpipe.models.build_model_profile") as build,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(_cmd_model_install(args), 0)

            runtime_dir = profile_runtime_dir(root / "models", "compact")
            self.assertEqual(Path(stdout.getvalue().strip()), runtime_dir)
            self.assertTrue((runtime_dir / "encoder.ort").exists())

        download.assert_not_called()
        build.assert_not_called()

    def test_model_install_treats_nemo_source_as_checkpoint_not_profile_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.nemo"
            source.write_bytes(b"not a built profile archive")
            args = argparse.Namespace(
                config=None,
                profile="compact",
                model_root=str(root / "models"),
                source=str(source),
                source_output=None,
                python="/venv/bin/python",
                force=True,
                force_source=False,
                dry_run=False,
                keep_build_dir=True,
            )

            with (
                mock.patch("wordpipe.models.install_built_profile") as install,
                mock.patch("wordpipe.models.download_nemo_source", return_value=source) as download,
                mock.patch("wordpipe.models.build_model_profile", return_value=Path("/models/runtime")) as build,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(_cmd_model_install(args), 0)

        install.assert_not_called()
        download.assert_called_once()
        build.assert_called_once()
        self.assertTrue(build.call_args.kwargs["keep_build_dir"])


if __name__ == "__main__":
    unittest.main()
