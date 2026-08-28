from __future__ import annotations

import contextlib
import io
import json
import shutil
import stat
import sys
import tempfile
import tarfile
import types
import unittest
import zipfile
from unittest import mock
from pathlib import Path

from wordpipe.models import (
    DEFAULT_MODEL_REPO,
    PROFILE_COMPLETION_MARKER,
    _progress_reporter,
    build_model_profile,
    build_profile_command,
    download_prebuilt_profile,
    download_nemo_source,
    install_built_profile,
    install_prebuilt_profile,
    make_download_plan,
    model_file_url,
    model_runtime_dir_valid,
    model_family_for_language,
    profile_build_dir,
    profile_runtime_dir_valid,
    profile_runtime_dir,
    source_may_be_built_profile_archive,
    source_is_built_profile,
)


class ModelDownloadTests(unittest.TestCase):
    def test_model_file_url(self) -> None:
        self.assertEqual(
            model_file_url(DEFAULT_MODEL_REPO, "tokens.txt"),
            f"https://huggingface.co/{DEFAULT_MODEL_REPO}/resolve/main/tokens.txt",
        )

    def test_download_plan_default_directory(self) -> None:
        plan = make_download_plan(Path("models"))

        self.assertEqual(plan.repo_id, DEFAULT_MODEL_REPO)
        self.assertEqual(plan.model_dir, Path("models") / DEFAULT_MODEL_REPO.split("/")[-1])
        self.assertIn("encoder.int8.onnx", plan.files)
        self.assertNotIn("test_wavs/en.wav", plan.files)

    def test_download_plan_can_include_test_wavs(self) -> None:
        plan = make_download_plan(Path("models"), include_test_wavs=True)

        self.assertIn("test_wavs/en.wav", plan.files)

    def test_progress_reporter_is_callable(self) -> None:
        reporter = _progress_reporter(Path("model.onnx"))

        with contextlib.redirect_stderr(io.StringIO()):
            reporter(0, 8192, 100)

    def test_fast_profile_build_command(self) -> None:
        command = build_profile_command(
            source=Path("/models/source.nemo"),
            model_root=Path("/models/wordpipe"),
            profile="fast",
            python=Path("/venv/bin/python"),
            force=True,
        )

        self.assertEqual(command[0], "/venv/bin/python")
        self.assertIn("--profile", command)
        self.assertIn("fp32-projected", command)
        self.assertNotIn("--emit-ort-format", command)
        self.assertIn("--force", command)
        self.assertEqual(
            profile_runtime_dir(Path("/models/wordpipe"), "fast"),
            Path("/models/wordpipe/nemotron-wordpipe-fast-fp32-projected"),
        )

    def test_compact_profile_build_command_emits_ort_format(self) -> None:
        command = build_profile_command(
            source=Path("/models/source.nemo"),
            model_root=Path("/models/wordpipe"),
            profile="compact",
            python=Path("/venv/bin/python"),
        )

        self.assertIn("--profile", command)
        self.assertIn("compact-fixed-shape", command)
        self.assertIn("--emit-ort-format", command)
        self.assertEqual(
            profile_runtime_dir(Path("/models/wordpipe"), "compact"),
            Path("/models/wordpipe/nemotron-wordpipe-compact-fixed-shape-ort-format"),
        )

    def test_english_profile_uses_family_specific_paths_and_pipeline(self) -> None:
        command = build_profile_command(
            source=Path("/models/nemotron-speech-streaming-en-0.6b.nemo"),
            model_root=Path("/models/wordpipe"),
            profile="compact",
            family="english",
            python=Path("/venv/bin/python"),
        )

        self.assertIn("--model-family", command)
        self.assertEqual(command[command.index("--model-family") + 1], "english")
        self.assertEqual(
            profile_runtime_dir(Path("/models/wordpipe"), "compact", "english"),
            Path("/models/wordpipe/nemotron-wordpipe-en-compact-fixed-shape-ort-format"),
        )

    def test_language_selects_dedicated_english_only_for_english_locales(self) -> None:
        self.assertEqual(model_family_for_language("en-US"), "english")
        self.assertEqual(model_family_for_language("en-GB"), "english")
        self.assertEqual(model_family_for_language("auto"), "multilingual")
        self.assertEqual(model_family_for_language("fr-FR"), "multilingual")

    def test_download_prebuilt_profile_downloads_raw_hub_files(self) -> None:
        class EntryNotFoundError(Exception):
            pass

        downloaded: list[str] = []

        def hf_hub_url(_repo_id: str, filename: str) -> str:
            return f"https://huggingface.co/example/resolve/main/{filename}"

        def hf_hub_download(**kwargs):  # type: ignore[no-untyped-def]
            filename = kwargs["filename"]
            downloaded.append(filename)
            if filename.endswith(".data"):
                raise EntryNotFoundError()
            destination = Path(kwargs["local_dir"]) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if filename == "config.json":
                _write_test_profile_config(destination.parent, "fast")
            else:
                destination.write_text(filename, encoding="utf-8")
            return str(destination)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.dict(
                sys.modules,
                {
                    "huggingface_hub": types.SimpleNamespace(
                        hf_hub_download=hf_hub_download,
                        hf_hub_url=hf_hub_url,
                    ),
                    "huggingface_hub.utils": types.SimpleNamespace(EntryNotFoundError=EntryNotFoundError),
                },
            ):
                source = download_prebuilt_profile(
                    profile="fast",
                    model_root=root,
                    repo_id="danhansen/example-fast",
                )

            self.assertEqual(source, root / "downloads" / "danhansen--example-fast" / "fast")
            self.assertEqual((source / "tokenizer.model").read_text(encoding="utf-8"), "tokenizer.model")
            self.assertEqual((source / "encoder.onnx").read_text(encoding="utf-8"), "encoder.onnx")
            self.assertEqual((source / "decoder_joint.onnx").read_text(encoding="utf-8"), "decoder_joint.onnx")
            self.assertTrue((source / PROFILE_COMPLETION_MARKER).exists())
            self.assertIn("encoder.onnx.data", downloaded)

    def test_download_prebuilt_profile_reports_structured_file_progress(self) -> None:
        class EntryNotFoundError(Exception):
            pass

        class HfHubHTTPError(Exception):
            pass

        sizes = {
            "tokenizer.model": 100,
            "encoder.onnx": 200,
            "encoder.onnx.data": 1000,
            "decoder_joint.onnx": 300,
            "config.json": 50,
            "preprocessor_config.json": 60,
            "tokenizer_config.json": 70,
        }
        progress: list[str] = []
        xet_values: list[str | None] = []

        def hf_hub_url(_repo_id: str, filename: str) -> str:
            return filename

        def get_hf_file_metadata(url: str) -> types.SimpleNamespace:
            if url not in sizes:
                raise EntryNotFoundError()
            return types.SimpleNamespace(size=sizes[url])

        def hf_hub_download(**kwargs):  # type: ignore[no-untyped-def]
            import os

            xet_values.append(os.environ.get("HF_HUB_DISABLE_XET"))
            filename = kwargs["filename"]
            destination = Path(kwargs["local_dir"]) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if filename == "config.json":
                _write_test_profile_config(destination.parent, "fast")
            else:
                destination.write_bytes(b"x" * sizes[filename])
            return str(destination)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.dict(
                sys.modules,
                {
                    "huggingface_hub": types.SimpleNamespace(
                        get_hf_file_metadata=get_hf_file_metadata,
                        hf_hub_download=hf_hub_download,
                        hf_hub_url=hf_hub_url,
                    ),
                    "huggingface_hub.utils": types.SimpleNamespace(
                        EntryNotFoundError=EntryNotFoundError,
                        HfHubHTTPError=HfHubHTTPError,
                    ),
                },
            ):
                with mock.patch.dict("os.environ", {}, clear=True):
                    download_prebuilt_profile(
                        profile="fast",
                        model_root=root,
                        repo_id="danhansen/example-fast",
                        progress=progress.append,
                    )

        events = [
            json.loads(line.removeprefix("wordpipe-progress "))
            for line in progress
            if line.startswith("wordpipe-progress ")
        ]
        encoder_data = [
            event
            for event in events
            if event.get("phase") == "downloading"
            and event.get("filename") == "encoder.onnx.data"
        ]
        self.assertEqual(len(encoder_data), 1)
        self.assertEqual(encoder_data[0]["file_size"], 1000)
        self.assertEqual(encoder_data[0]["total_bytes"], sum(sizes.values()))
        self.assertIn("encoder.onnx.data", encoder_data[0]["message"])
        self.assertTrue(xet_values)
        self.assertTrue(all(value == "1" for value in xet_values))

    def test_download_prebuilt_profile_uses_ranged_downloader_for_large_files(self) -> None:
        class EntryNotFoundError(Exception):
            pass

        class HfHubHTTPError(Exception):
            pass

        sizes = {
            "tokenizer.model": 100,
            "encoder.onnx": 70 * 1024 * 1024,
            "decoder_joint.onnx": 300,
        }
        hub_downloaded: list[str] = []

        def hf_hub_url(_repo_id: str, filename: str) -> str:
            return f"https://huggingface.co/example/resolve/main/{filename}"

        def get_hf_file_metadata(url: str) -> types.SimpleNamespace:
            filename = url.rsplit("/", 1)[-1]
            if filename not in sizes:
                raise EntryNotFoundError()
            return types.SimpleNamespace(size=sizes[filename])

        def hf_hub_download(**kwargs):  # type: ignore[no-untyped-def]
            filename = kwargs["filename"]
            hub_downloaded.append(filename)
            destination = Path(kwargs["local_dir"]) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"x" * sizes[filename])
            return str(destination)

        def run_ranged(command, _progress, *, env=None):  # type: ignore[no-untyped-def]
            destination = Path(command[3])
            destination.write_bytes(b"encoder")
            self.assertEqual(env["WORDPIPE_PROGRESS_FILENAME"], "encoder.onnx")
            self.assertEqual(env["WORDPIPE_PROGRESS_FILE_INDEX"], "2")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "huggingface_hub": types.SimpleNamespace(
                            get_hf_file_metadata=get_hf_file_metadata,
                            hf_hub_download=hf_hub_download,
                            hf_hub_url=hf_hub_url,
                        ),
                        "huggingface_hub.utils": types.SimpleNamespace(
                            EntryNotFoundError=EntryNotFoundError,
                            HfHubHTTPError=HfHubHTTPError,
                        ),
                    },
                ),
                mock.patch("wordpipe.models._run_with_progress", side_effect=run_ranged) as run,
            ):
                download_prebuilt_profile(
                    profile="compact",
                    model_root=root,
                    repo_id="danhansen/example-compact",
                )

        self.assertEqual(hub_downloaded, ["tokenizer.model", "decoder_joint.onnx"])
        run.assert_called_once()

    def test_external_onnx_data_must_exist_for_profile_to_be_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
            (source / "encoder.onnx").write_bytes(b"external encoder.onnx.data")
            (source / "decoder_joint.onnx").write_bytes(b"decoder")

            self.assertFalse(source_is_built_profile(source))

            (source / "encoder.onnx.data").write_bytes(b"weights")

            self.assertTrue(source_is_built_profile(source))

    def test_completion_marker_must_match_file_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            _write_test_runtime_profile(source)
            marker = source / PROFILE_COMPLETION_MARKER
            marker.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "profile": "compact",
                        "files": [
                            {"path": "tokenizer.model", "size": 9, "sha256": "unused"},
                            {"path": "encoder.ort", "size": 7, "sha256": "unused"},
                            {"path": "decoder_joint.ort", "size": 999, "sha256": "unused"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(model_runtime_dir_valid(source))

    def test_profile_runtime_dir_rejects_non_fixed_shape_fast_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            _write_test_runtime_profile(source)
            (source / "config.json").write_text(
                json.dumps({"projected_cache": True, "dynamic_quint8_quantization": False}),
                encoding="utf-8",
            )

            self.assertTrue(model_runtime_dir_valid(source))
            self.assertFalse(profile_runtime_dir_valid(source, "fast"))

            _write_test_profile_config(source, "fast")

            self.assertTrue(profile_runtime_dir_valid(source, "fast"))

    def test_completion_marker_rejects_bad_profile_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            _write_test_runtime_profile(source)
            (source / "config.json").write_text(
                json.dumps({"projected_cache": True, "dynamic_quint8_quantization": False}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "fixed-shape projected-cache"):
                install_built_profile(source=source, model_root=root / "installed", profile="fast")

    def test_install_built_profile_copies_runtime_dir_to_profile_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            _write_test_runtime_profile(source)
            _write_test_profile_config(source, "compact")

            runtime_dir = install_built_profile(
                source=source,
                model_root=root / "installed",
                profile="compact",
            )

            self.assertEqual(runtime_dir, profile_runtime_dir(root / "installed", "compact"))
            self.assertEqual((runtime_dir / "tokenizer.model").read_text(encoding="utf-8"), "tokenizer")
            self.assertTrue((runtime_dir / "encoder.ort").exists())
            self.assertTrue((runtime_dir / "decoder_joint.ort").exists())
            marker = json.loads((runtime_dir / PROFILE_COMPLETION_MARKER).read_text(encoding="utf-8"))
            self.assertEqual(marker["format"], 1)
            self.assertEqual(marker["profile"], "compact")
            self.assertTrue(all("sha256" in item for item in marker["files"]))

    def test_install_prebuilt_profile_uses_existing_runtime_before_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = profile_runtime_dir(root, "compact")
            _write_test_runtime_profile(runtime_dir)
            missing_source = root / "missing-source"

            result = install_prebuilt_profile(
                source=missing_source,
                model_root=root,
                profile="compact",
            )

            self.assertEqual(result, runtime_dir)
            self.assertTrue((runtime_dir / PROFILE_COMPLETION_MARKER).exists())

    def test_install_prebuilt_profile_reuses_cached_onnx_profile_for_ort_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx_dir = root / "nemotron-wordpipe-compact-fixed-shape"
            onnx_dir.mkdir()
            (onnx_dir / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
            (onnx_dir / "encoder.onnx").write_text("encoder", encoding="utf-8")
            (onnx_dir / "decoder_joint.onnx").write_text("decoder", encoding="utf-8")
            source = root / "missing-source"
            runtime_dir = profile_runtime_dir(root, "compact")

            def convert(command, _progress):  # type: ignore[no-untyped-def]
                self.assertEqual(Path(command[2]), onnx_dir)
                self.assertEqual(Path(command[3]), runtime_dir)
                _write_test_runtime_profile(runtime_dir)

            with mock.patch("wordpipe.models._run_with_progress", side_effect=convert) as run:
                result = install_prebuilt_profile(
                    source=source,
                    model_root=root,
                    profile="compact",
                    progress=lambda _message: None,
                )

            self.assertEqual(result, runtime_dir)
            self.assertEqual(run.call_count, 1)
            self.assertTrue((runtime_dir / PROFILE_COMPLETION_MARKER).exists())

    def test_install_built_profile_preserves_existing_profile_when_force_copy_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "tokenizer.model").write_text("new-tokenizer", encoding="utf-8")
            (source / "encoder.ort").write_text("new-encoder", encoding="utf-8")
            (source / "decoder_joint.ort").write_text("new-decoder", encoding="utf-8")
            destination = profile_runtime_dir(root / "installed", "compact")
            destination.mkdir(parents=True)
            (destination / "tokenizer.model").write_text("old-tokenizer", encoding="utf-8")
            (destination / "encoder.ort").write_text("old-encoder", encoding="utf-8")
            (destination / "decoder_joint.ort").write_text("old-decoder", encoding="utf-8")

            original_copytree = shutil.copytree

            def failing_copytree(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
                original_copytree(src, dst, *args, **kwargs)
                raise RuntimeError("copy failed")

            with (
                mock.patch("wordpipe.models.shutil.copytree", side_effect=failing_copytree),
                self.assertRaisesRegex(RuntimeError, "copy failed"),
            ):
                install_built_profile(
                    source=source,
                    model_root=root / "installed",
                    profile="compact",
                    force=True,
                )

            self.assertEqual(
                (destination / "tokenizer.model").read_text(encoding="utf-8"),
                "old-tokenizer",
            )
            self.assertFalse(any(destination.parent.glob(f".{destination.name}.tmp-*")))

    def test_install_built_profile_imports_zip_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("profile/tokenizer.model", "tokenizer")
                archive.writestr("profile/encoder.ort", "encoder")
                archive.writestr("profile/decoder_joint.ort", "decoder")

            runtime_dir = install_built_profile(
                source=archive_path,
                model_root=root / "installed",
                profile="compact",
            )

            self.assertEqual(runtime_dir, profile_runtime_dir(root / "installed", "compact"))
            self.assertEqual((runtime_dir / "tokenizer.model").read_text(encoding="utf-8"), "tokenizer")

    def test_install_built_profile_removes_zip_extract_tempdir_after_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extract_dir = root / "extract"
            archive_path = root / "profile.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("profile/tokenizer.model", "tokenizer")
                archive.writestr("profile/encoder.ort", "encoder")
                archive.writestr("profile/decoder_joint.ort", "decoder")

            def make_tempdir(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                extract_dir.mkdir()
                return str(extract_dir)

            with mock.patch("tempfile.mkdtemp", side_effect=make_tempdir):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

            self.assertFalse(extract_dir.exists())

    def test_install_built_profile_rejects_unsafe_zip_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.zip"
            extract_dir = root / "extract"
            outside = root / "evil.txt"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../evil.txt", "bad")
                archive.writestr("profile/tokenizer.model", "tokenizer")
                archive.writestr("profile/encoder.ort", "encoder")
                archive.writestr("profile/decoder_joint.ort", "decoder")

            def make_tempdir(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                extract_dir.mkdir()
                return str(extract_dir)

            with (
                mock.patch("tempfile.mkdtemp", side_effect=make_tempdir),
                self.assertRaisesRegex(RuntimeError, "unsafe zip member"),
            ):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

            self.assertFalse(outside.exists())
            self.assertFalse(extract_dir.exists())

    def test_install_built_profile_rejects_zip_symlink_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.zip"
            extract_dir = root / "extract"
            with zipfile.ZipFile(archive_path, "w") as archive:
                link = zipfile.ZipInfo("profile/encoder.ort")
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(link, "/tmp/encoder.ort")
                archive.writestr("profile/tokenizer.model", "tokenizer")
                archive.writestr("profile/decoder_joint.ort", "decoder")

            def make_tempdir(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                extract_dir.mkdir()
                return str(extract_dir)

            with (
                mock.patch("tempfile.mkdtemp", side_effect=make_tempdir),
                self.assertRaisesRegex(RuntimeError, "unsupported zip member"),
            ):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

            self.assertFalse(extract_dir.exists())

    def test_install_built_profile_imports_tar_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                _add_tar_text(archive, "profile/tokenizer.model", "tokenizer")
                _add_tar_text(archive, "profile/encoder.ort", "encoder")
                _add_tar_text(archive, "profile/decoder_joint.ort", "decoder")

            runtime_dir = install_built_profile(
                source=archive_path,
                model_root=root / "installed",
                profile="compact",
            )

            self.assertEqual(runtime_dir, profile_runtime_dir(root / "installed", "compact"))
            self.assertEqual((runtime_dir / "tokenizer.model").read_text(encoding="utf-8"), "tokenizer")

    def test_install_built_profile_rejects_unsafe_tar_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.tar"
            outside = root / "evil.txt"
            with tarfile.open(archive_path, "w") as archive:
                _add_tar_text(archive, "../evil.txt", "bad")
                _add_tar_text(archive, "profile/tokenizer.model", "tokenizer")
                _add_tar_text(archive, "profile/encoder.ort", "encoder")
                _add_tar_text(archive, "profile/decoder_joint.ort", "decoder")

            with self.assertRaisesRegex(RuntimeError, "unsafe tar member"):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

            self.assertFalse(outside.exists())

    def test_install_built_profile_rejects_tar_symlink_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.tar"
            with tarfile.open(archive_path, "w") as archive:
                link = tarfile.TarInfo("profile/encoder.ort")
                link.type = tarfile.SYMTYPE
                link.linkname = "/tmp/encoder.ort"
                archive.addfile(link)
                _add_tar_text(archive, "profile/tokenizer.model", "tokenizer")
                _add_tar_text(archive, "profile/decoder_joint.ort", "decoder")

            with self.assertRaisesRegex(RuntimeError, "unsupported tar member"):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

    def test_nemo_source_is_not_built_profile_archive(self) -> None:
        self.assertFalse(source_may_be_built_profile_archive(Path("source.nemo")))
        self.assertTrue(source_may_be_built_profile_archive(Path("profile.tar.gz")))

    def test_build_model_profile_removes_build_dir_after_success_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = profile_build_dir(root, "compact")
            build_dir.mkdir(parents=True)
            (build_dir / "intermediate.onnx").write_text("temporary", encoding="utf-8")

            def run_build(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                _write_test_runtime_profile(profile_runtime_dir(root, "compact"))

            with mock.patch("subprocess.run", side_effect=run_build) as run:
                runtime_dir = build_model_profile(
                    source=root / "source.nemo",
                    model_root=root,
                    profile="compact",
                    python=Path("/usr/bin/python3"),
                )

            run.assert_called_once()
            self.assertFalse(build_dir.exists())
            self.assertEqual(runtime_dir, profile_runtime_dir(root, "compact"))

    def test_build_model_profile_can_keep_build_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = profile_build_dir(root, "compact")
            build_dir.mkdir(parents=True)

            def run_build(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                _write_test_runtime_profile(profile_runtime_dir(root, "compact"))

            with mock.patch("subprocess.run", side_effect=run_build):
                build_model_profile(
                    source=root / "source.nemo",
                    model_root=root,
                    profile="compact",
                    python=Path("/usr/bin/python3"),
                    keep_build_dir=True,
                )

            self.assertTrue(build_dir.exists())

    def test_download_nemo_source_reports_cached_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.nemo"
            source.write_text("cached", encoding="utf-8")
            events: list[str] = []

            result = download_nemo_source(str(source), progress=events.append)

        self.assertEqual(result, source)
        self.assertEqual(events, [f"Using local source model: {source}"])

    def test_build_model_profile_reports_command_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[str] = []

            def run_build(_command, _progress):  # type: ignore[no-untyped-def]
                _write_test_runtime_profile(profile_runtime_dir(root, "compact"))

            with mock.patch("wordpipe.models._run_with_progress", side_effect=run_build) as run:
                runtime_dir = build_model_profile(
                    source=root / "source.nemo",
                    model_root=root,
                    profile="compact",
                    python=Path("/usr/bin/python3"),
                    progress=events.append,
                )

        run.assert_called_once()
        self.assertEqual(run.call_args.args[1], events.append)
        self.assertEqual(runtime_dir, profile_runtime_dir(root, "compact"))
        self.assertTrue(any("build_nemotron_wordpipe_model.py" in event for event in events))
        self.assertEqual(events[-1], f"Model profile ready: {runtime_dir}")


def _add_tar_text(archive: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _write_test_runtime_profile(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
    (path / "encoder.ort").write_text("encoder", encoding="utf-8")
    (path / "decoder_joint.ort").write_text("decoder", encoding="utf-8")


def _write_test_profile_config(path: Path, profile: str) -> None:
    (path / "config.json").write_text(
        json.dumps(
            {
                "projected_cache": True,
                "dynamic_quint8_quantization": profile == "compact",
                "fixed_streaming_shapes": {
                    "input_frames": 65,
                    "output_frames": 7,
                    "num_layers": 24,
                    "cache_len": 56,
                    "hidden_dim": 1024,
                    "conv_context": 8,
                },
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
