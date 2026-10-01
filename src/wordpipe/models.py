from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from typing import Callable, Literal
from urllib.request import urlretrieve
import zipfile


DEFAULT_MODEL_REPO = (
    "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
)
DEFAULT_MODEL_FILES = (
    "tokens.txt",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "README.md",
)
DEFAULT_NEMO_SOURCE_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
DEFAULT_NEMO_SOURCE_FILENAME = "nemotron-3.5-asr-streaming-0.6b.nemo"
DEFAULT_ENGLISH_NEMO_SOURCE_REPO = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_ENGLISH_NEMO_SOURCE_FILENAME = "nemotron-speech-streaming-en-0.6b.nemo"
DEFAULT_PREBUILT_PROFILE_REPO = "fractalyzer/wordpipe-nemotron-fast-fp32-projected"
PREBUILT_PROFILE_FILES = (
    "tokenizer.model",
    "encoder.onnx",
    "encoder.onnx.data",
    "encoder.shared-weights.data",
    "decoder_joint.onnx",
    "decoder_joint.onnx.data",
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "wordpipe-streaming-bundle.json",
)
REQUIRED_PREBUILT_PROFILE_FILES = ("tokenizer.model", "encoder.onnx", "decoder_joint.onnx")
PROFILE_COMPLETION_MARKER = ".wordpipe-profile.json"
RANGED_DOWNLOAD_MIN_SIZE = 64 * 1024 * 1024
ModelProfile = Literal["fast", "compact"]
ModelFamily = Literal["multilingual", "english"]
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class DownloadPlan:
    repo_id: str
    output_dir: Path
    files: tuple[str, ...]

    @property
    def model_dir(self) -> Path:
        return self.output_dir / self.repo_id.split("/")[-1]


@dataclass(frozen=True)
class ModelProfileSpec:
    name: ModelProfile
    title: str
    description: str
    build_profile: str
    output_name: str
    prebuilt_repo: str
    english_output_name: str
    english_prebuilt_repo: str
    emit_ort_format: bool = False

    def output_name_for_family(self, family: ModelFamily) -> str:
        return self.english_output_name if family == "english" else self.output_name

    def prebuilt_repo_for_family(self, family: ModelFamily) -> str:
        return self.english_prebuilt_repo if family == "english" else self.prebuilt_repo

    def output_dir(self, model_root: Path, family: ModelFamily = "multilingual") -> Path:
        return model_root.expanduser() / self.output_name_for_family(family)

    def runtime_dir(self, model_root: Path, family: ModelFamily = "multilingual") -> Path:
        output = self.output_dir(model_root, family)
        if _dynamic_config_at(output, family) is not None:
            return output
        if self.emit_ort_format:
            return output.with_name(f"{output.name}-ort-format")
        return output


@dataclass(frozen=True)
class _PreparedProfileSource:
    path: Path
    cleanup_dir: Path | None = None


@dataclass(frozen=True)
class _PrebuiltFile:
    filename: str
    required: bool
    size: int | None = None


MODEL_PROFILES: dict[ModelProfile, ModelProfileSpec] = {
    "fast": ModelProfileSpec(
        name="fast",
        title="Fast",
        description="FP32 projected-cache model; larger memory and download footprint.",
        build_profile="fp32-projected",
        output_name="nemotron-wordpipe-fast-fp32-projected",
        prebuilt_repo="fractalyzer/wordpipe-nemotron-fast-fp32-projected",
        english_output_name="nemotron-wordpipe-en-fast-fp32-projected",
        english_prebuilt_repo="fractalyzer/wordpipe-nemotron-en-fast-dynamic-fp32-projected",
    ),
    "compact": ModelProfileSpec(
        name="compact",
        title="Compact",
        description="Quantized projected-cache model; generic ONNX or legacy ORT runtime.",
        build_profile="compact-fixed-shape",
        output_name="nemotron-wordpipe-compact-fixed-shape",
        prebuilt_repo="fractalyzer/wordpipe-nemotron-compact-fixed-shape",
        english_output_name="nemotron-wordpipe-en-compact-fixed-shape",
        english_prebuilt_repo="fractalyzer/wordpipe-nemotron-en-compact-dynamic-quint8",
        emit_ort_format=True,
    ),
}


def make_download_plan(
    output_dir: Path,
    repo_id: str = DEFAULT_MODEL_REPO,
    include_test_wavs: bool = False,
) -> DownloadPlan:
    files = list(DEFAULT_MODEL_FILES)
    if include_test_wavs:
        files.extend(["test_wavs/en.wav", "test_wavs/ja.wav"])
    return DownloadPlan(repo_id=repo_id, output_dir=output_dir.expanduser(), files=tuple(files))


def download_model(plan: DownloadPlan, force: bool = False) -> Path:
    model_dir = plan.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in plan.files:
        destination = model_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            print(f"skip {destination}", file=sys.stderr)
            continue
        url = model_file_url(plan.repo_id, filename)
        print(f"download {url}", file=sys.stderr)
        temporary = destination.with_name(destination.name + ".part")
        urlretrieve(url, temporary, _progress_reporter(destination))
        temporary.replace(destination)
    return model_dir


def model_file_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _progress_reporter(destination: Path):
    last_report = 0.0

    def report(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < 5 and block_count != 0:
            return
        last_report = now
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            print(
                f"  {destination.name}: {downloaded}/{total_size} bytes ({percent:.1f}%)",
                file=sys.stderr,
            )
        else:
            print(f"  {destination.name}: {downloaded} bytes", file=sys.stderr)

    return report


def default_model_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "wordpipe" / "models"
    return Path.home() / ".local" / "share" / "wordpipe" / "models"


def default_nemo_source_path(
    model_root: Path | None = None,
    family: ModelFamily = "multilingual",
) -> Path:
    root = model_root.expanduser() if model_root is not None else default_model_root()
    filename = (
        DEFAULT_ENGLISH_NEMO_SOURCE_FILENAME
        if family == "english"
        else DEFAULT_NEMO_SOURCE_FILENAME
    )
    return root / "sources" / filename


def model_family_for_language(language: str) -> ModelFamily:
    normalized = language.strip().lower()
    return "english" if normalized == "en" or normalized.startswith("en-") else "multilingual"


def profile_spec(name: str) -> ModelProfileSpec:
    if name not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile: {name}")
    return MODEL_PROFILES[name]  # type: ignore[index]


def streaming_model_root(model_root: Path, latency_ms: int = 560) -> Path:
    if latency_ms <= 0 or latency_ms % 80:
        raise ValueError("streaming latency must be a positive multiple of 80 ms")
    return model_root if latency_ms == 560 else model_root / f"{latency_ms}ms"


def profile_streaming_latency(runtime_dir: Path) -> int:
    config_path = runtime_dir / "config.json"
    if not config_path.exists():
        return 560
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("dynamic_streaming") is not None:
        dynamic = dynamic_streaming_config(config, config.get("model_family", "multilingual"))
        if dynamic is None:
            raise ValueError("invalid dynamic streaming contract")
        return dynamic["default_chunk_frames"] * 80
    right = config.get("right_context", 6)
    if right not in (6, 13):
        raise ValueError(f"unsupported right_context: {right!r}")
    return (right + 1) * 80


def dynamic_streaming_config(config: dict, family: ModelFamily) -> dict | None:
    dynamic = config.get("dynamic_streaming")
    cache_len = 70 if family == "english" else 56
    expected = {
        "format": 1,
        "subsampling_factor": 8, "mel_frames_overhead": 9,
        "cache_len": cache_len, "shape_derived_attention_context": True,
    }
    if not isinstance(dynamic, dict) or any(dynamic.get(key) != value for key, value in expected.items()):
        return None
    sizes = dynamic.get("supported_chunk_frames")
    if not isinstance(sizes, list) or not sizes or any(type(size) is not int or size <= 0 or cache_len % size for size in sizes):
        return None
    if sizes != sorted(set(sizes)) or dynamic.get("default_chunk_frames") not in sizes:
        return None
    if "right_context" in config and (
        type(config["right_context"]) is not int
        or config["right_context"] + 1 != dynamic["default_chunk_frames"]
    ):
        return None
    if any(config.get(key) != value for key, value in (
        ("num_encoder_layers", 24), ("hidden_dim", 1024), ("conv_context", 8),
    )):
        return None
    shapes = config.get("cache_shapes", {})
    if not isinstance(shapes, dict) or shapes.get("cache_last_channel") != [24, 1, cache_len, 1024] or shapes.get("cache_last_time") != [24, 1, 1024, 8]:
        return None
    return dynamic


def _dynamic_config_at(runtime_dir: Path, family: ModelFamily) -> dict | None:
    try:
        config = json.loads((runtime_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    return dynamic_streaming_config(config, family)


def profile_is_dynamic(runtime_dir: Path, family: ModelFamily = "multilingual") -> bool:
    return _dynamic_config_at(runtime_dir, family) is not None


def profile_supports_streaming_latency(runtime_dir: Path, latency_ms: int) -> bool:
    if latency_ms <= 0 or latency_ms % 80:
        return False
    try:
        config = json.loads((runtime_dir / "config.json").read_text())
    except FileNotFoundError:
        return latency_ms == 560
    except (OSError, ValueError):
        return False
    dynamic = dynamic_streaming_config(config, config.get("model_family", "multilingual"))
    if dynamic is not None:
        return latency_ms // 80 in dynamic["supported_chunk_frames"]
    if config.get("dynamic_streaming") is not None:
        return False
    try:
        return profile_streaming_latency(runtime_dir) == latency_ms
    except ValueError:
        return False


def profile_runtime_dir(
    model_root: Path,
    profile: str,
    family: ModelFamily = "multilingual",
) -> Path:
    return profile_spec(profile).runtime_dir(model_root, family)


def profile_installed(
    model_root: Path,
    profile: str,
    family: ModelFamily = "multilingual",
) -> bool:
    runtime_dir = profile_runtime_dir(model_root, profile, family)
    return profile_runtime_dir_valid(runtime_dir, profile, family)


def ensure_profile_completion_marker(
    model_root: Path,
    profile: str,
    family: ModelFamily = "multilingual",
) -> Path:
    runtime_dir = profile_runtime_dir(model_root, profile, family)
    if not profile_runtime_dir_valid(runtime_dir, profile, family):
        raise RuntimeError(f"model profile {profile!r} is not installed at {runtime_dir}")
    if not _profile_completion_marker(runtime_dir).exists():
        _write_profile_completion_marker(runtime_dir, profile=profile, family=family)
    return runtime_dir


def model_runtime_dir_valid(runtime_dir: Path) -> bool:
    if not _runtime_structure_valid(runtime_dir):
        return False
    marker = _profile_completion_marker(runtime_dir)
    return not marker.exists() or _profile_completion_marker_valid(runtime_dir, verify_hashes=False)


def profile_runtime_dir_valid(
    runtime_dir: Path,
    profile: str,
    family: ModelFamily = "multilingual",
) -> bool:
    return model_runtime_dir_valid(runtime_dir) and _profile_config_valid_if_present(
        runtime_dir, profile, family
    )


def _runtime_structure_valid(runtime_dir: Path) -> bool:
    return (
        (runtime_dir / "tokenizer.model").exists()
        and _graph_file_valid(runtime_dir, "encoder")
        and _graph_file_valid(runtime_dir, "decoder_joint")
    )


def _graph_file_valid(runtime_dir: Path, stem: str) -> bool:
    if (runtime_dir / f"{stem}.ort").exists():
        return True
    onnx_path = runtime_dir / f"{stem}.onnx"
    if not onnx_path.exists():
        return False
    data_path = runtime_dir / f"{stem}.onnx.data"
    if data_path.exists():
        return True
    try:
        if onnx_path.stat().st_size > 128 * 1024 * 1024:
            return True
    except OSError:
        return False
    return not _onnx_references_external_data(onnx_path, data_path.name)


def _onnx_references_external_data(onnx_path: Path, marker: str) -> bool:
    marker_bytes = marker.encode("utf-8")
    try:
        with onnx_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if marker_bytes in chunk:
                    return True
    except OSError:
        return True
    return False


def _profile_config_valid_if_present(
    runtime_dir: Path,
    profile: str,
    family: ModelFamily = "multilingual",
) -> bool:
    config_path = runtime_dir / "config.json"
    if not config_path.exists():
        return True
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    fixed = config.get("fixed_streaming_shapes")
    shared_files = config.get("shared_weight_files", [])
    if not isinstance(shared_files, list) or any(
        not isinstance(name, str) or not _safe_relative_path(name)
        or not (runtime_dir / name).is_file() for name in shared_files
    ):
        return False
    generic = config.get("dynamic_streaming") is not None
    if generic:
        if dynamic_streaming_config(config, family) is None:
            return False
    elif not isinstance(fixed, dict):
        return False
    right_context = config.get("right_context", 6)
    if not generic and right_context not in (6, 13):
        return False
    if not generic and "1120ms" in runtime_dir.parts and right_context != 13:
        return False
    output_frames = right_context + 1
    expected_fixed = {
        "input_frames": output_frames * 8 + 9,
        "output_frames": output_frames,
        "num_layers": 24,
        "cache_len": 70 if family == "english" else 56,
        "hidden_dim": 1024,
        "conv_context": 8,
    }
    if not generic and any(fixed.get(key) != value for key, value in expected_fixed.items()):
        return False
    if config.get("projected_cache") is not True:
        return False
    configured_family = config.get("model_family")
    if configured_family is not None and configured_family != family:
        return False
    if family == "english" and configured_family != "english":
        return False
    quantized = bool(config.get("dynamic_quint8_quantization"))
    if profile == "fast":
        return not quantized
    if profile == "compact":
        return quantized
    return False


def source_is_built_profile(source: Path) -> bool:
    return source.is_dir() and model_runtime_dir_valid(source)


def source_may_be_built_profile_archive(source: Path) -> bool:
    name = source.name.lower()
    if name.endswith(".nemo"):
        return False
    return name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))


def prebuilt_profile_cache_dir(model_root: Path, repo_id: str, profile: str) -> Path:
    return model_root.expanduser() / "downloads" / repo_id.replace("/", "--") / profile


def install_built_profile(
    *,
    source: Path,
    model_root: Path,
    profile: str,
    family: ModelFamily = "multilingual",
    force: bool = False,
    streaming_latency_ms: int | None = None,
) -> Path:
    prepared_source = _prepare_built_profile_source(source)
    if streaming_latency_ms is not None and not profile_supports_streaming_latency(prepared_source.path, streaming_latency_ms):
        if prepared_source.cleanup_dir is not None:
            shutil.rmtree(prepared_source.cleanup_dir, ignore_errors=True)
        raise RuntimeError("source model streaming latency does not match the requested mode")
    destination = (profile_spec(profile).output_dir(model_root, family)
                   if _dynamic_config_at(prepared_source.path, family) is not None
                   else profile_runtime_dir(model_root, profile, family))
    if destination.exists():
        if not force:
            raise RuntimeError(f"profile {profile!r} is already installed at {destination}; pass --force to overwrite it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(prepared_source.path, temporary, copy_function=_link_or_copy)
        _write_profile_completion_marker(temporary, profile=profile, family=family)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        if prepared_source.cleanup_dir is not None:
            shutil.rmtree(prepared_source.cleanup_dir, ignore_errors=True)
    return destination


def download_prebuilt_profile(
    *,
    profile: str,
    model_root: Path,
    family: ModelFamily = "multilingual",
    repo_id: str | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    spec = profile_spec(profile)
    selected_repo = repo_id or spec.prebuilt_repo_for_family(family)
    output_dir = prebuilt_profile_cache_dir(model_root, selected_repo, profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_is_built_profile(output_dir) and _profile_config_valid_if_present(output_dir, profile, family) and not force:
        if not _profile_completion_marker(output_dir).exists():
            _write_profile_completion_marker(output_dir, profile=profile, family=family)
        _progress(progress, f"Using cached prebuilt profile: {output_dir}")
        return output_dir
    if _profile_completion_marker(output_dir).exists() and not _profile_completion_marker_valid(output_dir, verify_hashes=False):
        _progress(progress, f"Removing invalid cached prebuilt profile: {output_dir}")
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    xet_env_value = os.environ.get("HF_HUB_DISABLE_XET")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
        from huggingface_hub.utils import EntryNotFoundError

        _progress(progress, f"Downloading {spec.title} profile from {selected_repo}")
        files = _prebuilt_download_plan(selected_repo)
        total_size = sum(item.size or 0 for item in files)
        completed_size = 0
        env_value = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        try:
            import hf_transfer  # noqa: F401
        except ImportError:
            enable_hf_transfer = False
        else:
            enable_hf_transfer = True
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        try:
            for index, item in enumerate(files):
                _progress_event(
                    progress,
                    profile=spec.name,
                    phase="downloading",
                    message=_download_message(item, index, len(files)),
                    filename=item.filename,
                    file_index=index + 1,
                    file_count=len(files),
                    file_size=item.size or 0,
                    completed_bytes=completed_size,
                    total_bytes=total_size,
                    fraction=_download_fraction(completed_size, total_size, index, len(files)),
                )
                try:
                    path = _download_prebuilt_file(
                        repo_id=selected_repo,
                        item=item,
                        output_dir=output_dir,
                        force=force,
                        hf_hub_download=hf_hub_download,
                        url=hf_hub_url(selected_repo, item.filename),
                        progress=progress,
                        profile=spec.name,
                        file_index=index + 1,
                        file_count=len(files),
                        completed_base=completed_size,
                        total_bytes=total_size,
                    )
                except EntryNotFoundError:
                    if item.required:
                        raise
                    continue
                actual_size = _downloaded_size(Path(path), output_dir / item.filename)
                completed_size += item.size or actual_size
                _progress_event(
                    progress,
                    profile=spec.name,
                    phase="downloaded",
                    message=f"Downloaded {item.filename}",
                    filename=item.filename,
                    file_index=index + 1,
                    file_count=len(files),
                    file_size=item.size or actual_size,
                    completed_bytes=completed_size,
                    total_bytes=total_size,
                    fraction=_download_fraction(completed_size, total_size, index + 1, len(files)),
                )
        finally:
            if enable_hf_transfer:
                if env_value is None:
                    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
                else:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = env_value
        if not _profile_config_valid_if_present(output_dir, profile, family):
            raise RuntimeError(
                f"downloaded {profile!r} profile at {output_dir} does not match Wordpipe's "
                "fixed-shape projected-cache runtime contract"
            )
        _write_profile_completion_marker(output_dir, profile=profile, family=family)
        _progress(progress, f"Prebuilt profile ready: {output_dir}")
        return output_dir
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download prebuilt Wordpipe model profiles. "
            "Install it with hf_transfer support, or pass --source with a local profile directory/archive."
        ) from exc
    finally:
        if xet_env_value is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = xet_env_value


def _download_prebuilt_file(
    *,
    repo_id: str,
    item: _PrebuiltFile,
    output_dir: Path,
    force: bool,
    hf_hub_download: object,
    url: str,
    progress: ProgressCallback | None,
    profile: str,
    file_index: int,
    file_count: int,
    completed_base: int,
    total_bytes: int,
) -> Path:
    destination = output_dir / item.filename
    if (
        item.size is not None
        and item.size >= RANGED_DOWNLOAD_MIN_SIZE
        and _download_hf_ranged_script().exists()
    ):
        if destination.exists() and not force and destination.stat().st_size == item.size:
            return destination
        command = [
            str(Path(sys.executable).expanduser()),
            str(_download_hf_ranged_script()),
            url,
            str(destination),
            "--size",
            str(item.size),
            "--workers",
            "8",
        ]
        env = os.environ.copy()
        env.update(
            {
                "WORDPIPE_PROGRESS_PROFILE": profile,
                "WORDPIPE_PROGRESS_FILENAME": item.filename,
                "WORDPIPE_PROGRESS_FILE_INDEX": str(file_index),
                "WORDPIPE_PROGRESS_FILE_COUNT": str(file_count),
                "WORDPIPE_PROGRESS_COMPLETED_BASE": str(completed_base),
                "WORDPIPE_PROGRESS_TOTAL_BYTES": str(total_bytes),
            }
        )
        _run_with_progress(command, progress, env=env)
        return destination

    path = hf_hub_download(
        repo_id=repo_id,
        filename=item.filename,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        force_download=force,
    )
    return Path(path)


def _download_hf_ranged_script() -> Path:
    return wordpipe_scripts_dir() / "download_hf_ranged.py"


def _prebuilt_download_plan(repo_id: str) -> list[_PrebuiltFile]:
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
    except ImportError:
        return [
            _PrebuiltFile(filename, filename in REQUIRED_PREBUILT_PROFILE_FILES)
            for filename in PREBUILT_PROFILE_FILES
        ]

    files: list[_PrebuiltFile] = []
    for filename in PREBUILT_PROFILE_FILES:
        required = filename in REQUIRED_PREBUILT_PROFILE_FILES
        try:
            metadata = get_hf_file_metadata(hf_hub_url(repo_id, filename))
        except EntryNotFoundError:
            if required:
                raise
            continue
        except HfHubHTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404 and not required:
                continue
            files.append(_PrebuiltFile(filename, required))
        else:
            files.append(_PrebuiltFile(filename, required, getattr(metadata, "size", None)))
    return files


def _download_message(item: _PrebuiltFile, index: int, total_files: int) -> str:
    file_number = f"file {index + 1}/{total_files}"
    if item.size:
        return f"Downloading {item.filename} ({_format_bytes(item.size)}, {file_number})"
    return f"Downloading {item.filename} ({file_number})"


def _download_fraction(completed_size: int, total_size: int, index: int, total_files: int) -> float:
    if total_size > 0:
        return max(0.0, min(1.0, completed_size / total_size))
    if total_files > 0:
        return max(0.0, min(1.0, index / total_files))
    return 0.0


def _downloaded_size(primary: Path, fallback: Path) -> int:
    for path in (primary, fallback):
        try:
            return path.stat().st_size
        except OSError:
            continue
    return 0


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{value} B"
            return f"{amount:.1f} {unit}"
        amount /= 1024.0


def install_prebuilt_profile(
    *,
    source: Path,
    model_root: Path,
    profile: str,
    family: ModelFamily = "multilingual",
    python: Path = Path(sys.executable),
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    spec = profile_spec(profile)
    runtime_dir = spec.runtime_dir(model_root, family)
    source_dynamic = _dynamic_config_at(source, family) is not None
    if runtime_dir.exists() and not force and profile_runtime_dir_valid(runtime_dir, profile, family) and (
        not source_dynamic or _dynamic_config_at(runtime_dir, family) is not None
    ):
        if not _profile_completion_marker(runtime_dir).exists():
            _write_profile_completion_marker(runtime_dir, profile=profile, family=family)
        _progress(progress, f"Using installed model profile: {runtime_dir}")
        return runtime_dir

    onnx_dir = spec.output_dir(model_root, family)
    if onnx_dir.exists() and not force and profile_runtime_dir_valid(onnx_dir, profile, family) and (
        not source_dynamic or _dynamic_config_at(onnx_dir, family) is not None
    ):
        if not _profile_completion_marker(onnx_dir).exists():
            _write_profile_completion_marker(onnx_dir, profile=profile, family=family)
        _progress(progress, f"Using cached ONNX profile: {onnx_dir}")
    else:
        prepared_source = _prepare_built_profile_source(source)
        try:
            _install_prepared_profile(
                prepared_source.path,
                onnx_dir,
                profile=profile,
                family=family,
                force=force,
            )
        finally:
            if prepared_source.cleanup_dir is not None:
                shutil.rmtree(prepared_source.cleanup_dir, ignore_errors=True)

    if not spec.emit_ort_format or _dynamic_config_at(onnx_dir, family) is not None:
        _progress(progress, f"Model profile ready: {onnx_dir}")
        return onnx_dir

    if runtime_dir.exists() and not force and profile_runtime_dir_valid(runtime_dir, profile, family):
        if not _profile_completion_marker(runtime_dir).exists():
            _write_profile_completion_marker(runtime_dir, profile=profile, family=family)
        _progress(progress, f"Using cached ORT runtime profile: {runtime_dir}")
        return runtime_dir

    command = [
        str(python.expanduser()),
        str(wordpipe_scripts_dir() / "convert_nemotron_to_ort_format.py"),
        str(onnx_dir),
        str(runtime_dir),
        "--force",
    ]
    _progress(progress, " ".join(command))
    _run_with_progress(command, progress)
    _write_profile_completion_marker(runtime_dir, profile=profile, family=family)
    _progress(progress, f"Model profile ready: {runtime_dir}")
    return runtime_dir


def _install_prepared_profile(
    source: Path,
    destination: Path,
    *,
    profile: str,
    family: ModelFamily,
    force: bool,
) -> None:
    if destination.exists():
        if not force:
            raise RuntimeError(
                f"profile output already exists at {destination}; pass --force to overwrite it"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(source, temporary, copy_function=_link_or_copy)
        _write_profile_completion_marker(temporary, profile=profile, family=family)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _link_or_copy(source: str | Path, destination: str | Path) -> str:
    if not str(source).endswith(".data"):
        shutil.copy2(source, destination)
        return str(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def download_streaming_profile(
    *, profile: str, model_root: Path, family: ModelFamily, latency_ms: int,
    repo_id: str | None = None, force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    """Fetch a generic profile, or fall back to the legacy shared-weight bundle."""
    if latency_ms <= 0 or latency_ms % 80:
        raise ValueError("streaming latency must be a positive multiple of 80 ms")
    from huggingface_hub import hf_hub_download, hf_hub_url
    spec = profile_spec(profile)
    repo = repo_id or spec.prebuilt_repo_for_family(family)
    shared_cache = prebuilt_profile_cache_dir(model_root, repo, profile)
    shared_cache.mkdir(parents=True, exist_ok=True)
    probe = hf_hub_download(
        repo_id=repo, filename="config.json", local_dir=shared_cache / "config-probe",
        force_download=force,
    )
    probe_config = json.loads(Path(probe).read_text())
    generic = dynamic_streaming_config(probe_config, family)
    if generic is not None and latency_ms // 80 in generic["supported_chunk_frames"]:
        return download_prebuilt_profile(
            profile=profile, model_root=model_root, family=family, repo_id=repo,
            force=force or _dynamic_config_at(shared_cache, family) is None, progress=progress,
        )
    manifest_path = hf_hub_download(
        repo_id=repo, filename="wordpipe-streaming-bundle.json", local_dir=shared_cache,
        force_download=force,
    )
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("format") != 1 or str(latency_ms) not in manifest.get("modes", {}):
        raise RuntimeError(f"{repo} does not publish the requested streaming mode")
    mode = manifest["modes"][str(latency_ms)]
    runtime = shared_cache / f"{latency_ms}ms"
    runtime.mkdir(exist_ok=True)
    names = [mode["encoder"], mode["config"], *manifest["shared_files"]]
    installed_base = spec.output_dir(model_root, family)
    for index, name in enumerate(dict.fromkeys(names)):
        if not isinstance(name, str) or not _safe_relative_path(name):
            raise RuntimeError("unsafe streaming bundle file name")
        expected = manifest["files"][name]
        cached = shared_cache / name
        cached.parent.mkdir(parents=True, exist_ok=True)
        candidates = [cached, installed_base / name]
        usable = next((path for path in candidates if path.is_file()
                       and path.stat().st_size == expected["size"]
                       and _sha256_file(path) == expected["sha256"]), None)
        if usable is None:
            usable = _download_prebuilt_file(
                repo_id=repo, item=_PrebuiltFile(name, True, expected["size"]),
                output_dir=shared_cache, force=force, hf_hub_download=hf_hub_download,
                url=hf_hub_url(repo, name), progress=progress, profile=profile,
                file_index=index + 1, file_count=len(names), completed_base=0,
                total_bytes=0,
            )
            if _sha256_file(usable) != expected["sha256"]:
                raise RuntimeError(f"streaming bundle checksum mismatch: {name}")
        target_name = "encoder.onnx" if name == mode["encoder"] else (
            "config.json" if name == mode["config"] else name)
        destination = runtime / target_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        _link_or_copy(usable, destination)
    _write_profile_completion_marker(runtime, profile=profile, family=family)
    return runtime


def _prepare_built_profile_source(source: Path) -> _PreparedProfileSource:
    source = source.expanduser()
    if source_is_built_profile(source):
        return _PreparedProfileSource(source)
    if source.is_file() and source.name.lower().endswith(".nemo"):
        raise RuntimeError(f"{source} is a NeMo source model, not a built Wordpipe model profile.")
    if source.is_file() and source_may_be_built_profile_archive(source) and tarfile.is_tarfile(source):
        import tempfile

        tempdir = Path(tempfile.mkdtemp(prefix="wordpipe-profile-"))
        try:
            with tarfile.open(source, "r:*") as archive:
                _extract_tar_safely(archive, tempdir)
            return _PreparedProfileSource(_find_built_profile_dir(tempdir), tempdir)
        except Exception:
            shutil.rmtree(tempdir, ignore_errors=True)
            raise
    if source.is_file() and source_may_be_built_profile_archive(source) and zipfile.is_zipfile(source):
        import tempfile

        tempdir = Path(tempfile.mkdtemp(prefix="wordpipe-profile-"))
        try:
            with zipfile.ZipFile(source) as archive:
                _extract_zip_safely(archive, tempdir)
            return _PreparedProfileSource(_find_built_profile_dir(tempdir), tempdir)
        except Exception:
            shutil.rmtree(tempdir, ignore_errors=True)
            raise
    raise RuntimeError(
        f"{source} is not a built Wordpipe model profile. Expected a directory or archive "
        "containing tokenizer.model plus encoder/decoder_joint ONNX or ORT files."
    )


def _extract_zip_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for info in archive.infolist():
        target = (destination / info.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Refusing unsafe zip member: {info.filename}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise RuntimeError(f"Refusing unsupported zip member: {info.filename}")
        archive.extract(info, destination)


def _extract_tar_safely(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Refusing unsafe tar member: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"Refusing unsupported tar member: {member.name}")
    archive.extractall(destination)


def _find_built_profile_dir(root: Path) -> Path:
    if source_is_built_profile(root):
        return root
    matches = [path for path in root.rglob("tokenizer.model") if source_is_built_profile(path.parent)]
    if len(matches) == 1:
        return matches[0].parent
    if not matches:
        raise RuntimeError(f"archive did not contain a built Wordpipe model profile: {root}")
    raise RuntimeError(f"archive contained multiple built Wordpipe model profiles: {root}")


def download_nemo_source(
    source: str = DEFAULT_NEMO_SOURCE_REPO,
    output_path: Path | None = None,
    *,
    family: ModelFamily = "multilingual",
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    candidate = Path(source).expanduser()
    if candidate.exists():
        _progress(progress, f"Using local source model: {candidate}")
        return candidate

    destination = (
        output_path.expanduser()
        if output_path is not None
        else default_nemo_source_path(family=family)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        _progress(progress, f"Using cached source model: {destination}")
        return destination

    try:
        from huggingface_hub import hf_hub_download

        _progress(progress, f"Downloading source model from {source}")
        env_value = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        try:
            import hf_transfer  # noqa: F401
        except ImportError:
            enable_hf_transfer = False
        else:
            enable_hf_transfer = True
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        try:
            downloaded = hf_hub_download(
                repo_id=source,
                filename=(
                    DEFAULT_ENGLISH_NEMO_SOURCE_FILENAME
                    if family == "english"
                    else DEFAULT_NEMO_SOURCE_FILENAME
                ),
                local_dir=destination.parent,
                local_dir_use_symlinks=False,
                force_download=force,
            )
        finally:
            if enable_hf_transfer:
                if env_value is None:
                    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
                else:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = env_value
        path = Path(downloaded)
        if path != destination:
            if destination.exists():
                destination.unlink()
            path.replace(destination)
        _progress(progress, f"Source model ready: {destination}")
        return destination
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the NeMo source model. "
            "Install it with hf_transfer support or pass a local .nemo path."
        ) from exc


def build_profile_command(
    *,
    source: Path,
    model_root: Path,
    profile: str,
    family: ModelFamily = "multilingual",
    python: Path,
    force: bool = False,
    streaming_latency_ms: int = 560,
    dynamic_streaming: bool = False,
) -> list[str]:
    spec = profile_spec(profile)
    output_dir = spec.output_dir(model_root, family)
    work_dir = profile_build_dir(model_root, profile, family)
    return [
        str(python),
        str(wordpipe_scripts_dir() / "build_nemotron_wordpipe_model.py"),
        str(source.expanduser()),
        str(output_dir),
        "--work-dir",
        str(work_dir),
        "--profile",
        spec.build_profile,
        "--model-family",
        family,
        *(["--streaming-latency-ms", str(streaming_latency_ms)] if streaming_latency_ms != 560 else []),
        *(["--dynamic-streaming"] if dynamic_streaming else []),
        *(["--emit-ort-format"] if spec.emit_ort_format and not dynamic_streaming else []),
        *(["--force"] if force else []),
    ]


def profile_build_dir(
    model_root: Path,
    profile: str,
    family: ModelFamily = "multilingual",
) -> Path:
    spec = profile_spec(profile)
    base = model_root.expanduser() / "build"
    return base / spec.name if family == "multilingual" else base / family / spec.name


def wordpipe_scripts_dir() -> Path:
    override = os.environ.get("WORDPIPE_SCRIPTS_DIR")
    if override:
        return Path(override).expanduser()

    repo_scripts = Path(__file__).resolve().parents[2] / "scripts"
    if (repo_scripts / "build_nemotron_wordpipe_model.py").exists():
        return repo_scripts

    installed_scripts = Path(sys.prefix) / "share" / "wordpipe" / "scripts"
    if (installed_scripts / "build_nemotron_wordpipe_model.py").exists():
        return installed_scripts

    app_scripts = Path("/app/share/wordpipe/scripts")
    if (app_scripts / "build_nemotron_wordpipe_model.py").exists():
        return app_scripts

    return repo_scripts


def build_model_profile(
    *,
    source: Path,
    model_root: Path,
    profile: str,
    family: ModelFamily = "multilingual",
    python: Path = Path(sys.executable),
    force: bool = False,
    dry_run: bool = False,
    keep_build_dir: bool = False,
    progress: ProgressCallback | None = None,
    streaming_latency_ms: int = 560,
    dynamic_streaming: bool = False,
) -> Path:
    command = build_profile_command(
        source=source,
        model_root=model_root,
        profile=profile,
        family=family,
        python=python,
        force=force,
        streaming_latency_ms=streaming_latency_ms,
        dynamic_streaming=dynamic_streaming,
    )
    rendered_command = " ".join(command)
    print(rendered_command, file=sys.stderr)
    _progress(progress, rendered_command)
    build_dir = profile_build_dir(model_root, profile, family)
    if not dry_run:
        _run_with_progress(command, progress)
        _write_profile_completion_marker(
            profile_runtime_dir(model_root, profile, family),
            profile=profile,
            family=family,
        )
        if not keep_build_dir and build_dir.exists():
            _progress(progress, f"Removing build intermediates: {build_dir}")
            shutil.rmtree(build_dir)
    runtime_dir = profile_runtime_dir(model_root, profile, family)
    _progress(progress, f"Model profile ready: {runtime_dir}")
    return runtime_dir


def _run_with_progress(
    command: list[str],
    progress: ProgressCallback | None,
    *,
    env: dict[str, str] | None = None,
) -> None:
    if progress is None:
        subprocess.run(command, check=True, env=env)
        return

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        message = line.rstrip()
        if message:
            _progress(progress, message)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _progress_event(progress: ProgressCallback | None, **payload: object) -> None:
    _progress(progress, "wordpipe-progress " + json.dumps(payload, sort_keys=True))


def _profile_completion_marker(runtime_dir: Path) -> Path:
    return runtime_dir / PROFILE_COMPLETION_MARKER


def _write_profile_completion_marker(
    runtime_dir: Path,
    *,
    profile: str,
    family: ModelFamily = "multilingual",
) -> None:
    if not _runtime_structure_valid(runtime_dir):
        raise RuntimeError(
            f"model runtime profile is incomplete at {runtime_dir}; expected tokenizer.model "
            "plus encoder and decoder_joint ONNX/ORT graphs"
        )
    if not _profile_config_valid_if_present(runtime_dir, profile, family):
        raise RuntimeError(
            f"model runtime profile at {runtime_dir} does not match Wordpipe's "
            f"{profile!r} fixed-shape projected-cache contract"
        )
    marker = _profile_completion_marker(runtime_dir)
    files = []
    for path in _runtime_files(runtime_dir):
        relative = path.relative_to(runtime_dir).as_posix()
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    payload = {
        "format": 1,
        "profile": profile,
        "model_family": family,
        "created_at_unix": int(time.time()),
        "files": files,
    }
    temporary = marker.with_name(f"{marker.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(marker)


def _profile_completion_marker_valid(runtime_dir: Path, *, verify_hashes: bool) -> bool:
    marker = _profile_completion_marker(runtime_dir)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("format") != 1:
        return False
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return False
    for item in files:
        if not isinstance(item, dict):
            return False
        relative = item.get("path")
        expected_size = item.get("size")
        expected_sha256 = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_size, int):
            return False
        if verify_hashes and not isinstance(expected_sha256, str):
            return False
        if not _safe_relative_path(relative):
            return False
        path = runtime_dir / relative
        try:
            if path.stat().st_size != expected_size:
                return False
        except OSError:
            return False
        if verify_hashes and _sha256_file(path) != expected_sha256:
            return False
    return True


def _runtime_files(runtime_dir: Path) -> list[Path]:
    files: list[Path] = []
    marker = _profile_completion_marker(runtime_dir)
    for path in runtime_dir.rglob("*"):
        if not path.is_file():
            continue
        if path == marker or path.name.startswith(f"{PROFILE_COMPLETION_MARKER}.tmp-"):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(runtime_dir).as_posix())


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
