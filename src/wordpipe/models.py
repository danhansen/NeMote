"""Nemotron model paths and capabilities shared by the download helpers."""
from pathlib import Path

MODEL_FAMILIES = ("english", "multilingual")
STREAMING_LATENCIES = (80, 160, 560, 1120)


def profile_runtime_dir(model_root, profile="nemo-q8", family="english"):
    if profile != "nemo-q8" or family not in MODEL_FAMILIES:
        raise ValueError("Only English and Multilingual Nemotron models are supported")
    return Path(model_root) / ("nemotron-nemo-en-q8" if family == "english" else "nemotron-nemo-q8")


def profile_installed(model_root, profile="nemo-q8", family="english"):
    from .nemo_models import MODELS
    path = profile_runtime_dir(model_root, profile, family) / "model.gguf"
    try:
        with path.open("rb") as model:
            return path.stat().st_size == MODELS[family][3] and model.read(4) == b"GGUF"
    except OSError:
        return False


def profile_supports_streaming_latency(runtime_dir, latency_ms):
    return latency_ms in STREAMING_LATENCIES
