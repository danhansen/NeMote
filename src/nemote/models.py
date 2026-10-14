"""Nemotron model paths and capabilities shared by the download helpers."""
from pathlib import Path

MODEL_FAMILIES = ("english", "multilingual")
STREAMING_LATENCIES = (80, 160, 560, 1120)


def profile_runtime_dir(model_root, family="english"):
    if family not in MODEL_FAMILIES:
        raise ValueError("Only English and Multilingual Nemotron models are supported")
    return Path(model_root) / ("nemotron-nemo-en-q8" if family == "english" else "nemotron-nemo-q8")
