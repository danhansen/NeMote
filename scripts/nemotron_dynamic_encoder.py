"""Tensor-valued streaming context for the supported NeMo export ABI.

Only used while exporting/evaluating the encoder wrapper. The native NeMo
methods are restored even if the forward call fails. Learned layers, positional
encoding, convolutions, and cache updates remain NeMo's implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch


def chunked_attention_masks(
    *, chunk_frames, left_context: int, padding_length,
    max_audio_length, offset, device,
):
    """Equivalent to NeMo's chunked_limited _create_masks for finite context."""
    frames = torch.arange(max_audio_length, dtype=torch.int64, device=device)
    chunks = torch.div(frames, chunk_frames, rounding_mode="trunc")
    differences = chunks.unsqueeze(1) - chunks.unsqueeze(0)
    left_chunks = torch.div(left_context, chunk_frames, rounding_mode="trunc")
    allowed = (differences <= left_chunks) & (differences >= 0)
    valid = frames.unsqueeze(0) < padding_length.unsqueeze(1)
    if offset is not None:
        valid = valid & (frames.unsqueeze(0) >= offset.unsqueeze(1))
    padding_allowed = valid.unsqueeze(1) & valid.unsqueeze(2)
    return ~valid, ~(padding_allowed & allowed.unsqueeze(0))


def validate_dynamic_encoder(encoder) -> None:
    if encoder.att_context_style != "chunked_limited":
        raise ValueError("dynamic export requires chunked_limited attention")
    if encoder.self_attention_model != "rel_pos":
        raise ValueError("dynamic export currently supports rel_pos attention")
    if encoder.streaming_cfg.cache_drop_size != 0:
        raise ValueError("dynamic export requires non-overlapping chunks")
    if encoder.streaming_cfg.last_channel_cache_size <= 0:
        raise ValueError("dynamic export requires a finite channel cache")
    if int(encoder.att_context_size[0]) != encoder.streaming_cfg.last_channel_cache_size:
        raise ValueError("attention left context must match the channel cache")


def supported_chunk_frames(encoder) -> list[int]:
    """Read finite trained contexts for this encoder's fixed left-cache ABI."""
    validate_dynamic_encoder(encoder)
    left = int(encoder.att_context_size[0])
    frames = sorted({
        int(context[1]) + 1 for context in encoder.att_context_size_all
        if len(context) == 2 and int(context[0]) == left and int(context[1]) >= 0
    })
    if not frames or any(left % size for size in frames):
        raise ValueError("checkpoint exposes no valid finite chunked contexts")
    return frames


@contextmanager
def dynamic_attention_context(encoder, chunk_frames):
    """Keep context arithmetic in the traced tensor graph, not Python config."""
    had_instance_method = "_create_masks" in encoder.__dict__
    previous_instance_method = encoder.__dict__.get("_create_masks")
    left_context = int(encoder.att_context_size[0])

    def create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        return chunked_attention_masks(
            chunk_frames=chunk_frames, left_context=left_context,
            padding_length=padding_length, max_audio_length=max_audio_length,
            offset=offset, device=device,
        )

    encoder._create_masks = MethodType(create_masks, encoder)
    try:
        yield
    finally:
        if had_instance_method:
            encoder._create_masks = previous_instance_method
        else:
            del encoder._create_masks
