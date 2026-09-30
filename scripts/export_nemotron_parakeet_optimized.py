#!/usr/bin/env python3
"""Export Nemotron streaming ASR for parakeet-rs with sherpa-style quantization.

This script keeps the file layout and ONNX ABI expected by parakeet-rs:

  - encoder.onnx
  - decoder_joint.onnx
  - tokenizer.model

but borrows the useful parts of sherpa-onnx's NeMo export pipeline:

  1. export a fresh FP32 NeMo encoder graph,
  2. run one full ONNX Runtime dynamic QUInt8 quantization pass,
  3. rewrite the quantized encoder to use projected K/V cache.

The point is to avoid post-hoc quantizing an already partially quantized graph.
Sherpa starts from FP32 and lets ORT quantize MatMul and Conv coherently; this
script does the same while preserving the parakeet-rs runtime interface.
"""
from __future__ import annotations

import argparse
import functools
import gc
import glob
import json
import logging
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
import soundfile as sf
import torch

from rewrite_nemotron_projected_kv_cache import rewrite_model as rewrite_projected_cache

ORT_OPTIMIZATION_LEVELS = {"disable", "basic", "extended", "all"}


INPUT_NAMES = [
    "processed_signal",
    "processed_signal_length",
    "cache_last_channel",
    "cache_last_time",
    "cache_last_channel_len",
]
OUTPUT_NAMES = [
    "encoded",
    "encoded_len",
    "cache_last_channel_next",
    "cache_last_time_next",
    "cache_last_channel_len_next",
]


def patch_legacy_torch_export() -> None:
    version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    marker = "_wordpipe_legacy_onnx_patched"
    if version < (2, 9) or getattr(torch.onnx.export, marker, False):
        return

    original = torch.onnx.export

    @functools.wraps(original)
    def patched(*args: Any, **kwargs: Any):
        kwargs.setdefault("dynamo", False)
        return original(*args, **kwargs)

    setattr(patched, marker, True)
    torch.onnx.export = patched


def load_model(input_path: str, device: torch.device):
    import nemo.collections.asr as nemo_asr

    if Path(input_path).exists():
        print(f"[export] restoring NeMo model from {input_path}", flush=True)
        return nemo_asr.models.ASRModel.restore_from(input_path, map_location=device)
    print(f"[export] loading NeMo model {input_path}", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=input_path)
    try:
        model = model.to(device)
    except Exception:
        pass
    return model


def extract_tokenizer_from_nemo(input_path: str, output_dir: Path) -> bool:
    path = Path(input_path)
    if not path.exists():
        return False
    try:
        with tarfile.open(path, "r:*") as tar:
            for member in tar.getnames():
                if member.endswith("tokenizer.model"):
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    (output_dir / "tokenizer.model").write_bytes(extracted.read())
                    return True
    except tarfile.TarError:
        return False
    return False


def save_tokenizer(model, input_path: str, output_dir: Path) -> None:
    if extract_tokenizer_from_nemo(input_path, output_dir):
        return

    tokenizer = getattr(model, "tokenizer", None)
    for attr in ("tokenizer", "sp_model", "model", "processor"):
        candidate = getattr(tokenizer, attr, None)
        serialized = getattr(candidate, "serialized_model_proto", None)
        if callable(serialized):
            (output_dir / "tokenizer.model").write_bytes(serialized())
            return

    raise RuntimeError(
        "Could not extract tokenizer.model. Pass a .nemo input, or extend "
        "save_tokenizer() for this NeMo tokenizer wrapper."
    )


def jsonable(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def cfg_get(cfg: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
    except Exception:
        return default


def prompt_dictionary(model) -> dict[str, int]:
    try:
        from omegaconf import OmegaConf

        value = model.cfg.model_defaults.prompt_dictionary
        return {str(k): int(v) for k, v in OmegaConf.to_container(value, resolve=True).items()}
    except Exception:
        return {}


def tokenizer_vocab_size(model) -> int | None:
    tokenizer = getattr(model, "tokenizer", None)
    value = getattr(tokenizer, "vocab_size", None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model, drop_extra: int, prompted: bool):
        super().__init__()
        self.encoder = model.encoder
        self.drop_extra = drop_extra
        self.prompted = prompted
        if prompted:
            self.prompt_kernel = model.prompt_kernel
            self.num_prompts = int(model.num_prompts)

    def forward(
        self,
        processed_signal,
        processed_signal_length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        prompt_index=None,
    ):
        encoded, enc_len, ch_next, tm_next, len_next = self.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop_extra,
        )
        if self.prompted:
            encoded = encoded.transpose(1, 2)
            batch, frames, _ = encoded.shape
            prompt = torch.zeros(
                batch,
                frames,
                self.num_prompts,
                dtype=encoded.dtype,
                device=encoded.device,
            )
            prompt.scatter_(
                2,
                prompt_index.view(batch, 1, 1).expand(-1, frames, -1),
                1.0,
            )
            encoded = self.prompt_kernel(torch.cat([encoded, prompt], dim=-1))
            encoded = encoded.transpose(1, 2)
        return encoded, enc_len, ch_next, tm_next, len_next


def verify_prompt_wrapper(
    model,
    wrapper: EncoderWrapper,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    prompt_dict: dict[str, int],
    verify_lang: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None, int | None, float | None]:
    if not wrapper.prompted:
        return None, None, None, None, None

    lang = verify_lang if verify_lang in prompt_dict else "auto"
    if lang not in prompt_dict:
        lang = next(iter(prompt_dict), None)
    if lang is None:
        raise RuntimeError("Prompted model did not expose a prompt_dictionary.")

    if hasattr(model, "set_inference_prompt"):
        model.set_inference_prompt(lang)
    prompt_index = torch.tensor(
        [int(getattr(model, "_inference_prompt_index", prompt_dict[lang]))],
        dtype=torch.long,
    )
    processed_signal, processed_signal_length, cache_last_channel, cache_last_time, cache_last_channel_len = inputs
    with torch.no_grad():
        raw, encoded_len, _, _, _ = model.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=wrapper.drop_extra,
        )
        reference = model._apply_prompt_to_encoded(raw)
        wrapped, _, _, _, _ = wrapper(
            processed_signal,
            processed_signal_length,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            prompt_index,
        )
    diff = float((wrapped - reference).abs().max().item())
    print(f"[export] prompt wrapper parity {lang} idx={int(prompt_index[0])}: max diff={diff:.2e}", flush=True)
    if diff > 1e-4:
        raise RuntimeError(
            "Prompt wrapper does not match NeMo _apply_prompt_to_encoded; "
            "aborting before writing ONNX weights."
        )
    return reference, encoded_len, lang, int(prompt_index[0]), diff


def consolidate_onnx(input_path: Path, output_path: Path, external_data_name: str) -> None:
    model = onnx.load(input_path, load_external_data=True)
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=0,
    )


def quantize_to_single_file(input_path: Path, output_path: Path, *, per_channel: bool) -> None:
    print(
        f"[export] quantizing {input_path.name} -> {output_path.name} perChannel={per_channel}",
        flush=True,
    )
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        per_channel=per_channel,
        use_external_data_format=False,
    )
    onnx.checker.check_model(str(output_path))


def ort_optimize_to_file(input_path: Path, output_path: Path, level: str, threads: int) -> None:
    import onnxruntime as ort

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    print(f"[export] ORT optimizing {input_path.name} -> {output_path.name} level={level}", flush=True)
    options = ort.SessionOptions()
    options.optimized_model_filepath = str(output_path)
    options.graph_optimization_level = levels[level]
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])
    onnx.checker.check_model(str(output_path))


def cleanup_patterns(directory: Path, patterns: list[str]) -> None:
    for pattern in patterns:
        for path in glob.glob(str(directory / pattern)):
            try:
                Path(path).unlink()
            except OSError:
                pass


def make_streaming_example(model, sample_rate: int):
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=False,
        pad_and_drop_preencoded=True,
    )
    audio = (np.random.randn(sample_rate * 2).astype(np.float32) * 0.1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav:
        sf.write(wav.name, audio, sample_rate)
        wav_path = wav.name
    try:
        streaming_buffer.append_audio_file(wav_path, stream_id=-1)
        return next(iter(streaming_buffer))
    finally:
        Path(wav_path).unlink(missing_ok=True)


def export_decoder_joint(model, output_dir: Path) -> Path:
    print("[export] exporting decoder/joint FP32 ONNX", flush=True)
    temp_prefix = output_dir / "temp_model"
    with torch.no_grad():
        model.export(output=str(temp_prefix) + ".onnx", check_trace=False)

    final_path = output_dir / "decoder_joint.fp32.onnx"
    for path in output_dir.glob("*.onnx"):
        name = path.name.lower()
        if "decoder" in name and "joint" in name:
            path.rename(final_path)
            return final_path
    raise RuntimeError("NeMo export did not produce a decoder_joint ONNX file.")


def projected_cache_current_projection(args: argparse.Namespace) -> str:
    if args.projected_cache_current_projection != "auto":
        return args.projected_cache_current_projection
    return "dynamic-int8" if args.quantize else "fp32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help=".nemo path or NeMo/Hugging Face model id")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--model-family",
        choices=("multilingual", "english"),
        default="multilingual",
    )
    parser.add_argument(
        "--left-context",
        type=int,
        default=56,
        help="Attention left context. Nemotron 3.5 multilingual expects 56.",
    )
    parser.add_argument("--right-context", type=int, default=6)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--projected-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply projected K/V cache rewrite after quantization.",
    )
    parser.add_argument(
        "--projected-cache-current-projection",
        choices=("auto", "dynamic-int8", "fp32"),
        default="auto",
        help=(
            "Projection used for the current chunk inside the projected-cache "
            "rewrite. auto preserves the existing behavior: dynamic-int8 for "
            "quantized graphs, fp32 for unquantized graphs."
        ),
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run full dynamic QUInt8 quantization before projected-cache rewrite.",
    )
    parser.add_argument(
        "--quantize-per-channel",
        action="store_true",
        help="Use per-channel weight quantization during dynamic QUInt8 quantization.",
    )
    parser.add_argument(
        "--fp32-decoder",
        action="store_true",
        help=(
            "Keep decoder_joint.onnx as the FP32 NeMo export even when the "
            "encoder is dynamically quantized. Benchmark before promoting; "
            "this can trade strict spelling WER for throughput."
        ),
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Stop after writing FP32 encoder/decoder artifacts for a separate transform step.",
    )
    parser.add_argument(
        "--ort-optimize-final",
        choices=sorted(ORT_OPTIMIZATION_LEVELS),
        help="Serialize the final encoder through ONNX Runtime at the selected optimization level.",
    )
    parser.add_argument(
        "--ort-optimize-threads",
        type=int,
        default=1,
        help="Intra-op threads to use while serializing the ORT-optimized final encoder.",
    )
    parser.add_argument(
        "--verify-lang",
        default="en-US",
        help="Language prompt to use for pre-export wrapper parity verification.",
    )
    args = parser.parse_args()

    logging.getLogger("nemo_logging").setLevel(logging.ERROR)
    try:
        from nemo.core.classes.common import typecheck

        typecheck.set_typecheck_enabled(False)
    except Exception:
        pass

    patch_legacy_torch_export()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    device = torch.device("cpu")
    model = load_model(args.input, device)
    model.eval()
    print(f"[export] model class: {type(model).__name__}", flush=True)
    save_tokenizer(model, args.input, args.output_dir)

    if hasattr(model.encoder, "set_default_att_context_size"):
        model.encoder.set_default_att_context_size([args.left_context, args.right_context])

    streaming_cfg = model.encoder.streaming_cfg
    print(f"[export] att_context_size=[{args.left_context}, {args.right_context}]", flush=True)
    print(f"[export] streaming_cfg={streaming_cfg}", flush=True)
    model.encoder.setup_streaming_params(
        chunk_size=args.right_context + 1,
        shift_size=args.right_context + 1,
    )
    drop_extra = int(getattr(streaming_cfg, "drop_extra_pre_encoded", 0))

    processed_signal, processed_signal_length = make_streaming_example(model, args.sample_rate)
    cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(
        batch_size=1
    )

    prompted = hasattr(model, "prompt_kernel") and hasattr(model, "num_prompts")
    prompt_dict = prompt_dictionary(model) if prompted else {}
    prompt_index = torch.tensor([prompt_dict.get("auto", 101)], dtype=torch.long)

    wrapper = EncoderWrapper(model, drop_extra, prompted).eval()
    encoder_inputs = (
        processed_signal,
        processed_signal_length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
    )
    reference_encoded = None
    reference_encoded_len = None
    verified_lang = None
    verified_prompt_index = None
    parity_diff = None
    if prompted:
        reference_encoded, reference_encoded_len, verified_lang, verified_prompt_index, parity_diff = verify_prompt_wrapper(
            model,
            wrapper,
            encoder_inputs,
            prompt_dict,
            args.verify_lang,
        )
        prompt_index = torch.tensor([verified_prompt_index], dtype=torch.long)
    input_names = list(INPUT_NAMES)
    dynamic_axes = {
        "processed_signal": {0: "batch", 2: "mel_frames"},
        "processed_signal_length": {0: "batch"},
        "encoded": {0: "batch", 2: "encoder_frames"},
        "encoded_len": {0: "batch"},
    }
    if prompted:
        encoder_inputs = (*encoder_inputs, prompt_index)
        input_names.append("prompt_index")
        dynamic_axes["prompt_index"] = {0: "batch"}

    fp32_encoder = args.output_dir / "encoder.fp32.onnx"
    print("[export] exporting encoder FP32 ONNX", flush=True)
    torch.onnx.export(
        wrapper,
        encoder_inputs,
        str(fp32_encoder),
        input_names=input_names,
        output_names=OUTPUT_NAMES,
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )

    consolidated_encoder = args.output_dir / "encoder.fp32.consolidated.onnx"
    print("[export] consolidating encoder external data", flush=True)
    consolidate_onnx(fp32_encoder, consolidated_encoder, "encoder.fp32.consolidated.data")
    cleanup_patterns(
        args.output_dir,
        ["encoder.fp32.onnx*", "*.weight", "*MatMul*", "Constant_*", "onnx__*", "encoder.pre_encode*"],
    )

    decoder_fp32 = export_decoder_joint(model, args.output_dir)

    config = {
        "model_name": args.input,
        "model_family": args.model_family,
        "sample_rate": args.sample_rate,
        "n_mels": 128,
        "subsampling_factor": int(cfg_get(model.cfg.encoder, "subsampling_factor", 8)),
        "att_context_size": [args.left_context, args.right_context],
        "left_context": args.left_context,
        "right_context": args.right_context,
        "chunk_size_output_frames": args.right_context + 1,
        "drop_extra_pre_encoded": drop_extra,
        "num_encoder_layers": int(cache_last_channel.shape[0]),
        "hidden_dim": int(cache_last_channel.shape[3]),
        "conv_context": int(cache_last_time.shape[3]),
        "vocab_size": tokenizer_vocab_size(model),
        "blank_id": tokenizer_vocab_size(model),
        "num_prompts": int(getattr(model, "num_prompts", 0)) if prompted else 0,
        "projected_cache": args.projected_cache,
        "projected_cache_current_projection": (
            projected_cache_current_projection(args) if args.projected_cache else None
        ),
        "dynamic_quint8_quantization": args.quantize,
        "dynamic_quint8_per_channel": args.quantize_per_channel if args.quantize else False,
        "decoder_joint_fp32": bool(args.fp32_decoder or not args.quantize),
        "ort_optimized_final_encoder": args.ort_optimize_final,
        "prompt_dictionary": prompt_dict,
        "preprocessor": jsonable(getattr(model.cfg, "preprocessor", {})),
        "cache_shapes": {
            "cache_last_channel": list(cache_last_channel.shape),
            "cache_last_time": list(cache_last_time.shape),
            "cache_last_channel_len": list(cache_last_channel_len.shape),
        },
        "test_input": {
            "mel_shape": list(processed_signal.shape),
            "mel_length": int(processed_signal_length[0]),
            "prompt_index": verified_prompt_index,
        },
        "test_output": {
            "encoded_shape": list(reference_encoded.shape) if reference_encoded is not None else None,
            "encoded_len": int(reference_encoded_len[0]) if reference_encoded_len is not None else None,
            "verify_lang": verified_lang,
            "wrapper_max_diff": parity_diff,
        },
    }
    (args.output_dir / "export_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.export_only:
        cleanup_patterns(
            args.output_dir,
            [
                "encoder.fp32.onnx",
                "encoder.fp32.onnx.data",
                "encoder-temp_model.onnx",
                "temp_model*",
                "Constant_*",
                "onnx__*",
                "layers.*",
                "pre_encode.*",
                "prompt_kernel.*",
            ],
        )
        gc.collect()
        print(f"[export] wrote FP32 export artifacts to {args.output_dir}", flush=True)
        return

    encoder_for_projected = consolidated_encoder
    if args.quantize:
        quant_encoder = args.output_dir / "encoder.quant.onnx"
        quant_decoder = args.output_dir / "decoder_joint.onnx"
        quantize_to_single_file(consolidated_encoder, quant_encoder, per_channel=args.quantize_per_channel)
        if args.fp32_decoder:
            shutil.copy2(decoder_fp32, quant_decoder)
        else:
            quantize_to_single_file(decoder_fp32, quant_decoder, per_channel=args.quantize_per_channel)
        encoder_for_projected = quant_encoder
    else:
        shutil.copy2(decoder_fp32, args.output_dir / "decoder_joint.onnx")

    final_encoder = args.output_dir / "encoder.onnx"
    current_projection = projected_cache_current_projection(args)
    if args.projected_cache:
        print("[export] rewriting encoder with projected cache", flush=True)
        rewrite_projected_cache(
            encoder_for_projected,
            final_encoder,
            current_projection,
            external_data=not args.quantize,
        )
    else:
        shutil.copy2(encoder_for_projected, final_encoder)

    if args.ort_optimize_final:
        optimized_encoder = args.output_dir / "encoder.ort_optimized.onnx"
        ort_optimize_to_file(
            final_encoder,
            optimized_encoder,
            args.ort_optimize_final,
            args.ort_optimize_threads,
        )
        optimized_encoder.replace(final_encoder)

    cleanup_patterns(
        args.output_dir,
        [
            "encoder.fp32.consolidated.onnx",
            "encoder.fp32.consolidated.data",
            "encoder.quant.onnx",
            "decoder_joint.fp32.onnx",
            "temp_model*",
        ],
    )

    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    gc.collect()
    print(f"[export] wrote {args.output_dir}")
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
