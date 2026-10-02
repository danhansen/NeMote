# Sayboard Optimization Harvest

This document tracks the ONNX/ORT-relevant optimizations found in
`/home/dhansen/Downloads/Sayboard` and their NeMote status. The goal is to
avoid losing useful ideas while keeping each candidate tied to a repeatable
benchmark and WER check.

Benchmark convention for NeMote experiments:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  candidate=path/to/model \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/name.json

.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/name.json
```

## Current NeMote Default Candidate

The current conservative speed candidate remains
`build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort`:

- projected K/V cache;
- fixed c56 streaming shapes;
- ORT `extended` serialized encoder;
- feed-forward MatMul/Gemm blocks dequantized back to FP32.

On the concatenated LibriSpeech WAV, this candidate has repeatedly scored
`9 / 313 = 2.88%` rough WER. Throughput varies with power state, so compare
only within the same benchmark run.

## Harvest Matrix

| Sayboard optimization | Sayboard source | NeMote status | Evidence / next action |
| --- | --- | --- | --- |
| Deterministic LibriSpeech ablation protocol with timing and WER | `scripts/parakeet_ablation/run_ablation.py`, `README.md` | Ported | `scripts/benchmark_parakeet_variant.py`, `scripts/eval_librispeech_backends.py`, and `scripts/score_benchmark_wer.py` provide the long-WAV 3-run median plus rough WER convention. |
| Fixed streaming shapes | `rewrite_fixed_streaming_shapes.py` | Ported and kept | `scripts/build_nemotron_fixed_shape_model.py`; documented speed win in `docs/optimization-experiments.md`. |
| Resolve symbolic dims and replace static `Shape` nodes | `rewrite_fixed_streaming_shapes.py`, `sanitize_tflite_encoder.py` | Ported and kept | Fixed-shape build resolves known dims and replaces static `Shape` nodes. The effect is folded into current default builds. |
| ORT serialized graph optimization | `sanitize_tflite_encoder.py` uses `basic` for converter readiness; Sayboard runtime uses ORT/LiteRT sessions | Ported and benchmarked | NeMote tested runtime levels/session options. ORT `extended` serialized fixed-shape graph is the current default; `basic` is useful mainly for converter-safe folding. |
| Projected K/V cache | `rewrite_projected_kv_cache.py` | Ported and kept | NeMote supports projected-cache graphs and runtime cache rolling. Raw-cache FP32 control showed same WER as projected-cache FP32, with projected cache much faster. |
| Layered projected-cache ABI | `rewrite_projected_kv_cache.py` | Ported and kept | NeMote uses per-layer `cache_key_layer_N` / `cache_value_layer_N` inputs and `projected_current_*` outputs, matching Sayboard's layered ABI. |
| Stacked projected-cache ABI | `rewrite_projected_kv_cache.py` | Not ported | Not useful for NeMote today. Layered ABI avoids in-graph cache rolling and matches the Rust runtime. |
| FP32 current K/V projection after quantized source graph | `build_deployed_model.py` calls projected-cache rewrite with `current_projection="fp32"` | Implemented and rejected | `scripts/run_sayboard_harvest_experiments.py` built the variant. Same-run benchmark: baseline `0.642` real-audio RTF, FP32-current projection `0.729`; WER worsened from `9 / 313 = 2.88%` to `12 / 313 = 3.83%`. |
| Dynamic int8 quantization by operator family | `run_ablation.py` default variants | Ported and benchmarked for NeMote-relevant graph forms | NeMote tested the sherpa-derived broad quantized baseline, FFN dequantization, attention projection dequantization, pre-encoder output dequantization, conv dequantization, layer slices/even/odd variants, MatMulNBits, and clean-FP32 default dynamic quantization. Current best dequantizes FFN blocks back to FP32. |
| Per-channel dynamic quantization sweeps | `run_ablation.py` `*_pc` variants | Implemented and rejected for current path | `scripts/transform_nemotron_parakeet_export.py --quantize-per-channel` and `scripts/run_sayboard_harvest_experiments.py --experiment per-channel-quantization` test this from the clean FP32 export. Same-run benchmark: baseline `0.645` real-audio RTF, per-channel `0.723`; WER worsened from `9 / 313 = 2.88%` to `10 / 313 = 3.19%`. |
| Dynamic Conv quantization | `README.md` rejected conv variants | Ported and rejected | NeMote `scripts/quantize_nemotron_conv_dynamic.py` and conv dequant experiments showed throughput/accuracy tradeoffs were not attractive. |
| MatMul/Gemm dynamic quantization from fixed raw-cache FP32 | `build_deployed_model.py`, `run_ablation.py` `fullpre_*` variants | Tested and rejected for current path | NeMote tested clean-FP32 default dynamic quantization with projected cache: baseline `0.655` real-audio RTF, FP32-default quantization `0.747`; WER worsened from `9 / 313 = 2.88%` to `10 / 313 = 3.19%`. FP32 raw-cache/projected-cache controls also work, but the FP32 export path scores worse WER than the sherpa-derived candidate. |
| Remove fixed length input and replace with initializer | `rewrite_fixed_streaming_shapes.py --keep-length-input` default removes `length` | Implemented and rejected | `scripts/run_sayboard_harvest_experiments.py` built the ABI with `--constant-processed-signal-length`. Same-run benchmark: baseline `0.641` real-audio RTF, fixed-length `0.706`; WER worsened from `9 / 313 = 2.88%` to `10 / 313 = 3.19%`. |
| MatMulInteger quantization tail to FP32 `Gemm` cleanup | `rewrite_quantized_matmulinteger_to_gemm.py` | Not directly applicable as an ORT speed optimization | Sayboard used this before TFLite conversion. For ORT it intentionally dequantizes quantized blocks, overlapping with NeMote's targeted FFN FP32 dequantization but too broad for the default path. |
| TFLite/LiteRT conversion and static-RHS BMM to FC rewrite | `scripts/litert_spike/*` | Out of current Linux ORT scope | Useful if NeMote later adds a LiteRT backend. Sayboard's own notes show ORT was faster than host LiteRT FP32 for the simple encoder, while Android/device results were the main motivation. |
| Custom Android ORT with NCHWc/NEON | `build_onnxruntime_android_nchwc.sh` | Not applicable to Linux x86_64 | The analogous NeMote path is `scripts/build_onnxruntime_ivybridge.sh`, but current Python/Rust ORT binaries already execute acceptably and custom builds are deferred. |
| Runtime thread count sweep | Rust bridge config defaults and NeMote benchmark harness | Ported and benchmarked | NeMote tested 1-4 intra-op threads; 2 threads remains the practical benchmark default for this CPU. |
| ORT memory pattern / arena / parallel execution toggles | NeMote follow-up inspired by runtime tuning | Ported and benchmarked | `scripts/benchmark_parakeet_variant.py` exposes these toggles. Defaults remained best or close enough; explicit parallel execution was worse. |
| Allocation reduction in Rust runtime | Sayboard Rust bridge and NeMote fork changes | Ported where applicable | NeMote runtime updates caches in place and reuses audio buffers. Prior A/B showed small or noisy gains; no remaining obvious Sayboard allocation trick is unported for the Nemotron path. |

## Experiment Runner

The remaining Sayboard-harvest build/benchmark sequence is codified in one
wrapper:

```sh
.venv/bin/python scripts/run_sayboard_harvest_experiments.py --force
```

The wrapper rebuilds `target/release/wordpipe-parakeet-worker`, builds the model
variants, runs the long-WAV 3-run median benchmark under GNOME `balanced`, and
then scores WER. Use `--experiment fixed-length`,
`--experiment fp32-default-quantization`,
`--experiment fp32-current-projection`, or
`--experiment per-channel-quantization` to run one variant.

## Remaining Investigation

The ONNX/ORT-relevant Sayboard optimization ideas have same-WAV benchmark
results or are explicitly out of scope. The remaining investigation is export
parity:

1. Export parity investigation.
   `scripts/compare_nemotron_model_packages.py` now records package-level
   differences. Current evidence points away from tokenizer/prompt/preprocessor
   differences and toward encoder graph/export structure: the FP32-export
   quantized variants do not produce the same `FusedMatMul`/FFN-dequantized
   graph as the current sherpa-derived best model.

## Rejected Or Deferred

- Broad `MatMulNBits` on this Ivy Bridge CPU: model size improves, but
  throughput was much worse and WER did not improve.
- Conv quantization/dequantization variants: not attractive on the current
  rough WER sample.
- Full TFLite/LiteRT backend: outside the current Linux Wayland/GNOME ORT
  runtime goal, and Sayboard's host notes do not show an obvious Linux CPU win.
- Stacked projected-cache ABI: more graph-side cache rolling with no current
  runtime benefit.
