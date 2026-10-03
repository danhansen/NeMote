# Optimization Experiments

This log preserves runtime optimization experiments, including historical
ONNX/ORT work and investigations of alternatives to the current NeMo-Speech.cpp
backend.

See [sayboard-optimization-harvest.md](sayboard-optimization-harvest.md) for
the source-level Sayboard optimization inventory and harvest results.

## 2026-10-02: transcribe.cpp and Parakeet Unified — Not Adopted

The WAV-driven screen did not demonstrate a compelling benefit for dictation.
Nemotron performance was chunk-dependent: transcribe.cpp was about 11% slower
at 80 ms, close at 560 ms, and 9% faster at 1120 ms on an i5-1235U with two
threads and concurrency one. Differences within 5% were treated as noise.

Parakeet Unified Q8 did not sustain real time at 1120 ms buffering latency
(RTF 1.21 on the longer clip); at its default 2080 ms it did (RTF 0.73).
On 40 speakers / 914 words, default-setting WER was 2.19%, matching the
existing NeMo/Nemotron backend at 560 ms. Unified Q8 and F32 had identical
normalized transcripts at both tested accuracy settings. This small read-English
sample does not establish general accuracy equivalence or superiority.

Decision: retain NeMo-Speech.cpp/Nemotron; do not integrate the alternative
backend. Prototype code, build scripts and tests were removed from main.
Only the [archived report](research/transcribe-cpp-2026-10-02/transcribe-prototype-report.txt),
provenance, raw measurements and summary remain under
`docs/research/transcribe-cpp-2026-10-02/`. Commands in those artifacts are
historical records, not maintained tools.

## 2026-06-22: ORT Serialization And Linear-Pos Dequantization

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  baseline=models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56 \
  ort_extended=build/model-variants/nemotron-c56-ort-extended \
  linear_pos_fp32=build/model-variants/nemotron-c56-linear-pos-fp32 \
  --runs 3 \
  --output build/parakeet-variant-bench/onnx-ort-harvest-001.json
```

Settings:

- WAV: `build/allocation-ablation/librispeech-long.wav`
- Intra-op threads: `2`
- Flush chunks: `3`
- Runtime graph optimization flag: `all`

Results:

| Variant | Median RTF | Median real-audio RTF | Median decode seconds | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 1.028 | 1.043 | 128.970 | baseline |
| `ort_extended` | 0.883 | 0.896 | 110.745 | +14.1% decode speed |
| `linear_pos_fp32` | 0.877 | 0.890 | 110.072 | +14.7% decode speed |

Rough concatenated-reference WER, using
`build/librispeech-backend-eval/manifest.jsonl` as the source transcript for the
long WAV:

| Variant | Edits / words | WER |
| --- | ---: | ---: |
| `baseline` | 9 / 310 | 2.90% |
| `ort_extended` | 9 / 310 | 2.90% |
| `linear_pos_fp32` | 10 / 310 | 3.23% |

Observations:

- Serializing ORT's `extended` optimized encoder graph is a clean win in this
  run set. It produced the same transcript as baseline and reduced median
  decode time by about 14%.
- The targeted `linear_pos_fp32` rewrite was slightly faster than the serialized
  ORT variant, but it changed decoded text and slightly worsened the rough
  concatenated-reference WER in this run set.
- The `linear_pos_fp32` script rewrites one source `self_attn/linear_pos`
  quantized matmul. ORT expands/fuses the graph later, so the single source
  block corresponds to the repeated hot `linear_pos` kernels seen in ORT
  profiling.

Next candidates:

- Build future model directories with `--ort-optimize-final extended` in either
  `scripts/transform_nemotron_parakeet_export.py` or
  `scripts/export_nemotron_parakeet_optimized.py`; this is the clean candidate
  from this run set.
- Evaluate `linear_pos_fp32` on a larger LibriSpeech manifest before keeping it.
- Test fixed-shape/static-shape folding only after confirming it survives as a
  runtime win beyond ORT's session-time optimization.

## 2026-06-22: Fixed Streaming Shapes

Sayboard's Parakeet EOU export path specialized the runtime ONNX graph around
the actual streaming chunk/cache shapes. `scripts/build_nemotron_fixed_shape_model.py`
ports that idea to the Nemotron/Parakeet export by:

- fixing streaming input/output shapes for the current c56 model:
  `[1, 128, 65]` mel chunks, `[24, 1, 56, 1024]` channel cache,
  `[24, 1, 1024, 8]` time cache, and `[1, 1024, 7]` encoded output;
- resolving symbolic dimensions where those fixed values are known;
- replacing static `Shape` nodes with int64 initializers;
- optionally serializing ORT's final optimized graph with
  `--ort-optimize-final extended`.

Build commands:

```sh
.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56 \
  --output-dir build/model-variants/nemotron-c56-fixed-shape

.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56 \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Graph diagnostics:

| Variant | Nodes | Unresolved dims | `Shape` | `Gather` | `MatMulInteger` | `DynamicQuantizeMatMul` | `MatMulIntegerToFloat` | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ort_extended` | 2,981 | 4,660 | 111 | 211 | 0 | 195 | 24 | 626.3 MiB |
| `fixed_shape` source | 6,893 | 4,138 | 231 | n/a | n/a | n/a | n/a | 627.1 MiB |
| `fixed_shape` after ORT extended | 2,091 | 222 | 0 | 0 | 0 | 195 | 0 | 574.8 MiB |
| `fixed_shape_ort` serialized | 2,091 | 222 | 0 | 0 | 0 | 195 | 0 | 574.8 MiB |

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ort_extended=build/model-variants/nemotron-c56-ort-extended \
  fixed_shape=build/model-variants/nemotron-c56-fixed-shape \
  fixed_shape_ort=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  --runs 3 \
  --output build/parakeet-variant-bench/fixed-shape-ac-002.json
```

Settings:

- WAV: `build/allocation-ablation/librispeech-long.wav`
- Intra-op threads: `2`
- Flush chunks: `3`
- Runtime graph optimization flag: `all`
- Power metadata: AC online for all runs; battery charged from 29% to 45%;
  governor `schedutil`; turbo not disabled.

Results from this AC batch:

| Variant | Median RTF | Median real-audio RTF | Median decode seconds | Delta vs `ort_extended` |
| --- | ---: | ---: | ---: | ---: |
| `ort_extended` | 1.236 | 1.253 | 154.987 | baseline |
| `fixed_shape` | 0.889 | 0.902 | 111.493 | +28.0% real-audio RTF |
| `fixed_shape_ort` | 0.825 | 0.837 | 103.440 | +33.2% real-audio RTF |

Rough concatenated-reference WER, using the current
`build/librispeech-backend-eval/manifest.jsonl` parser:

| Variant | Edits / words | WER |
| --- | ---: | ---: |
| `ort_extended` | 10 / 313 | 3.19% |
| `fixed_shape` | 10 / 313 | 3.19% |
| `fixed_shape_ort` | 10 / 313 | 3.19% |

Observations:

- Fixed streaming shapes are a larger win than plain ORT serialization in this
  setup. They remove runtime shape plumbing that ORT could not eliminate from
  the symbolic export and allow the final optimized graph to drop from 2,981
  nodes to 2,091 nodes.
- The serialized fixed-shape graph is the best candidate artifact from this
  run set: it keeps the same transcript/WER as `ort_extended`, is smaller on
  disk, avoids session-time graph optimization work, and was fastest in the AC
  batch.
- Absolute RTF varied by machine state during the day. The benchmark harness now
  records AC, battery, governor, CPU frequency, turbo, and platform profile
  metadata so future timing JSON can be audited.

Next candidates:

- Prefer `build/model-variants/nemotron-c56-fixed-shape-ort-extended` as the
  current runtime model candidate.
- Add randomized variant order or warmup support to the benchmark harness if we
  need smaller deltas than this fixed-shape result.
- Continue looking for Sayboard optimizations beyond fixed-shape specialization
  only after this artifact is validated against a larger LibriSpeech sample.

## 2026-06-22: Fixed-Shape MatMul Dequantization Ablation

Sayboard's Parakeet EOU ablation suite tested selective dynamic-int8 MatMul/Gemm
quantization. For NeMote's already-quantized Nemotron export, the analogous
experiment is to rewrite selected static-RHS `MatMulInteger` blocks back to
float `MatMul`/`Gemm`, then serialize the final graph through ORT extended.

Applicability check on `build/model-variants/nemotron-c56-fixed-shape/encoder.onnx`:

| Candidate family | Rewritable blocks | Note |
| --- | ---: | --- |
| `linear_pos_fp32` | 1 | Shared relative-position projection block |
| `attn_proj_fp32` | 48 | Self-attention Q/K/V/output projections |
| `ffn_fp32` | 96 | Feed-forward MatMul/Gemm blocks |
| attention score/context | 0 | Dynamic-dynamic attention matmuls have no static RHS to dequantize |

Build commands:

```sh
.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-linear-pos-fp32-ort \
  --include /self_attn/linear_pos/ \
  --ort-optimize-final extended \
  --ort-optimize-threads 1

.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-attn-proj-fp32-ort \
  --include /self_attn/linear_q/ \
  --include /self_attn/linear_k/ \
  --include /self_attn/linear_v/ \
  --include /self_attn/linear_out/ \
  --ort-optimize-final extended \
  --ort-optimize-threads 1

.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --include /feed_forward \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Artifact sizes:

| Variant | Encoder size |
| --- | ---: |
| `fixed_shape_ort` | 575 MiB |
| `linear_pos_fp32` | 575 MiB |
| `attn_proj_fp32` | 719 MiB |
| `ffn_fp32` | 1.7 GiB |

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  fixed_shape_ort=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  linear_pos_fp32=build/model-variants/nemotron-c56-fixed-shape-linear-pos-fp32-ort \
  attn_proj_fp32=build/model-variants/nemotron-c56-fixed-shape-attn-proj-fp32-ort \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --runs 3 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --output build/parakeet-variant-bench/fixed-shape-dequant-001.json
```

Settings:

- WAV: `build/allocation-ablation/librispeech-long.wav`
- Intra-op threads: `2`
- Flush chunks: `3`
- Runtime graph optimization flag: `all`
- Memory guard: each run required at least 6 GiB `MemAvailable`; each worker
  subprocess had `RLIMIT_AS=10 GiB`.
- Memory metadata: start `MemAvailable=11.1 GiB`, swap used `3.1 GiB`; end
  `MemAvailable=11.6 GiB`, swap used `2.9 GiB`.

Results:

| Variant | Median RTF | Median real-audio RTF | Median decode seconds | Delta vs `fixed_shape_ort` |
| --- | ---: | ---: | ---: | ---: |
| `fixed_shape_ort` | 0.709 | 0.719 | 88.922 | baseline |
| `linear_pos_fp32` | 0.827 | 0.839 | 103.700 | -16.7% |
| `attn_proj_fp32` | 0.710 | 0.720 | 89.011 | -0.1% |
| `ffn_fp32` | 0.618 | 0.627 | 77.506 | +12.8% |

Rough concatenated-reference WER, using the current
`build/librispeech-backend-eval/manifest.jsonl` parser:

| Variant | Edits / words | WER |
| --- | ---: | ---: |
| `fixed_shape_ort` | 10 / 313 | 3.19% |
| `linear_pos_fp32` | 12 / 313 | 3.83% |
| `attn_proj_fp32` | 12 / 313 | 3.83% |
| `ffn_fp32` | 9 / 313 | 2.88% |

Graph diagnostics for `ffn_fp32`:

| Variant | Nodes | `MatMul` | `DynamicQuantizeMatMul` | `ConvInteger` | Float initializer bytes | UINT8 initializer bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_fp32` | 2,235 | 120 | 99 | 77 | 1549.1 MiB | 176.8 MiB |

Observations:

- `linear_pos_fp32` does not carry over as a win after fixed-shape folding. It
  is slower and worsens the rough WER in this run set.
- `attn_proj_fp32` is effectively runtime parity but larger and changes text,
  so it is not attractive by itself.
- `ffn_fp32` is the first selective dequantization that moves the performance
  needle after fixed-shape specialization: +12.8% real-audio RTF versus
  `fixed_shape_ort`, with a slightly better rough WER on this concatenated
  sample. The tradeoff is model size: 1.7 GiB for the encoder.
- The likely explanation is CPU-kernel economics: for these large FFN
  static-RHS matmuls, ORT's FP32 GEMM path beats dynamic activation
  quantization plus int8 matmul overhead on this machine.

Next candidates:

- Validate `ffn_fp32` on the larger LibriSpeech sampled evaluation before
  promoting it as the runtime model.
- Consider an intermediate FFN subset ablation by layer if the 1.7 GiB model
  size is too expensive.
- Keep `fixed_shape_ort` as the compact default candidate until the larger WER
  run confirms `ffn_fp32`.

### LibriSpeech Sample Validation

The existing 9-utterance LibriSpeech manifest was rerun with per-utterance
worker subprocesses and the same memory safety guard:

```sh
.venv/bin/python scripts/eval_librispeech_backends.py \
  --manifest build/librispeech-backend-eval/manifest.jsonl \
  --work-dir build/librispeech-ffn-validation/fixed-shape-ort \
  --backend parakeet \
  --parakeet-model-dir build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10

.venv/bin/python scripts/eval_librispeech_backends.py \
  --manifest build/librispeech-backend-eval/manifest.jsonl \
  --work-dir build/librispeech-ffn-validation/ffn-fp32 \
  --backend parakeet \
  --parakeet-model-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10
```

Results:

| Variant | Samples | WER | Decode RTF | Real-audio decode RTF | Wall RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_shape_ort` | 9 | 9 / 313 = 2.88% | 0.707 | 0.807 | 0.921 |
| `ffn_fp32` | 9 | 9 / 313 = 2.88% | 0.610 | 0.697 | 1.017 |

Validation observations:

- `ffn_fp32` preserved aggregate WER on this sampled LibriSpeech set and
  improved real-audio decode RTF by about 13.7%.
- Wall RTF was worse for `ffn_fp32` in this eval harness because each utterance
  starts a fresh worker process/session and the `ffn_fp32` encoder is much
  larger. The long-WAV benchmark is the better proxy for a persistent dictation
  worker.
- This moves `ffn_fp32` from "interesting" to "candidate", with model size and
  session-load behavior as the remaining tradeoffs to quantify.

### FFN Slice Ablation

To test whether the 1.7 GiB full-FFN artifact could be reduced, four 48-block
FFN variants were built from the same fixed-shape source:

- `ffn_early_fp32`: layers 0-11 feed-forward blocks.
- `ffn_late_fp32`: layers 12-23 feed-forward blocks.
- `ffn_even_fp32`: even-numbered layer feed-forward blocks.
- `ffn_odd_fp32`: odd-numbered layer feed-forward blocks.

Each slice rewrote 48 blocks, pruned 96 now-unused initializers, and produced a
1.2 GiB encoder.

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  fixed_shape_ort=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  ffn_early_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-early-fp32-ort \
  ffn_late_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-late-fp32-ort \
  ffn_even_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-even-fp32-ort \
  ffn_odd_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-odd-fp32-ort \
  --runs 3 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --output build/parakeet-variant-bench/ffn-slices-002.json
```

The benchmark harness now writes checkpoint JSON after each completed run, so
long ablation batches leave `status=partial` evidence if interrupted.

Results:

| Variant | Encoder size | Median real-audio RTF | Delta vs `fixed_shape_ort` | Rough WER |
| --- | ---: | ---: | ---: | ---: |
| `fixed_shape_ort` | 575 MiB | 0.708 | baseline | 10 / 313 = 3.19% |
| `ffn_fp32` | 1.7 GiB | 0.608 | +14.1% | 9 / 313 = 2.88% |
| `ffn_early_fp32` | 1.2 GiB | 0.655 | +7.5% | 10 / 313 = 3.19% |
| `ffn_late_fp32` | 1.2 GiB | 0.657 | +7.2% | 10 / 313 = 3.19% |
| `ffn_even_fp32` | 1.2 GiB | 0.655 | +7.5% | 11 / 313 = 3.51% |
| `ffn_odd_fp32` | 1.2 GiB | 0.676 | +4.5% | 10 / 313 = 3.19% |

Slice observations:

- Half-FFN dequantization gives roughly half the full-FFN speedup, not most of
  it. The runtime benefit appears fairly proportional to the number of FFN
  blocks moved back to FP32.
- Early and late contiguous halves are equivalent on this benchmark and preserve
  rough WER. Even is not attractive because rough WER worsened; odd is slower.
- Full `ffn_fp32` remains the best speed candidate. The half variants are
  fallback compromises only if the 1.7 GiB encoder is unacceptable.

## 2026-06-22: Conv Dequantization Ablation

Sayboard rejected conv quantization for its Parakeet EOU export. NeMote's
Nemotron export still contains quantized `ConvInteger` blocks after fixed-shape
specialization, so `scripts/dequantize_nemotron_conv_blocks.py` tests the
analogous rewrite: replace selected dynamic-quantized conv blocks with float
`Conv`, then serialize through ORT extended.

Applicability on `build/model-variants/nemotron-c56-fixed-shape/encoder.onnx`:

| Candidate family | Rewritable blocks | Encoder size |
| --- | ---: | ---: |
| `preconv_fp32` | 5 | 575 MiB |
| `allconv_fp32` | 77 | 792 MiB |

Build commands:

```sh
.venv/bin/python scripts/dequantize_nemotron_conv_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-preconv-fp32-ort \
  --include /pre_encode/conv/ \
  --ort-optimize-final extended \
  --ort-optimize-threads 1

.venv/bin/python scripts/dequantize_nemotron_conv_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-allconv-fp32-ort \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  fixed_shape_ort=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  preconv_fp32=build/model-variants/nemotron-c56-fixed-shape-preconv-fp32-ort \
  allconv_fp32=build/model-variants/nemotron-c56-fixed-shape-allconv-fp32-ort \
  --runs 3 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/conv-dequant-002.json
```

Settings:

- WAV: `build/allocation-ablation/librispeech-long.wav`
- Intra-op threads: `2`
- Flush chunks: `3`
- Runtime graph optimization flag: `all`
- Power profile: GNOME `balanced` at benchmark start and end. AC changed from
  offline to online during the run, but the profile stayed `balanced`.

Results:

| Variant | Encoder size | Median real-audio RTF | Delta vs `fixed_shape_ort` | Median decode seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_shape_ort` | 575 MiB | 0.703 | baseline | 86.877 | 10 / 313 = 3.19% |
| `preconv_fp32` | 575 MiB | 0.698 | +0.7% | 86.263 | 12 / 313 = 3.83% |
| `allconv_fp32` | 792 MiB | 0.637 | +9.4% | 78.753 | 12 / 313 = 3.83% |

Conv observations:

- Dequantizing only the five pre-encoder conv blocks is essentially runtime
  parity and worsens rough WER, so it is not attractive.
- Dequantizing all conv blocks is a real speed win on this benchmark, but it
  also worsens rough WER and previously changed the short smoke sample from
  "gold" to "code". Treat it as rejected unless a larger WER run shows that the
  regression is sample noise.
- The benchmark harness now records GNOME power-profiles-daemon state and can
  set a profile with `--set-power-profile`, which makes future runs more
  comparable than AC status alone. This daemon rejects `HoldProfile` for
  `balanced`, so the harness records that and falls back to set-and-verify for
  balanced-profile runs.

## 2026-06-22: FFN FP32 Thread-Count Sweep

The current best speed candidate, `ffn_fp32`, was benchmarked with ORT intra-op
threads set to 1, 2, 3, and 4 on the Ivy Bridge i5-3320M machine
(2 physical cores / 4 hardware threads). Each setting used three runs of the
long WAV benchmark under GNOME `balanced`.

Commands followed this pattern:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --runs 3 \
  --num-threads <N> \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-thread-sweep-t<N>.json
```

Results:

| ORT threads | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.879 | 0.866 | 108.632 | 112.647 |
| 2 | 0.606 | 0.597 | 74.930 | 78.985 |
| 3 | 0.860 | 0.848 | 106.333 | 110.653 |
| 4 | 0.820 | 0.808 | 101.338 | 105.483 |

Thread observations:

- The existing default, `--num-threads 2`, is the clear winner and should stay
  the default for this CPU.
- One thread underutilizes the two physical cores.
- Three and four threads are much worse than two threads, likely because this
  workload does not benefit from SMT oversubscription on the i5-3320M and pays
  extra scheduling/cache overhead.

## 2026-06-22: Live Audio Buffer Reuse

Sayboard's native bridge work emphasized avoiding avoidable allocation in the
hot recognition path. NeMote's CPAL live-input path still allocated a fresh
`Vec<f32>` in every audio callback before sending samples to the recognition
thread.

Change:

- Added a bounded audio buffer pool alongside the bounded audio queue in
  `wordpipe-parakeet-worker`.
- The CPAL callback now reuses a returned `Vec<f32>` when one is available,
  reserving only if the callback delivers a larger buffer than previously seen.
- The recognition thread recycles each buffer after extending the pending audio
  accumulator.

Validation:

```sh
cargo fmt
cargo check -p wordpipe-parakeet-worker
```

Notes:

- This targets live dictation stability and callback overhead, not long-WAV
  benchmark RTF. The benchmark path reads from a file and already reuses its
  chunk buffer.
- The bounded queue/drop behavior is unchanged: if the recognition thread falls
  behind, incoming callback buffers are still dropped and counted.

## 2026-06-22: ORT Session-Option Ablation

After thread count was settled at two intra-op threads, the worker exposed
several ORT session knobs for controlled benchmarking:

- `--ort-memory-pattern auto|enable|disable`
- `--ort-parallel-execution`
- `--ort-cpu-arena auto|enable|disable`

`auto` preserves the previous behavior and lets ORT choose its default. The
non-auto settings explicitly call the corresponding ORT session/CPU EP APIs.

Benchmark commands followed this pattern:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --runs 3 \
  --num-threads 2 \
  <ORT option under test> \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-ort-options-<variant>.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds |
| --- | ---: | ---: | ---: | ---: |
| default / auto | 0.606 | 0.597 | 74.890 | 78.905 |
| `--ort-memory-pattern disable` | 0.630 | 0.621 | 77.944 | 81.948 |
| `--ort-memory-pattern enable` | 0.650 | 0.641 | 80.417 | 84.624 |
| `--ort-parallel-execution` | 0.644 | 0.635 | 79.627 | 83.893 |
| `--ort-cpu-arena disable` | 0.643 | 0.634 | 79.524 | 83.757 |
| `--ort-cpu-arena enable` | 0.610 | 0.602 | 75.477 | 79.548 |

Session-option observations:

- The existing ORT defaults are best on this machine.
- Explicit CPU arena enable is close to default but still slightly slower;
  forcing CPU EP registration is not useful here.
- Disabling the CPU arena is a throughput loss, even though it might reduce
  allocator memory retention in other conditions.
- Parallel graph execution and explicit memory-pattern overrides are both
  slower for this fixed-shape streaming graph.

## 2026-06-22: Current-Best ORT Profile

The current best speed candidate, `ffn_fp32`, was profiled directly through
ONNX Runtime with fixed synthetic streaming inputs. This isolates the encoder
graph from file I/O, decoder accumulation, and worker JSON output.

Command:

```sh
.venv/bin/python scripts/profile_nemotron_ort.py \
  --model-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --threads 2 \
  --graph-optimization all \
  --warmup 3 \
  --iterations 30 \
  --output-dir build/ort-profile/ffn-fp32-solo
```

Summary:

| Metric | Value |
| --- | ---: |
| Mean encoder run | 302.183 ms |
| Min encoder run | 290.339 ms |
| Max encoder run | 322.502 ms |
| Profiled node time | 9,601.885 ms |
| Provider | CPUExecutionProvider |

Top operation families by profiled node time:

| Op | Time |
| --- | ---: |
| `MatMul` | 2,309.615 ms |
| `FusedMatMul` | 2,187.441 ms |
| `ConvInteger` | 2,146.987 ms |
| `DynamicQuantizeMatMul` | 1,476.772 ms |
| `LayerNormalization` | 189.985 ms |
| `DynamicQuantizeLinear` | 161.063 ms |
| `Mul` | 155.017 ms |
| `Transpose` | 143.702 ms |

Profile observations:

- Full FFN dequantization did what we expected: the huge
  `DynamicQuantizeMatMul` block from the compact graph is mostly gone, replaced
  by FP32 `MatMul`/`FusedMatMul` work.
- The remaining major costs are now broad and structural: FP32 FFN matmuls,
  quantized convs, and the still-quantized non-FFN matmuls.
- All-conv FP32 was faster in the long-WAV benchmark but regressed rough WER, so
  the `ConvInteger` block is not a clean optimization target unless a larger
  validation run clears it.
- The remaining `DynamicQuantizeMatMul` cost is led by non-FFN nodes such as
  `/encoder/pre_encode/out/MatMul_quant`; attention-projection dequantization
  already tested poorly on rough WER.

### FFN Plus Pre-Encoder Output Dequantization

The ORT profile above showed `/encoder/pre_encode/out/MatMul_quant` as the
largest remaining non-FFN `DynamicQuantizeMatMul` site. The first attempt to
rewrite it from `build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort`
rewrote zero blocks because the source had already been serialized through ORT,
where the original `MatMulInteger` pattern was fused into
`DynamicQuantizeMatMul`. The valid procedure is to apply all selected
dequantization before the final ORT optimization pass.

`scripts/dequantize_nemotron_matmul_blocks.py` was adjusted so biased matmuls
whose input rank is unknown or not statically 2-D fall back to `MatMul` plus the
original `Add`, instead of emitting an invalid `Gemm`.

Build command:

```sh
.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-preout-fp32-ort-v2 \
  --include /feed_forward \
  --include /pre_encode/out/ \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Result: `rewritten_blocks=97`, `prunedInitializers=194`, encoder size
`1739.6 MiB`.

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  ffn_preout_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-preout-fp32-ort-v2 \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-preout-001.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_fp32` | 0.685 | 0.675 | 84.710 | 89.447 | 9 / 313 = 2.88% |
| `ffn_preout_fp32` | 0.654 | 0.645 | 80.849 | 85.735 | 10 / 313 = 3.19% |

Observations:

- Adding the pre-encoder output projection to the FFN FP32 rewrite improved
  median real-audio RTF by about 4.5% relative to the same-session `ffn_fp32`
  baseline.
- The transcript changed from "had seemed" to "had seem", raising rough WER by
  one edit on the concatenated sample. Treat this as speed-positive but
  accuracy-suspicious until a broader validation run decides whether it is
  acceptable.
- This keeps `ffn_fp32` as the conservative current-best candidate, while
  `ffn_preout_fp32` is a follow-up WER-validation candidate.

## 2026-06-22: Nemotron Streaming Paper Leads

The paper at https://arxiv.org/html/2604.14493v1 is directly relevant because
it describes ONNX Runtime optimization for Nemotron Speech Streaming on
resource-constrained CPU devices.

Applicable learnings:

- Their selected streaming configuration is the same `(7, 10, 7)` chunk/history
  setting used by NeMote's c56 export: 560 ms chunks with 5.6 s history. This
  supports keeping the projected-cache rewrite aligned with that left context
  rather than shrinking cache length only for speed.
- Their quantization boundary matches our empirical direction: the encoder is
  the optimization target, while decoder/joiner stay FP32 because they are
  smaller and repeatedly invoked in the RNNT loop.
- Their strongest ONNX lead is weight-only block quantization using ORT
  `MatMulNBits`, with activations kept FP32. That is different from our current
  dynamic activation quantization path (`DynamicQuantizeLinear` plus
  `MatMulInteger`, often fused to `DynamicQuantizeMatMul`).
- Their `ConvInteger`/`MatMulInteger` result is a useful caution. It improves
  throughput but worsens WER, matching NeMote's all-conv FP32 ablation in the
  other direction: conv/integer arithmetic changes can be speed-positive while
  still being accuracy-risky.

Reported paper results for Nemotron Speech Streaming:

| Variant | Size | WER | RTFx |
| --- | ---: | ---: | ---: |
| FP32 ONNX | 2.47 GB | 8.03 | 6.73 |
| Int8 k-quant | 1.28 GB | 8.01 | 7.25 |
| Int4 k-quant | 0.67 GB | 8.20 | 7.20 |
| Int4 plus `ConvInteger`/`MatMulInteger` | 0.64 GB | 10.14 | 8.74 |

Next paper-derived experiments:

- Prototype an ORT `MatMulNBits` weight-only block-quantized encoder from the
  fixed-shape source, starting with int8 k-quant before int4.
- Compare against `ffn_fp32` and compact `fixed_shape_ort` with the same
  long-WAV 3-run median harness and the larger LibriSpeech sampled validation.
- Preserve decoder/joiner FP32 in that experiment.

### Int8 K-Quant MatMulNBits First Pass

`scripts/quantize_nemotron_matmul_nbits.py` wraps ONNX Runtime's lower-level
weight-only quantization implementation so the `MatMulNBits` experiment is
repeatable. The first pass targeted the current `ffn_fp32` candidate rather
than a fully FP32 pre-fusion encoder; this is intentionally narrow and should
not be read as a complete reproduction of the paper export.

Dry-run:

```sh
.venv/bin/python scripts/quantize_nemotron_matmul_nbits.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-dryrun \
  --bits 8 \
  --block-size 32 \
  --algorithm k_quant \
  --dry-run
```

Result: 72 static-RHS `MatMul` nodes selected.

Build:

```sh
.venv/bin/python scripts/quantize_nemotron_matmul_nbits.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-k32 \
  --bits 8 \
  --block-size 32 \
  --algorithm k_quant
```

Build output:

| Artifact | Value |
| --- | ---: |
| Selected `MatMul` nodes | 72 |
| Saved `MatMulNBits` nodes | 48 |
| Encoder size | 1.2 GiB |
| Baseline `ffn_fp32` encoder size | 1.7 GiB |

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  ffn_nbits_int8_k32=build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-k32 \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-nbits-int8-k32-001.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_fp32` | 0.623 | 0.614 | 76.992 | 81.073 | 9 / 313 = 2.88% |
| `ffn_nbits_int8_k32` | 1.580 | 1.558 | 195.385 | 198.132 | 10 / 313 = 3.19% |

Observations:

- ORT 1.27's CPUExecutionProvider can execute `MatMulNBits` from the Rust
  worker, so the contrib-op path is functionally viable.
- This first pass is a throughput loss on the Ivy Bridge i5-3320M despite
  reducing encoder size by about 30%. It is also one edit worse on the rough
  concatenated sample.
- The likely reason is that this path quantized a partially optimized mixed
  graph: 99 `DynamicQuantizeMatMul`, 48 `FusedMatMul`, and 72 ordinary
  `MatMul` nodes remained alongside 48 `MatMulNBits` nodes. It did not recreate
  the paper's encoder-wide FP32-to-weight-only export.
- Next `MatMulNBits` attempts, if any, should start earlier in the pipeline:
  export/fold/fix shapes while preserving ordinary static-RHS `MatMul` nodes,
  then apply weight-only quantization before ORT fuses the graph.

### Pre-Fusion FFN MatMulNBits

The post-ORT `MatMulNBits` pass above only produced 48 saved `MatMulNBits`
nodes because much of the graph had already been fused. To test whether
pre-fusion placement changes the result, a non-ORT-serialized FFN-FP32 source
was built first:

```sh
.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-preort \
  --include /feed_forward
```

Result: `rewritten_blocks=96`, `prunedInitializers=192`, `ortFinal=None`,
encoder size `1779.0 MiB`. A dry-run from this source selected 96 static-RHS
FFN `MatMul` nodes for weight-only quantization.

The low-level k-quant path completed quantization but produced an invalid graph
for this pre-fusion source: ONNX checker reported a missing producer for
`/encoder/pre_encode/conv/conv.0/Conv_output_0_bias_reshape_output`. This
appears to be a limitation or bug in the neural-compressor-derived ORT helper
on this graph, so the wrapper was extended to use ORT's newer
`MatMulNBitsQuantizer` default path as `--algorithm default`.

Build:

```sh
.venv/bin/python scripts/quantize_nemotron_matmul_nbits.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-preort \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-default-k32-preort \
  --bits 8 \
  --block-size 32 \
  --algorithm default
```

Build output:

| Artifact | Value |
| --- | ---: |
| Selected `MatMul` nodes | 96 |
| Saved `MatMulNBits` nodes | 96 |
| Remaining `DynamicQuantizeMatMul` nodes | 48 |
| Remaining ordinary `MatMul` nodes | 72 |
| Encoder proto | 1.9 MiB |
| Encoder external data | 685.2 MiB |

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  ffn_nbits_int8_default_preort=build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-default-k32-preort \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-nbits-int8-default-preort-001.json
```

The benchmark produced three baseline `ffn_fp32` runs but the first candidate
run timed out after the 300 second per-run limit. The partial result file is
`build/parakeet-variant-bench/ffn-nbits-int8-default-preort-001.json`.

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Result |
| --- | ---: | ---: | ---: | --- |
| `ffn_fp32` | 0.616 | 0.607 | 76.183 | baseline median |
| `ffn_nbits_int8_default_preort` | n/a | n/a | >300 | timed out on run 1 |

Observations:

- Applying `MatMulNBits` before ORT fusion does create the intended 96 FFN
  weight-only nodes and reduces the model size substantially.
- Throughput is still decisively worse on this CPU. The candidate failed to
  finish a single long-WAV run within 300 seconds, versus a 76.183 second
  baseline median decode.
- This rejects the FFN-only `MatMulNBits` path for the current Ivy Bridge CPU
  and ORT 1.27 CPU EP. The paper's full encoder-wide export may still behave
  differently on newer CPUs or with a different ORT build, but the practical
  NeMote path remains `ffn_fp32`.

## 2026-06-22: FP32 NeMo Export With Projected Cache

The projected-cache rewrite originally targeted the quantized sherpa-style
encoder graph. To test whether the same cache rewrite was applicable earlier in
the NeMo export pipeline, `scripts/rewrite_nemotron_projected_kv_cache.py` was
extended to support native FP32 `MatMul` K/V projection nodes and external-data
serialization. `scripts/build_nemotron_fixed_shape_model.py` was also updated
to save/check large external-data models by path so FP32 graphs do not hit
ONNX's in-memory >2 GB checker path.

Build from the existing interrupted FP32 export artifacts:

```sh
.venv/bin/python scripts/transform_nemotron_parakeet_export.py \
  build/model-variants/nemotron-fp32-projected \
  --no-quantize \
  --projected-cache \
  --keep-fp32
```

Fixed-shape and ORT-optimized build:

```sh
.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir build/model-variants/nemotron-fp32-projected \
  --output-dir build/model-variants/nemotron-fp32-projected-fixed-shape \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Resulting fixed-shape artifact:

| Artifact | Size |
| --- | ---: |
| `encoder.onnx` | 12.2 MiB |
| `encoder.onnx.data` | 2340.7 MiB |
| `decoder_joint.onnx` | 93.1 MiB |

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  fp32_projected=build/model-variants/nemotron-fp32-projected-fixed-shape \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/fp32-projected-001.json
```

The system reported `BAT0/status=Discharging` for this run even though GNOME's
power profile was pinned to `balanced`, so compare it as a same-run A/B result
rather than a clean AC-powered absolute benchmark.

WER scoring:

```sh
.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/fp32-projected-001.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_fp32` | 0.666 | 0.656 | 82.289 | 87.045 | 9 / 313 = 2.88% |
| `fp32_projected` | 0.590 | 0.582 | 73.013 | 76.506 | 11 / 313 = 3.51% |

Observations:

- The FP32 NeMo export plus projected cache is throughput-positive in this
  A/B: median real-audio RTF improved by about 11.4% versus `ffn_fp32`.
- It fails the WER gate. The transcript changed consistently across all three
  runs, including "had seemed" -> "had seem", raising rough WER from 9/313 to
  11/313.
- Do not promote this as the default runtime candidate. The useful harvest is
  the hardened projected-cache/export tooling and the reusable
  `scripts/score_benchmark_wer.py` WER check.

Later note: the broader 985-word RTF/WER sweep below supersedes this narrow
313-word gate and promotes the full FP32 projected-cache profile as the current
high-performance default.

### Raw-Cache Control

To isolate whether the WER regression came from projected-cache rewriting or
from the FP32 NeMo export path itself, the same consolidated FP32 encoder was
also transformed without projected cache and then built with the same
fixed-shape/ORT-extended path.

Build:

```sh
.venv/bin/python scripts/transform_nemotron_parakeet_export.py \
  build/model-variants/nemotron-fp32-rawcache \
  --no-quantize \
  --no-projected-cache \
  --keep-fp32

.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir build/model-variants/nemotron-fp32-rawcache \
  --output-dir build/model-variants/nemotron-fp32-rawcache-fixed-shape \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  fp32_rawcache=build/model-variants/nemotron-fp32-rawcache-fixed-shape \
  fp32_projected=build/model-variants/nemotron-fp32-projected-fixed-shape \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/fp32-rawcache-isolation-001.json
```

WER scoring:

```sh
.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/fp32-rawcache-isolation-001.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp32_rawcache` | 0.687 | 0.677 | 84.911 | 88.036 | 11 / 313 = 3.51% |
| `fp32_projected` | 0.516 | 0.509 | 63.825 | 67.981 | 11 / 313 = 3.51% |

The system again reported battery discharging (`25%` -> `18%`) with GNOME
profile `balanced`, so use this as an in-run A/B only.

Conclusion:

- The WER regression is not caused by the projected-cache rewrite. Raw-cache
  FP32 and projected-cache FP32 produce the same transcript and the same rough
  WER.
- Projected cache is doing what we expected on this export: it removes the cost
  of repeatedly projecting the full raw K/V cache, improving median real-audio
  RTF by about 24.9% versus raw-cache FP32.
- The remaining accuracy question is why this FP32 NeMo export path differs
  from the sherpa-derived `ffn_fp32` candidate (`11/313` vs `9/313` on this
  rough sample).

### FP32 Decoder Hybrid Isolation

The `11/313` FP32 projected result contains two additional countable errors
relative to current best:

- `seemed` -> `seem`
- `marvelous` -> `marvellous`

Two hardlink-only hybrid packages isolated where those errors come from:

| Hybrid | Encoder | Decoder/joint | Rough WER | Interpretation |
| --- | --- | --- | ---: | --- |
| `fp32_projected_quant_decoder` | FP32 projected encoder | current quantized decoder | 10 / 313 = 3.19% | Quantized decoder fixes `marvellous`; `seem` remains encoder/export-side. |
| `current_encoder_fp32_decoder` | current-best encoder | FP32 decoder | 10 / 313 = 3.19% | Current encoder preserves `seemed`; FP32 decoder introduces only `marvellous`. |

The inverse hybrid also found a practical speed candidate. Same-file 3x A/B:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  --wav build/allocation-ablation/librispeech-long.wav \
  --runs 3 \
  --num-threads 2 \
  --flush-chunks 3 \
  --graph-optimization all \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --output build/parakeet-variant-bench/current-encoder-fp32-decoder-ab-001.json \
  baseline=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  current_encoder_fp32_decoder=build/model-variants/nemotron-current-encoder-fp32-decoder-hybrid
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 0.606 | 0.598 | 74.961 | 9 / 313 = 2.88% |
| `current_encoder_fp32_decoder` | 0.573 | 0.565 | 70.820 | 10 / 313 = 3.19% |

This is about a 5.4% median real-audio RTF improvement in the same run set,
with the strict WER delta isolated to one spelling variant. The export tooling
now exposes this as `--fp32-decoder` on
`scripts/transform_nemotron_parakeet_export.py`,
`scripts/export_nemotron_parakeet_optimized.py`, and the top-level
`scripts/build_nemotron_wordpipe_model.py` wrapper.

### Token-Level FP32 Divergence Diagnostics

`wordpipe-parakeet-worker --trace-token-decisions` now emits one
`token_decision` JSON event for each nonblank RNNT token decision. Each event
records chunk/frame/symbol position, input token, selected token, blank logit,
selected-vs-blank margin, and top-k logits. The helper script
`scripts/compare_token_traces.py` compares two JSONL traces and can print
windows around visible-text substrings.

Trace generation:

```sh
env ORT_DYLIB_PATH=.venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.27.0 \
  target/release/wordpipe-parakeet-worker \
  --model-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --wav build/allocation-ablation/librispeech-long.wav \
  --num-threads 2 \
  --flush-chunks 3 \
  --graph-optimization all \
  --trace-token-decisions \
  > build/token-trace/current-best.jsonl
```

FP32 decoder spelling flip:

```sh
scripts/compare_token_traces.py \
  build/token-trace/current-best.jsonl \
  build/token-trace/current-encoder-fp32-decoder.jsonl \
  --left-label current_best \
  --right-label fp32_decoder \
  --needle marvelous \
  --needle marvellous \
  --window 4
```

The models agree through token `mar`, then diverge:

| Model | Token path | Local top-k evidence |
| --- | --- | --- |
| Current best quantized decoder | `mar` + `ve` + `lo` + `us` | after `mar`, `ve=-42.350` vs `v=-42.683` |
| FP32 decoder | `mar` + `v` + `ell` + `ou` + `s` | after `mar`, `v=-42.398` vs `ve=-43.364` |

So this is not a large semantic failure. It is an RNNT tokenization-path flip
between two valid spelling variants.

FP32 encoder/export `seemed` flip:

```sh
scripts/compare_token_traces.py \
  build/token-trace/current-best.jsonl \
  build/token-trace/fp32-projected-quant-decoder.jsonl \
  --left-label current_best \
  --right-label fp32_encoder_quant_decoder \
  --needle seemed \
  --needle seem \
  --window 5
```

Here the decoder is held constant as the current quantized decoder. The
decision after token `see` is very close:

| Model | Token path | Local top-k evidence |
| --- | --- | --- |
| Current-best encoder | `see` + `me` + `d` | `me=-25.190` vs `m=-25.328` |
| FP32-export encoder | `see` + `m` | `m=-26.161` vs `me=-26.198` |

The `me`/`m` separation is only `0.138` logit in current best and `0.037` logit
in the FP32-export path. Because RNNT decoding feeds each emitted token back
into the prediction network, that tiny local flip changes the subsequent state:
the current-best path emits `d`, while the FP32-export path moves directly to
` t`.

Interpretation: FP32 itself is not inherently less accurate here. The tested
FP32 paths alter graph/export/runtime numerics enough to cross borderline RNNT
decision boundaries. Some dynamic quantization noise happens to bias this small
sample toward the LibriSpeech reference spelling/inflection.

### Broader High-Performance RTF/WER Sweep

The narrow 313-word sample made the full FP32 projected-cache export look like
an accuracy regression. To check whether that held up, a broader concatenated
LibriSpeech sample was built and scored with RTF and WER from the same benchmark
JSON.

Build the broader long WAV and matching manifest:

```sh
scripts/build_librispeech_long_wav.py \
  --work-dir build/librispeech-highperf-validation \
  --count 40 \
  --candidate-count 1000 \
  --max-audio-sec 420 \
  --seed 29
```

Dataset summary:

| Samples | Audio seconds | Reference words |
| ---: | ---: | ---: |
| 40 | 374.190 | 985 |

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  --wav build/librispeech-highperf-validation/librispeech-long.wav \
  --runs 1 \
  --num-threads 2 \
  --flush-chunks 3 \
  --graph-optimization all \
  --timeout-seconds 900 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 12 \
  --output build/parakeet-variant-bench/highperf-broad-wer-rtf-001.json \
  baseline=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  fp32_projected=build/model-variants/nemotron-fp32-projected-fixed-shape \
  fp32_decoder=build/model-variants/nemotron-current-encoder-fp32-decoder-hybrid
```

Score WER and RTF together from the same run rows:

```sh
.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/highperf-broad-wer-rtf-001.json \
  --manifest build/librispeech-highperf-validation/manifest.jsonl
```

Results:

| Variant | Real-audio RTF | RTF | Decode seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 0.600 | 0.596 | 224.429 | 27 / 985 = 2.74% |
| `fp32_projected` | 0.500 | 0.498 | 187.273 | 25 / 985 = 2.54% |
| `fp32_decoder` | 0.567 | 0.563 | 212.035 | 28 / 985 = 2.84% |

Edit-diff summary versus `baseline`:

- `fp32_projected` fixed 8 baseline edits and introduced 6 new edits, for a net
  improvement of 2 strict WER edits. Most changed edits are spelling/name or
  tokenization variants; one clear new bad regression was `secured` -> `sered`.
- `fp32_decoder` fixed 5 baseline edits and introduced 6 new edits, for one
  additional strict WER edit.

Interpretation:

- On this broader same-run sample, full FP32 projected-cache is both faster and
  slightly better under the rough strict WER scorer. Its real-audio RTF is 0.500
  versus the baseline's 0.600, a 16.7% lower RTF and 37.156 fewer decode seconds
  over 374.190 seconds of audio.
- The earlier 313-word result was too small to treat the FP32 projected-cache
  export as an accuracy failure. The token-level divergences are still real, but
  on a broader sample they are not consistently negative.
- `fp32_decoder` remains a modest-speed experimental option. It improved
  real-audio RTF by 5.5% versus baseline here but was slightly worse on WER.
- This is one long run per variant. The transcripts are deterministic for a
  fixed graph and input, but if absolute timing confidence becomes important,
  rerun this same broader benchmark with `--runs 3` and compare medians.

Decision:

- Promote `fp32_projected` as the current high-performance build profile. It is
  the cleanest of the performance-competitive options because it avoids dynamic
  quantization, selective FFN dequantization, and hybrid decoder packaging. The
  remaining nontrivial graph change is projected-cache rewriting, which directly
  matches the runtime's streaming cache contract.
- Keep `ffn_fp32` as the smaller mixed-int8/FP32 fallback, not the default.

### Model Load Timing And Hotkey Latency

For dictation UX, the relevant latency is keyboard shortcut press to first heard
word. Cold model construction must not be on that path. The Parakeet worker now
emits `model_loaded` with `load_seconds`, and the benchmark harness records that
field separately from decode RTF.

Short-WAV fresh-process load benchmark:

```sh
scripts/benchmark_parakeet_variant.py \
  --wav build/librispeech-backend-eval-smoke/wavs/1089-134686-0019.wav \
  --runs 3 \
  --num-threads 2 \
  --flush-chunks 0 \
  --graph-optimization all \
  --timeout-seconds 120 \
  --min-mem-available-gb 4 \
  --child-memory-limit-gb 12 \
  --output build/parakeet-variant-bench/load-time-fp32-projected-001.json \
  fp32_projected=build/model-variants/nemotron-fp32-projected-fixed-shape \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort
```

Results:

| Variant | Load seconds | Median load seconds |
| --- | --- | ---: |
| `fp32_projected` | 3.662, 5.053, 4.747 | 4.747 |
| `ffn_fp32` | 5.630, 4.634, 4.145 | 4.634 |

Interpretation:

- Sub-second cold process/model load is not realistic for either top model on
  this machine. The smaller mixed model did not materially improve load time.
- The Parakeet worker now preloads the model before emitting `ready`; hotkey
  `start` resets the already-loaded Nemotron instance instead of constructing a
  new ORT session. That moves the 4-5 second load cost to daemon startup.
- After `ready`, shortcut-to-first-word is bounded by audio stream startup plus
  the model's streaming cadence. With 560 ms chunks and roughly 0.28 seconds of
  decode per chunk in the broad `fp32_projected` run, the theoretical first
  token latency is near the one-second target if the speaker starts promptly and
  the mic stream opens quickly.
- A live `start` -> `listening` measurement was attempted, but the current
  default input device rejected the worker's requested 16 kHz config in this
  environment. Re-run this after selecting a working input device or teaching
  the worker to negotiate/resample from the device's native rate.

### Compact Model Broad Comparison

The compact fixed-shape candidate keeps the sherpa-derived quantized/projected
encoder and only applies fixed streaming shapes plus serialized ORT
optimization. It is the best small model recipe:

```sh
.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56 \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Broad interleaved benchmark:

```sh
scripts/benchmark_parakeet_variant.py \
  --interleave \
  --wav build/librispeech-highperf-validation/librispeech-long.wav \
  --runs 3 \
  --num-threads 2 \
  --flush-chunks 3 \
  --graph-optimization all \
  --timeout-seconds 900 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 12 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/compact-vs-top-broad-interleaved-001.json \
  fp32_projected=build/model-variants/nemotron-fp32-projected-fixed-shape \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  compact_fixed_shape=build/model-variants/nemotron-c56-fixed-shape-ort-extended
```

Power state: AC online throughout, battery charging/not charging, GNOME profile
`balanced`.

Score:

```sh
.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/compact-vs-top-broad-interleaved-001.json \
  --manifest build/librispeech-highperf-validation/manifest.jsonl
```

Results:

| Variant | Size | Median load seconds | Median real-audio RTF | Median decode seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp32_projected` | 2.4 GiB | 2.717 | 0.501 | 187.533 | 25 / 985 = 2.54% |
| `ffn_fp32` | 1.8 GiB | 3.728 | 0.603 | 225.780 | 27 / 985 = 2.74% |
| `compact_fixed_shape` | 575 MiB | 1.101 | 0.701 | 262.237 | 29 / 985 = 2.94% |

Interpretation:

- `compact_fixed_shape` is the clear small-model choice. It is about one
  quarter the size of `fp32_projected` and loads about 1.6 seconds faster.
- The compact model pays for that footprint: about 40% higher real-audio RTF
  than `fp32_projected` and four additional strict WER edits on this 985-word
  sample.
- `ffn_fp32` is not a compelling middle ground here. It is much larger than the
  compact model, loads slower than both alternatives, and is slower/less
  accurate than `fp32_projected`.
- Keep `fp32_projected` as the default quality/speed profile, and keep
  `compact_fixed_shape` as the documented small-footprint fallback.

## 2026-06-23: ORT Format Load-Time Experiment

Sayboard's model startup work suggested two distinct approaches:

1. save an optimized ONNX graph after ORT's offline graph optimizations;
2. convert the model to native ORT format and load it with
   `session.use_ort_model_bytes_directly`.

NeMote now supports both paths experimentally. The Rust worker accepts
`--ort-optimized-model-cache-dir` for the first approach, but this is not
enabled by default because the smoke results were mixed:

| Variant | No-cache median load | Optimized-ONNX cache median load | Notes |
| --- | ---: | ---: | --- |
| `fp32_projected` | 2.781 | 3.235 | External-data cache works after hard-linking `encoder.onnx.data`, but load time is noisy and not improved. |
| `ffn_fp32` | 3.845 | 3.875 | First cache creation was expensive; steady-state was no better than normal ONNX. |
| `compact_fixed_shape` | 1.135 | 1.127 | Small improvement, but not enough to justify default multi-file cache artifacts. |

The native ORT-format path was more useful for the compact model. Conversion:

```sh
.venv-nemo-export/bin/python scripts/convert_nemotron_to_ort_format.py \
  build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  build/model-variants/nemotron-c56-fixed-shape-ort-extended-format \
  --force \
  --optimization-level all
```

Benchmark:

```sh
PARAKEET_LOAD_TRACE=1 scripts/benchmark_parakeet_variant.py \
  --interleave \
  --wav build/librispeech-backend-eval-smoke/wavs/1089-134686-0019.wav \
  --runs 5 \
  --num-threads 2 \
  --flush-chunks 0 \
  --timeout-seconds 120 \
  --min-mem-available-gb 5 \
  --child-memory-limit-gb 8 \
  --output build/parakeet-variant-bench/load-ort-format-compact-smoke-001.json \
  compact_onnx=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  compact_ort_format=build/model-variants/nemotron-c56-fixed-shape-ort-extended-format
```

Results:

| Variant | Size | Median load seconds | Median real-audio RTF | Median decode seconds |
| --- | ---: | ---: | ---: | ---: |
| `compact_onnx` | 599 MiB | 1.154 | 0.749 | 10.413 |
| `compact_ort_format` | 599 MiB | 0.461 | 0.774 | 10.754 |

The transcript was identical across all five runs. Trace output confirmed that
the `.ort` model uses the direct ORT-format path for both encoder and decoder.

Interpretation:

- ORT format gets the compact model comfortably below the one-second model-load
  target. This is the best current answer for shortcut-to-first-word startup
  when a resident preloaded daemon is not enough.
- The ORT-format files are about the same size as the compact ONNX files; this
  is a load-time/deserialization optimization, not a storage optimization.
- The FP32 projected encoder conversion crashed with exit 139 on the 16 GB test
  machine after swap pressure rose. Keep ORT-format conversion optional and use
  it first for the compact model.
- `session.use_ort_model_bytes_directly` avoids ORT copying the supplied model
  bytes internally. A separate mmap-backed buffer may still be useful later to
  avoid reading the whole file into heap memory first, but it is not required
  for the ORT-side zero-copy initializer path.

### OnnxSlim Before ORT Format

The FP32 projected ORT-format conversion failed under memory pressure, so the
next experiment tested whether `onnxslim` can simplify the ONNX graph before
any ORT-format conversion. The goal was explicitly to validate graph and runtime
behavior first, not to proceed straight to `.ort`.

Slim command:

```sh
.venv-nemo-export/bin/python scripts/slim_nemotron_onnx.py \
  build/model-variants/nemotron-fp32-projected-fixed-shape \
  build/model-variants/nemotron-fp32-projected-slim-onnx \
  --force \
  --size-threshold 1048576
```

Notes:

- `scripts/slim_nemotron_onnx.py` stages private copies of ONNX external-data
  files before invoking `onnxslim`. This avoids ONNX's hard-link safety guard,
  which rejected the source `encoder.onnx.data` after earlier cache experiments
  created additional hard links.
- Defaults are conservative for large external-data models: no shape inference,
  external-data output, and a 1 MiB constant-folding threshold.

Graph diagnostics:

| Component | Nodes | Initializers | Key op changes |
| --- | ---: | ---: | --- |
| Encoder original | 1735 | 888 | Baseline |
| Encoder slim | 1703 | 691 | `Pad` 51 -> 24, `Expand` 13 -> 8 |
| Decoder original | 42 | 13 | Baseline |
| Decoder slim | 26 | 16 | `Constant` 16 -> 0 |

Smoke benchmark:

```sh
PARAKEET_LOAD_TRACE=1 scripts/benchmark_parakeet_variant.py \
  --interleave \
  --wav build/librispeech-backend-eval-smoke/wavs/1089-134686-0019.wav \
  --runs 5 \
  --num-threads 2 \
  --flush-chunks 0 \
  --timeout-seconds 120 \
  --min-mem-available-gb 5 \
  --child-memory-limit-gb 12 \
  --output build/parakeet-variant-bench/fp32-projected-slim-onnx-smoke-001.json \
  fp32_projected=build/model-variants/nemotron-fp32-projected-fixed-shape \
  fp32_projected_slim=build/model-variants/nemotron-fp32-projected-slim-onnx
```

| Variant | Median load seconds | Median real-audio RTF | Median decode seconds | Transcript |
| --- | ---: | ---: | ---: | --- |
| `fp32_projected` | 2.865 | 0.570 | 7.926 | Same |
| `fp32_projected_slim` | 2.895 | 0.555 | 7.706 | Same |

Broad slim-only check, compared against the recorded baseline from
`compact-vs-top-broad-interleaved-001.json` instead of rerunning the unchanged
baseline:

```sh
PARAKEET_LOAD_TRACE=1 scripts/benchmark_parakeet_variant.py \
  --wav build/librispeech-highperf-validation/librispeech-long.wav \
  --runs 1 \
  --num-threads 2 \
  --flush-chunks 3 \
  --timeout-seconds 900 \
  --min-mem-available-gb 5 \
  --child-memory-limit-gb 12 \
  --output build/parakeet-variant-bench/fp32-projected-slim-onnx-broad-slim-only-001.json \
  fp32_projected_slim=build/model-variants/nemotron-fp32-projected-slim-onnx
```

| Variant | Source | Load seconds | Real-audio RTF | Decode seconds | WER |
| --- | --- | ---: | ---: | ---: | ---: |
| `fp32_projected` | recorded 3-run median | 2.717 | 0.501 | 187.533 | 25 / 985 = 2.54% |
| `fp32_projected_slim` | new 1-run check | 2.866 | 0.534 | 199.931 | 25 / 985 = 2.54% |

Interpretation:

- The graph was genuinely simplified and WER was unchanged on the 985-word
  sampled LibriSpeech validation.
- Performance was not clearly improved. The smoke run was neutral/slightly
  positive, but the broad one-run check was slower than the recorded baseline.
- Do not promote slimming into the default FP32 projected recipe yet. If we use
  it as a preprocessing step for another ORT-format conversion attempt, treat it
  as a memory-pressure experiment and validate performance again afterward.

Follow-up ORT-format conversion attempts:

- `--optimization-level all` on the slimmed FP32 projected model still drove
  available memory below 1 GiB during encoder conversion, so the run was
  interrupted before swap pressure became dangerous.
- `--optimization-level disable` avoided the first memory collapse but still
  crashed with exit 139 during encoder conversion.
- Conclusion: `onnxslim` reduces graph/protobuf complexity, but does not make
  the 2.3 GiB FP32 external-data encoder safe to convert to ORT format on this
  16 GiB system. Keep ORT format focused on the compact model unless we move
  conversion to a larger-memory host or find a converter path that streams
  external data instead of materializing it.

## 2026-06-22: Sayboard Harvest Wrapper Results

The remaining Sayboard-derived experiments are now captured by:

```sh
.venv/bin/python scripts/run_sayboard_harvest_experiments.py --force
```

The wrapper rebuilds the release worker before benchmarking, because stale
worker binaries can otherwise hide ABI changes such as removing
`processed_signal_length` from the encoder inputs.

Fixed-length ABI result:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.641 | 0.631 | 79.202 | 83.357 | 9 / 313 = 2.88% |
| `fixed_length` | 0.706 | 0.696 | 87.338 | 88.534 | 10 / 313 = 3.19% |

Result file:
`build/parakeet-variant-bench/sayboard-fixed-length-001.json`

FP32 current-projection result:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.642 | 0.632 | 79.330 | 83.529 | 9 / 313 = 2.88% |
| `fp32_current_projection` | 0.729 | 0.718 | 90.114 | 91.732 | 12 / 313 = 3.83% |

Result file:
`build/parakeet-variant-bench/sayboard-fp32-current-projection-001.json`

Conclusion:

- Removing `processed_signal_length` is not a win for the current ORT CPU path.
  It made the graph smaller but produced slower decoding and a small WER
  regression on this sample.
- FP32 current K/V projection after quantization is also not a win here. It is
  slower than the current `ffn_fp32` baseline and inherits/worsens the FP32
  export accuracy gap.
- Keep `build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort` as the
  current best local candidate.

### Clean FP32 Default Dynamic Quantization

This tests the default clean-FP32 transform path directly: dynamic QUInt8
quantization, projected cache with dynamic-int8 current projection, fixed-shape
ORT `extended`, and the same FFN FP32 post-pass.

```sh
.venv/bin/python scripts/run_sayboard_harvest_experiments.py \
  --experiment fp32-default-quantization \
  --force
```

The FFN dequantization phase rewrote `0` blocks, so the clean FP32 quantized
graph still does not reach the same mixed int8/FP32 encoder form as the current
sherpa-derived best candidate.

Result file:
`build/parakeet-variant-bench/sayboard-fp32-default-quantization-001.json`

Power state: AC online, battery charging, GNOME profile `balanced`.

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.655 | 0.646 | 81.000 | 85.521 | 9 / 313 = 2.88% |
| `fp32_default_quantization` | 0.747 | 0.736 | 92.377 | 93.683 | 10 / 313 = 3.19% |

Conclusion:

- The default clean-FP32 dynamic quantization path is perf-negative and worsens
  rough WER on the current long-WAV sample.
- This reinforces that the current best result depends on the sherpa-derived
  encoder graph form plus the FFN FP32 rewrite, not merely on starting from a
  clean FP32 export and applying ORT dynamic quantization.

### Per-Channel Dynamic Quantization

Sayboard's ablation matrix included `*_pc` variants using ONNX Runtime dynamic
quantization with `per_channel=True`. NeMote now exposes the same switch on
the clean FP32 transform path:

```sh
.venv/bin/python scripts/run_sayboard_harvest_experiments.py \
  --experiment per-channel-quantization \
  --force
```

The build path is:

1. Copy the existing FP32 consolidated export artifacts.
2. Run `scripts/transform_nemotron_parakeet_export.py --quantize --projected-cache --quantize-per-channel`.
3. Run fixed-shape specialization with ORT `extended`.
4. Run the same FFN FP32 dequantization phase.
5. Benchmark against `build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort`.

The FFN dequantization phase rewrote `0` blocks for this variant because
per-channel quantized weights use vector scale/zero-point tensors, while the
current MatMul dequantizer intentionally handles only scalar-scale quantized
weights. This result is therefore a clean FP32-export per-channel quantization
test, not the current best `ffn_fp32` pipeline with per-channel weights.

Result file:
`build/parakeet-variant-bench/sayboard-per-channel-quantization-001.json`

Power state: AC online, battery charging, GNOME profile `balanced`.

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.645 | 0.636 | 79.792 | 84.353 | 9 / 313 = 2.88% |
| `per_channel_quantization` | 0.723 | 0.713 | 89.429 | 90.704 | 10 / 313 = 3.19% |

Conclusion:

- Per-channel dynamic QUInt8 quantization from the clean FP32 export is
  perf-negative on this ORT CPU path.
- It also carries the same rough accuracy concern as the other FP32-export
  variants.
- Do not promote per-channel quantization into the default model pipeline.

## 2026-06-22: Export-Parity Graph Audit

To make the FP32-export parity investigation repeatable, package comparison is
now scripted:

```sh
.venv/bin/python scripts/compare_nemotron_model_packages.py \
  build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  build/model-variants/sayboard-harvest/fp32-current-projection-ffn-fp32-ort \
  --json-out build/export-parity/current-best-vs-fp32-current-projection.json
```

Additional reports:

- `build/export-parity/current-best-vs-fp32-projected.json`
- `build/export-parity/current-best-vs-fp32-default-quantization.json`
- `build/export-parity/current-best-vs-per-channel.json`

Findings:

- `tokenizer.model` is byte-identical across current best, FP32 projected,
  FP32-current-projection, and per-channel variants:
  `ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291`.
- Prompt dictionary, preprocessor config, vocab size, blank id, test input
  shape, and test output shape match between current best and the quantized
  FP32-export variants.
- The decoder/joint graph is equivalent at the useful level for the quantized
  FP32-export variants: `DynamicQuantizeLSTM=2`, `MatMulInteger=3`, and
  `DynamicQuantizeLinear=3`, matching current best.
- The encoder graph is not equivalent. Current best has `FusedMatMul=48`,
  `DynamicQuantizeMatMul=99`, `DynamicQuantizeLinear=173`,
  `ConvInteger=77`, and `MatMul=120`. The FP32-default-quantization variant has
  `FusedMatMul=0`, `DynamicQuantizeMatMul=195`, `DynamicQuantizeLinear=77`,
  `ConvInteger=77`, and `MatMul=72`; FP32-current-projection has
  `DynamicQuantizeMatMul=147` and `MatMul=120`.
- The current best FFN dequantization pass rewrote `96` blocks and pruned `192`
  initializers. The FP32-export quantized variants rewrote `0` blocks in the
  same phase, so they did not reach the same mixed int8/FP32 encoder graph.

Interpretation:

- The observed FP32-export WER/perf differences are not explained by tokenizer,
  prompt selection, preprocessor metadata, or decoder/joint ABI differences.
- The likely remaining parity issue is encoder graph structure after ORT
  quantization/fusion: the sherpa-derived/current-best path exposes scalar-scale
  FFN quantization tails that NeMote can dequantize back to FP32 and retains
  `FusedMatMul` nodes after ORT optimization, while the clean FP32 export path
  produces a different fused graph.
- Further parity work should focus on reproducing the current-best encoder
  graph form from the NeMo export path, or on extending the dequantizer to
  handle the FP32-export quantized patterns if they can be matched without
  hurting WER.

## 2026-06-23: Microsoft Olive Spike

Olive setup was done in the existing Nemo export environment instead of a fresh
default Olive venv. A normal `pip install olive-ai` attempted to pull a large
PyTorch/CUDA dependency stack, while `.venv-nemo-export` already had CPU torch
and the ONNX stack. The working lightweight setup was:

```sh
.venv-nemo-export/bin/pip install olive-ai==0.13.0 --no-deps
.venv-nemo-export/bin/pip install \
  onnxscript==0.7.0 \
  onnxoptimizer==0.4.2 \
  onnx_ir==0.2.1 \
  questionary \
  opentelemetry-sdk \
  opentelemetry-api==1.42.1 \
  opentelemetry-semantic-conventions==0.63b1
```

Use `MPLCONFIGDIR=build/matplotlib-cache` for Olive commands so matplotlib
does not try to populate an unwritable home-cache path.

The repeatable wrapper is:

```sh
MPLCONFIGDIR=build/matplotlib-cache \
  .venv-nemo-export/bin/python scripts/run_olive_onnx_pass.py \
  build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  build/model-variants/nemotron-c56-fixed-shape-olive-peephole \
  --pass-name peephole \
  --force
```

It preserves the NeMote model directory layout and writes
`olive_pass_summary.json` with before/after graph counts. Supported experiment
passes are `peephole`, `quant-preprocess`, `dynamic-quant`, and
`ort-transformers`.

### Compact Fixed-Shape Peephole

```sh
MPLCONFIGDIR=build/matplotlib-cache \
  .venv-nemo-export/bin/python scripts/run_olive_onnx_pass.py \
  build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  build/model-variants/nemotron-c56-fixed-shape-olive-peephole \
  --pass-name peephole \
  --force
```

Graph changes:

| Component | Nodes | Initializers | Size delta | Notable op deltas |
| --- | ---: | ---: | ---: | --- |
| encoder | 2,091 -> 2,036 | 1,428 -> 1,056 | -253 KiB | `Slice -26`, `Pad -24`, `Expand -5` |
| decoder_joint | 55 -> 39 | 29 -> 29 | -818 B | `Constant -16` |

Additional diagnostics:

- Encoder unresolved dims improved from `222` to `0`.
- Quantized operator counts were preserved:
  `DynamicQuantizeMatMul=195`, `DynamicQuantizeLinear=77`,
  `ConvInteger=77`.
- Decoder quantized operator counts were preserved:
  `DynamicQuantizeLinear=3`, `MatMulInteger=3`, `DynamicQuantizeLSTM=2`.

Long-WAV benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  compact=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  olive_peephole=build/model-variants/nemotron-c56-fixed-shape-olive-peephole \
  --runs 3 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/olive-peephole-compact-001.json

.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/olive-peephole-compact-001.json
```

| Variant | Median load seconds | Median real-audio RTF | Median decode seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: |
| `compact` | 1.169 | 0.736 | 91.064 | 10 / 313 = 3.19% |
| `olive_peephole` | 1.072 | 0.735 | 90.933 | 10 / 313 = 3.19% |

Interpretation: Olive peephole is correct and slightly reduces ONNX startup
cost, but decode speed is effectively neutral.

### Compact Peephole Plus ORT Format

The peephole output converted cleanly to native ORT format:

```sh
.venv/bin/python scripts/convert_nemotron_to_ort_format.py \
  build/model-variants/nemotron-c56-fixed-shape-olive-peephole \
  build/model-variants/nemotron-c56-fixed-shape-olive-peephole-ort-format \
  --optimization-level all \
  --optimization-style fixed \
  --force
```

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  compact_ort=build/model-variants/nemotron-c56-fixed-shape-ort-extended-format \
  olive_peephole_ort=build/model-variants/nemotron-c56-fixed-shape-olive-peephole-ort-format \
  --runs 3 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/olive-peephole-ort-format-001.json

.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/olive-peephole-ort-format-001.json
```

| Variant | Median load seconds | Median real-audio RTF | Median decode seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: |
| `compact_ort` | 0.473 | 0.765 | 94.567 | 10 / 313 = 3.19% |
| `olive_peephole_ort` | 0.452 | 0.756 | 93.438 | 10 / 313 = 3.19% |

Interpretation: ORT format remains the main load-time win. Olive peephole adds
a small, clean improvement on top, but the delta is small enough to treat as
nice-to-have rather than a major optimization.

### Quantization Preprocess

Default Olive quantization preprocessing failed on the compact encoder:

```text
Exception: Incomplete symbolic shape inference
```

Retrying with symbolic shape inference disabled completed:

```sh
MPLCONFIGDIR=build/matplotlib-cache \
  .venv-nemo-export/bin/python scripts/run_olive_onnx_pass.py \
  build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  build/model-variants/nemotron-c56-fixed-shape-olive-quant-preprocess-nosym \
  --pass-name quant-preprocess \
  --pass-config '{"skip_symbolic_shape": true}' \
  --force
```

| Component | Nodes | Size delta |
| --- | ---: | ---: |
| encoder | 2,091 -> 2,091 | +42 KiB |
| decoder_joint | 55 -> 55 | +2 KiB |

Interpretation: this path is not useful for the current already-fixed-shape,
already-optimized compact graph.

### Dynamic Quantization

```sh
MPLCONFIGDIR=build/matplotlib-cache \
  .venv-nemo-export/bin/python scripts/run_olive_onnx_pass.py \
  build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  build/model-variants/nemotron-c56-fixed-shape-olive-dynamic-quant \
  --pass-name dynamic-quant \
  --force
```

Olive/ORT preprocessing again failed symbolic shape inference and fell back to
the original graph. The encoder changed from `2,091` to `2,187` nodes:

| Op | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `MatMul` | 72 | 48 | -24 |
| `MatMulInteger` | 0 | 24 | +24 |
| `DynamicQuantizeLinear` | 77 | 101 | +24 |
| `Cast` | 92 | 116 | +24 |
| `Mul` | 235 | 283 | +48 |

Single-run long-WAV screen:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  olive_dynamic_quant=build/model-variants/nemotron-c56-fixed-shape-olive-dynamic-quant \
  --runs 1 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/olive-dynamic-quant-screen-001.json

.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/olive-dynamic-quant-screen-001.json
```

Result: `load=1.288s`, `real-audio RTF=0.764`, `decode=94.515s`,
`12 / 313 = 3.83%` rough WER.

Interpretation: rejected. It is slower than the compact ONNX baseline in the
same experiment family and worsens rough WER.

### ORT Transformer Optimization

```sh
MPLCONFIGDIR=build/matplotlib-cache \
  .venv-nemo-export/bin/python scripts/run_olive_onnx_pass.py \
  build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  build/model-variants/nemotron-c56-fixed-shape-olive-ort-transformers \
  --pass-name ort-transformers \
  --force
```

Olive skipped both graphs:

```text
Unsupported model type: None, will skip this pass.
```

Supported model families listed by Olive included BART, BERT, GPT-2, GPT-NeoX,
Phi, Qwen3, T5, ViT, conformer, and diffusion components. The current
Nemotron/Parakeet RNNT package does not present metadata or graph structure
that this pass recognizes.

### FP32 Projected Model Boundary

FP32 decoder-only peephole is safe:

```sh
MPLCONFIGDIR=build/matplotlib-cache \
  .venv-nemo-export/bin/python scripts/run_olive_onnx_pass.py \
  build/model-variants/nemotron-fp32-projected-fixed-shape \
  build/model-variants/nemotron-fp32-projected-fixed-shape-olive-peephole-decoder \
  --pass-name peephole \
  --component decoder_joint \
  --force
```

Decoder graph: `42 -> 26` nodes, size `+992 B`.

FP32 encoder peephole is not safe on this machine. The first attempt failed
before optimization because ONNX 1.22 refused to load hard-linked external data:

```text
ValidationError: Data of TensorProto (...) should be stored in .../encoder.onnx.data,
but it has multiple hard links, indicating a potential hardlink attack.
```

A private non-hardlinked copy bypassed that loader check, but the process ran
out of memory while Olive tried to load/optimize the 2.45 GiB external-data
encoder. This matches Olive's implementation: `OnnxPeepholeOptimizer` calls
`onnx.load(...)` and materializes external data in memory. Do not run Olive
peephole/quantization passes over the FP32 projected encoder on this 16 GiB
machine unless Olive gains a streaming external-data path or the conversion is
moved to a larger-memory host.

### Olive Upshot

- Useful: `OnnxPeepholeOptimizer` on the compact fixed-shape graph. It is
  correct, reproducible, removes some remaining graph clutter, improves encoder
  shape metadata, and gives a small load/RTF improvement when combined with ORT
  format.
- Not useful: `OnnxQuantizationPreprocess` for this graph. Default symbolic
  shape inference fails; the no-symbolic retry is a no-op.
- Rejected: `OnnxDynamicQuantization` on the compact graph. It quantizes
  remaining attention matmuls but regresses both speed and rough WER.
- Not applicable: `OrtTransformersOptimization`; the graph is unsupported.
- Unsafe locally: FP32 projected encoder Olive passes, due to full external
  data materialization and observed out-of-memory behavior.

Recommendation: keep the Olive wrapper as an experimental tool and consider
adding compact peephole before ORT-format conversion in the compact-model
recipe. Do not make Olive a required dependency of the normal model-tools path.
