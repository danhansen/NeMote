# Dynamic streaming pointwise graph investigation

Measured 2026-09-30, English Nemotron 0.6B, ORT 1.28.0 CPU, two inference
threads, default 560 ms chunks. This is a local regression check, not a broad
accuracy or hardware performance claim.

## Finding and targeted change

The new native export represented both pointwise channel projections in every
Conformer layer as Conv. The deployed compact graph represented them as MatMul.
After quantization and ORT specialization, this changed 48 fused
DynamicQuantizeMatMul nodes into ConvInteger plus separate quantization,
rescaling, cast, and layout operations. Dynamic Shape/Range operations had
already been eliminated; they were not the observed kernel difference.

`rewrite_nemotron_pointwise_convs.py` lowers only named Conformer pointwise
Conv nodes with static FP32 [out,in,1] weights, group/stride/dilation 1, and no
padding. Transpose → MatMul → optional bias → transpose is mathematically
equivalent. The rewrite preserves tensor output names, cache interfaces,
dynamic chunk dimensions, and numerical weight values. Depthwise, subsampling,
and temporal convolutions are untouched. Unsupported attributes fail closed.

The exact historical reason the older export chose MatMul is not established;
the measured graph difference and targeted intervention do not depend on it.

## Results

| Optimized compact graph | Unfixed generic | Fixed generic | Deployed baseline |
| --- | ---: | ---: | ---: |
| ConvInteger | 77 | 29 | 29 |
| DynamicQuantizeLinear | 77 | 29 | 29 |
| DynamicQuantizeMatMul | 193 | 241 | 241 |

The first instrumented 12-call encoder profile reduced summed pointwise-node
time from 114.9 ms to 69.7 ms (39%). This profile used identical native input
fixtures and session options, but sequential model order; use the counterbalanced
speech benchmark below for end-to-end comparisons.

Nine interleaved passes per model alternate model order. All samples, including
outliers, are retained. The statistic is the median of within-pass decode-time
ratios, not a ratio of independent medians.

| Speech clip | Fixed vs unfixed generic | Fixed vs deployed baseline |
| --- | ---: | ---: |
| 7.152 seconds | −7.42% | +3.22% |
| 29.4 seconds | −9.34% | +2.03% |

Both baseline comparisons fall within the user's ±5% noise band. These data
support restoring the fused projection path; they do not establish the cause
of individual timing spikes. No CPU governor or clock settings were changed.
Cold generic optimization remains a startup cost, separate from steady decode.
Short-run peak RSS was 1,327,160 KiB on first optimization versus 1,284,240 KiB
for the deployed baseline (+3.34%); cached candidate median was 710,516 KiB.

The unlowered native export passes the original 1e-4 absolute/relative parity
gate. Lowered FP32 passes all 12 native fixtures plus all four specialized
sessions and exact optimized-cache reload checks with explicit 1e-3 absolute /
1e-4 relative tolerance. Strict 1e-4 absolute failed on two full-cache elements
in the initial dynamic check; maximum observed difference across the completed
checks was approximately 0.000412. FP32 kernel accumulation is not bit-identical.
Unit tests separately check biased/bias-free equivalence across dynamic shapes.
The final quantized candidate retains identical baseline transcripts on five
LibriSpeech clips (151 reference words; 11 word errors in both). Runtime smoke
checks pass at all four advertised chunk sizes. This is transcript regression
evidence, not a claim that quantized activations equal FP32 tensors.

Local detailed evidence is retained under `build/dynamic-chunk-validation/`:
`pointwise-profile-report.json`, `pointwise-native-parity-1e-3.json`,
`compact-pointwise-benchmark.json`, and `compact-pointwise-long-benchmark.json`.
The release validation file binds final runtime artifacts by SHA-256.

## Product behavior

Only compact quantized builds enable the rewrite; the FP32 export is unchanged.
One generic encoder per precision supports every checkpoint-advertised chunk
size. The current English model advertises 80, 160, 560, and 1120 ms; preferences
populate Streaming Chunk Size from runtime metadata, not a fixed model list.
Users download ready-made models and do not export on their own machines.

The source version advances from untagged 0.1.19 to 0.1.20. Publish both generic
profile repositories before tagging the application release, so clients never
receive a release that references unavailable model downloads. Keep the older
fixed English model repositories unchanged for older clients.
