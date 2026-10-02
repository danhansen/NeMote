# NeMote graph and streaming pipeline performance audit

This audit examines the current 0.1.20 source, the English FP32 and corrected
QUInt8 model artifacts, and the pinned parakeet-rs fork. Its purpose is to
identify remaining performance headroom without treating every graph node or
timing fluctuation as a problem to fix.

The strongest remaining graph opportunity is the compact model's depthwise
convolution execution path. The strongest runtime investigation is coordination
of the encoder and decoder thread pools. A separate product concern is now
reproducible: FP32 at 80 ms chunks is marginally slower than real time on this
host. Raw-cache cleanup and QKV fusion are concrete structural opportunities,
but current measurements do not establish an end-to-end improvement exceeding
the user's 5% noise band.

No runtime code, shipped model, configuration, dependency, or publication was
changed for this audit. Diagnostic profiles, isolated kernel experiments, and
this report were created locally. No microphone recording was used.

## Scope and evidence

The audit uses the corrected compact generic encoder, not the earlier export
with 48 unnecessary ConvInteger pointwise projections. The fork is pinned to
`5d24800e435a48bcca28380cff3876f236cacce9`; the application source inspected is
`18c0c1c36b6704bb76939a77a10ff8e5d49efd4b`.

Measurements were made on September 30, 2026, local time, on an Intel Core
i5-1235U with hybrid P/E cores and 12 MiB L3, using CPU ORT 1.28.0 and two
intra-op threads. Historical experiments in this repository primarily used an
Ivy Bridge i5-3320M and multilingual c56 graphs. Those results identify prior
attempts and risks; they are not current-host benchmarks.

Fresh encoder profiles cover compact at all four supported contexts and FP32
at 560 ms, using full-cache-shaped synthetic fixtures. Each profile includes
three warmup calls and nine timed calls. Categories are based on node names,
and percentages are shares of recorded node-event duration, not end-to-end
worker time or aggregate CPU utilization. Profiling adds overhead, especially
to small graphs. Do not compare these wall times directly with uninstrumented
speech benchmarks.

Additional checks included nine counterbalanced pairs per shape for output
selection; isolated QKV kernels with nine counterbalanced timing blocks;
decoder shape-specialization diagnostics; CPU sampling of the actual worker
over the 29.4-second speech clip; and five independent FP32 80 ms worker runs
on the 5.855-second clip. Diagnostic artifacts and scripts are retained under
`build/performance-audit/` and are not part of the published models.

## Current graph and first principles cost model

The specialized corrected compact graph has 1,843 nodes, including 241
DynamicQuantizeMatMul and 29 ConvInteger nodes. Shape and Range operators are
absent. Relative-position `linear_pos` projections have also been folded away.
The existing generic export plus session dimension overrides already removes
the principal shape-plumbing cost without publishing per-context models.

At 560 ms, the encoder processes seven current frames. A feed-forward block
maps 1024 to 4096 channels and back. Across two feed-forward blocks per layer
and 24 layers, that is about 2.819 billion multiply-accumulates per chunk.
In contrast, the 24 depthwise convolutions perform only about 1.548 million
multiply-accumulates: 24 × 1024 channels × nine taps × seven frames.
Their radically different measured costs are more informative than node counts.

| Encoder component | Compact node time | FP32 node time |
| --- | ---: | ---: |
| Feed-forward blocks | 38.7% | 51.9% |
| Attention family including projections | 21.6% | 20.5% |
| Depthwise convolution family | 16.6% | 1.9% |
| Pointwise projections | 8.4% | 14.4% |
| Subsampling family | 4.7% | 2.2% |
| Other operations | 9.9% | 9.1% |

These are different precision profiles, not an isolated convolution A/B.
Nevertheless, the depthwise family averaged approximately 11.56 ms per profiled
compact call versus 2.42 ms per profiled FP32 call. This is a measured kernel
cost gap worth isolating, not a forecast of application speedup.

The optimized compact graph contains approximately 556.6 MiB of UINT8
initializers and 15.7 MiB of FLOAT initializers, vastly exceeding this host's
last-level cache. Small chunks reduce useful rows per matrix multiply without
proportionally reducing weight traversal or fixed dispatch costs. This suggests
a combination of memory traffic and small-matrix kernel overhead; hardware
bandwidth and stall counters would be needed to establish a roofline bottleneck.

Compact's profiled median encoder time grows from roughly 45.1 ms at 80 ms
chunks to 48.8, 74.9, and 113.6 ms at 160, 560, and 1120 ms. Computation per
second of audio therefore changes substantially with chunk size. Supporting
a context correctly does not imply equal efficiency across contexts.

## Dense kernels and precision policy depend on the host

Feed-forward layers are the largest cost center, but much of their work is
necessary dense arithmetic and weight movement. Their profile share alone does
not prove that changing precision, replacing GEMM, or removing operators will
help. The published fast/compact names reflect earlier product choices, not
a universal hardware ordering. On this host, the existing corrected speech
comparisons show compact decoding substantially faster than FP32, while the
small corpus gives FP32 fewer word errors. Preserve that accuracy/footprint
tradeoff rather than treating either profile as universally superior.

Before changing dense kernels, measure achieved bandwidth, cache misses, and
instructions on the actual supported hardware, especially at one and two
current frames. Compare matrix kernel choices and precision boundaries with
the same quantized values wherever possible. An FP32-only pointwise MatMul
lowering is a separate candidate from the validated compact lowering; its
roughly 14% profile share justifies an isolated test but does not establish a
win. Weight-only quantization, static activation quantization, BF16, or GPU
execution have different hardware and correctness requirements and are not
drop-in consequences of the current profile.

The historical FFN dequantization and MatMulNBits experiments are particularly
important negative controls: some helped old hardware, some hurt speed or WER,
and their results cannot be transplanted to the current English graph. Keep
one generic artifact per selected precision, and make hardware-dependent choices
in validated runtime specialization rather than multiplying published chunk
exports.

## Depthwise convolutions are the leading graph experiment

The remaining 24 Conformer depthwise nodes use group=1024, kernel=9, stride=1,
and weights shaped [1024,1,9]. They perform tiny independent channel-wise
filters, but currently use ConvInteger rather than a dedicated depthwise path.

The current upstream [ORT ConvInteger implementation](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/quantization/conv_integer.cc)
loops over groups and performs im2col plus quantized GEMM for non-unit kernels.
That mechanism is consistent with this graph's disproportionately expensive
depthwise nodes. The exact deployed 1.28.0 implementation was not matched to
symbols here, so kernel-source attribution remains an inference; the local
graph shapes and timings are directly observed.

The first experiment should isolate these 24 nodes, preserving all pointwise,
subsampling, and attention paths. Two options have different correctness risks:

- Keep only depthwise convolutions FP32 before quantization. The extra raw
  weight storage is approximately 0.63 MiB, not hundreds of MiB. This preserves
  compact's footprint, but changes activation quantization and numerical output.
  It needs native numerical checks and speech validation, not an assumption
  that higher precision cannot change decoding.
- Preserve existing integer arithmetic but implement a vectorized depthwise
  path. A portable ONNX tap-wise formulation or custom ORT kernel can avoid
  thousands of tiny grouped GEMMs. The former can add dispatch overhead; the
  latter adds packaging and maintenance requirements. Neither is yet measured.

Historical all-convolution FP32 experiments improved speed but changed speech
and worsened the small WER sample. That is a reason to isolate the depthwise
subset, not to repeat the broad rewrite. If this component's cost were merely
halved, Amdahl's law permits about an 8% reduction in encoder node time under
this profile. End-to-end benefit would be smaller and must be measured.

## Thread pools expose substantial waiting activity

The actual worker CPU sample contains 3,045 samples with no reported lost
samples. Approximately 43.2% of atom-event cycles and 46.1% of core-event
cycles were attributed to the same ORT wait routine. Disassembly shows TPAUSE
with a PAUSE fallback. The sampled location appears predominantly on background
threads, rather than the application's main thread.

The fork creates independent encoder and decoder sessions and gives both the
same two-thread configuration. The runtime executes them serially. An inactive
pool continuing to wait/spin while the other session computes is a plausible
source of avoidable CPU activity. Thread-pool synchronization within active
operators can also contribute. This trace includes cached session startup and
teardown and does not distinguish every cause of waiting.

Do not interpret these percentages as 45% removable decode latency or measured
power savings. Cycle sampling includes time spent waiting, including hardware
wait instructions. The observation justifies a controlled scheduling experiment.

The relevant [ORT threading controls](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
include shared global pools, spinning control, and bounded spin duration.
The pinned [1.28 configuration definitions](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.28.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
also expose `session.force_spinning_stop`, which stops pool spinning when the
last Run returns. The locally installed Rust ORT crate exposes spinning
configuration and global-pool support; its interface is not a blocker.

Compare default behavior with force-stop between Runs, encoder two threads /
decoder one thread, and a shared pool. Measure decode latency, p95 chunk time,
task CPU time, and, where available, energy. Explicit affinity on this hybrid
CPU is an experiment, not a portable default. Historical two-thread results
on Ivy Bridge do not establish the best policy for every graph and context.

## FP32 at 80 ms needs a real time performance gate

The previous all-context smoke test checked that transcripts existed and matched
the expected phrase. Its FP32 80 ms result was already slower than real time.
Five fresh worker runs reproduce the concern without profiling instrumentation:

- Audio duration: 5.855 seconds; 77 calls including three flush calls.
- Median decode time: 6.681 seconds.
- Median real-audio RTF: 1.141; range approximately 1.027 to 1.205.
- Median processed-audio RTF, including synthetic padding: 1.085.
- All five transcripts were identical. Startup was measured separately.

This is a specific local replay result, not proof that every device or live
utterance fails at 80 ms. It is nevertheless outside the 5% noise band at the
median and cannot be explained solely by normalizing against unpadded audio.
On a live stream, persistent processing slower than arrival accumulates backlog.

Keep all checkpoint-supported choices available, but distinguish model support
from measured real-time suitability. The release matrix should cover both
precisions at every supported size, with per-chunk p50/p95/p99 and backlog in
milliseconds. Consider an explicit warning or opt-in hardware check, not silent
replacement of the user's chosen size.

The CPU queue currently bounds callback-buffer count using an assumed 100
callbacks per second, while CPAL uses a device-default buffer size. Therefore
`queue_seconds` is not an exact sample-duration budget. Queue saturation and
dropped chunks can hide behind throughput summaries. A sample-count budget and
oldest-buffer age would make overload behavior predictable.

## Projected caching still carries an unused raw cache branch

The projected-cache rewrite avoids reprojecting old attention history, which is
already an important optimization. However, the final graph retains both raw
channel cache and projected K/V state.

Dependency analysis of the corrected optimized graph establishes that removing
only `cache_last_channel_next` from the requested outputs makes
`cache_last_channel` unnecessary to the remaining outputs. A 123-node exclusive
branch remains solely for raw cache maintenance: 49 Slice, one Split, 24
Squeeze, 25 Concat, and 24 Unsqueeze nodes. That branch accounts for about 4.2%
of recorded node time in the earlier corrected compact profile.

The raw state occupies 6.5625 MiB. Projected K/V occupies another 13.125 MiB.
The Rust runtime copies raw outputs back into host arrays and shifts projected
history with `copy_within` on every chunk. These are fixed-context costs that
become more frequent as chunks shrink.

I tested selecting all outputs except the raw channel-cache output with ORT's
only-execute-fetch-path option. All retained outputs were bit-identical. Across
all tested contexts and the FP32 560 ms case, median timing changes were roughly
−1.2% to +0.2%: noise, not a demonstrated application speed improvement.

Removing this branch could simplify a versioned projected-cache ABI and reduce
state and memory traffic. It should not be prioritized as a proven >5% speed
fix. Compatibility requires deriving dimensions from metadata rather than
requiring the removed raw input. Preserve the existing ABI for older models.

Borrowed TensorRef inputs already avoid blanket input copies. Output ownership
and cache rolling still merit measurement. The current Rust ORT crate supports
output binding, but [I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
is not an automatic CPU speedup. Any design must respect owner-bound direct-ORT
session lifetimes and avoid aliasing input and output cache buffers. A ring
buffer cannot simply replace ordered attention history without adapting layouts
and relative-position semantics.

## QKV fusion is feasible but not a universal win

Each of the 24 layers has three DynamicQuantizeMatMul nodes consuming the exact
same normalized activation: Q, current K, and current V. That creates 48
additional activation-quantization executions relative to one combined
projection per layer.

An isolated diagnostic concatenated the existing quantized weights, preserving
each original projection's scale and zero point as column-wise parameters,
then split the output. ORT's [operator schema](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/graph/contrib_ops/quantization_defs.cc)
supports those column-wise quantization parameters. No model-wide requantization
was used. Outputs were bit-identical for the sampled inputs at all four sizes.

The median isolated-kernel changes were approximately −26% at one frame, +37%
at two frames, −30% at seven frames, and −29% at fourteen frames. This is one
layer's kernel diagnostic, not full-graph native-cache or speech validation;
the two-frame regression also needs replication before attributing a cause.

At 560 ms, these three projections account for only 7.7% of compact node time.
Even a 30% improvement in them suggests roughly 2.3% encoder-time reduction
before downstream layout effects, inside the user's noise band. QKV fusion is
a technically plausible follow-up, especially for small chunks, but not an
evidence-backed blanket default. Do not fuse and globally requantize weights
in a way that loses the original three scale domains.

## Decoder and feature extraction repeat avoidable work

The RNNT loop runs the combined predictor/joint graph at every encoder frame,
possibly several times per frame. Predictor inputs depend on the previous
label and recurrent state, not the encoder frame. A blank leaves both unchanged,
yet the next frame recomputes the same predictor and candidate recurrent state.
The joint encoder projection also repeats when multiple labels are emitted at
one frame.

A split predictor and joint representation could cache predictor results until
a label is accepted, and cache the encoder projection within a frame. This is
an exact algorithmic redundancy in the current control flow. Changing the graph
boundary requires careful state-commit semantics, numerical/token-trace checks,
and preservation of the existing symbol cap. Its end-to-end payoff is unmeasured.

Decoder-only diagnostics measured approximately 0.145 ms per compact call and
0.835 ms per FP32 call on one synthetic input. Fixing all symbolic decoder
dimensions did not simplify compact's 22-node optimized graph; it changed FP32
kernel representation and was not a demonstrated win. Do not assume dynamic
axes alone are expensive when the optimizer has already removed shape work.

Each decoder call currently allocates/copies logits and recurrent-state arrays,
including candidate states discarded after a blank. Persistent buffers or
output selection could reduce this, but predictor caching is the more meaningful
structural experiment if decoder time warrants it.

Nemotron recomputes the mel spectrogram over its bounded audio window and builds
a new FFT plan, window, and scratch buffers for each call. A reusable FFT-plan
path already exists for other fork models but is not used by Nemotron. A cached
feature plan and eventually incremental STFT/mel extraction are reasonable
cleanup candidates. The buffer is trimmed: this is not quadratic growth over an
entire dictation session. Incremental features must match preemphasis, padding,
window alignment, overlap, and trimming at every supported context.

CPU samples attributed only about 0.1–0.5% of leaf cycles directly to the worker
binary, with most samples in ORT and libc. That is not a complete stage breakdown,
but it argues against assuming Rust loops, JSON output, or FFT planning are the
dominant current problem. The application mutex is held once per decoder loop,
and capture buffers are already pooled.

## Model caches leave disk and startup costs on the table

The published generic model contract is sound. The local cache implementation
is not yet compact: ORT serialization materializes another large optimized
weight sidecar for each cache entry. Two existing FP32 encoder cache entries
each contain about 2.187 GiB of optimized data, plus a conventional 2.269 GiB
source-data sidecar that their optimized graphs do not reference. One source
sidecar is hard-linked; the other is a physical copy. Apparent file totals and
unique physical storage must be distinguished.

The active policy can therefore create roughly another 2.2 GiB of optimized
weights per selected context, even though most weights are context-independent.
Four contexts could approach 8.8 GiB of additional optimized weights before
source files, stale entries, or decoder artifacts. This is a storage projection
from the serialization behavior, not four contexts measured on disk.

Cache keys conservatively include source path/mtime, configuration, ORT build,
hardware features, thread settings, and dimension overrides. This avoids unsafe
reuse, but decoder keys also vary with encoder-only context overrides even
though the decoder computation is unchanged. No explicit cache-size budget or
eviction policy is present in this path. The cache serializes graphs, not a
promise that all packed weights and execution state survive process restart.

Prioritize sharing only byte-identical immutable weights, separate per-context
folded constants, prune unreferenced sidecars, and add bounded cache lifecycle
management. Do not just repoint optimized initializer offsets into a source
file: transformed tensors may differ. Preserve hardware/version invalidation
and corrupted-cache fallback. These remain local optimization artifacts; they
do not require publishing several models or making users run NeMo exports.

The worker retains loaded sessions across dictation stops, which is already
correct. First use still loads lazily; context changes restart the worker and
reload both sessions. Optional prewarming or retaining a context plan may help
interactive startup, but keeping several full FP32 sessions alive would conflict
with lower-memory machines. Measure hotkey-to-listening separately from decode.

## Pipeline and release gates need stronger attribution

The canonical builder correctly separates NeMo/Torch export from ONNX
quantization, preventing both large stacks from remaining resident together.
Dynamic packaging preserves symbolic chunk dimensions, and the current lowering
checks restrict rewrites to mathematically compatible pointwise nodes. Those
decisions should be retained.

There are still multiple transformation entry points. The exporter has its own
inline quantization path; the canonical builder instead calls the separate
transformer that applies pointwise lowering. A maintainer using the former can
produce a different compact graph. Centralize the production transform contract
or make noncanonical paths unmistakable, and record graph execution fingerprints
so lost fusions cannot silently recur.

The publisher verifies hashes of final model files, but benchmark, corpus, and
parity reports are not all cryptographically bound to the artifacts when those
reports were measured. A freshly generated validation file can bind a changed
model to stale input reports. Path equality in one parity check does not prevent
replacement at the same path. Record artifact digests, checkpoint/fixture/audio
digests, script/fork/worker commits, ORT version, selected context, and session
settings in the measurement reports, then compare them at gate time.

Current performance gates use paired medians and the explicit 5% band, an
improvement over ratios of independent medians. They still require only five
small matched clips and one benchmark context. They do not establish full-corpus
WER, real-time tail latency, frontend parity, or equal performance at all sizes.
The numerical fixtures cover single-step cache occupancies, not every long
cross-step trajectory. Add longer stateful numerical/token checks, representative
speech/silence/punctuation sessions, and context-specific regression tests.

Stage observability is the highest-value prerequisite for runtime work:
`decode_seconds` currently combines feature extraction, encoder inference, cache
exchange, and RNNT decoding. Add diagnostic-only counters for each stage,
decoder/blank decisions, memory allocation, queue age, and flush time. Keep
profiling disabled in ordinary use and benchmark with identical instrumentation.

Three silence chunks are currently used for every context. That means 240 ms
of synthetic audio at 80 ms but 3.36 seconds at 1120 ms. This affects stop cost,
energy, and short-clip RTF. Derive and validate sufficient flushing from streaming
state/receptive field rather than assuming fewer calls preserve final tokens.

## Ranked follow up experiments

| Priority | Experiment | Evidence and acceptance condition |
| --- | --- | --- |
| First | All-context performance matrix and stage counters | FP32 80 ms real-time risk is reproducible; attribute costs and measure tails before changing defaults. |
| First | Depthwise-only kernel or precision experiment | About 17% of compact node time for only 1.55 million MACs; require >5% end-to-end benefit without transcript/accuracy regression. |
| First | Encoder and decoder pool coordination | Wait routine dominates many sampled cycles; require measured CPU/energy improvement with no meaningful latency regression. |
| Next | Cache weight sharing and bounded eviction | Multi-GiB duplicated local data is directly observed; verify reload parity and unique disk/RSS improvements. |
| Next | Predictor and joint caching | Repeated identical predictor inputs across blanks are source-proven; measure decoder share and verify token/state trajectories. |
| Later | Raw-cache ABI simplification | Redundant branch and 6.56 MiB state are proven, but fetch-only timing differences were noise. |
| Later | QKV fusion | Isolated feasibility established, but context-dependent timings and small total share preclude a blanket recommendation. |
| Later | Reusable feature buffers and incremental frontend | Source-level repetition exists; preserve exact streaming feature semantics and first establish a meaningful stage cost. |

Do not broaden quantization, enable approximation, shrink trained history,
blindly enable parallel graph execution, or switch to weight-only int4 based
on smaller files or fewer nodes. Historical experiments already show that such
changes can worsen either throughput or decoding. Revisit them only with a
specific hardware/kernel hypothesis and adequate validation. Likewise, generic
attention fusion is not automatically appropriate for this relative-position
Conformer; the attention score/context MatMul kernels account for only a small
fraction of current compact node time.

The performance discipline should remain simple: one targeted intervention,
counterbalanced comparisons, unchanged input/precision/context, correctness
first, and no adoption on an unexplained change inside ±5%. Report useful
memory, storage, startup, and energy improvements separately rather than
presenting them as decode speedups.

## Evidence and source navigation

Fresh measurements are in [encoder-audit.json](../build/performance-audit/encoder-audit.json),
[decoder-audit.json](../build/performance-audit/decoder-audit.json),
[qkv-audit.json](../build/performance-audit/qkv-audit.json),
[fast80-worker-benchmark.json](../build/performance-audit/fast80-worker-benchmark.json),
and `build/performance-audit/worker-perf.data`. These ignored local artifacts
are not carried by a clean checkout. Their diagnostic scripts are alongside
them. The earlier release comparison is documented in
[dynamic-chunk-performance.md](dynamic-chunk-performance.md).

Implementation references:

- [Projected cache graph rewrite](../scripts/rewrite_nemotron_projected_kv_cache.py).
- [Canonical model pipeline](../scripts/build_nemotron_wordpipe_model.py),
  [transformer](../scripts/transform_nemotron_parakeet_export.py), and
  [release gate](../scripts/validate_dynamic_nemotron_release.py).
- [Worker capture and decode metrics](../crates/wordpipe-parakeet-worker/src/main.rs)
  and [service lifecycle](../crates/wordpipe-service/src/main.rs).
- [Pinned fork model and cache runtime](https://github.com/danhansen/parakeet-rs/blob/5d24800e435a48bcca28380cff3876f236cacce9/src/model_nemotron.rs),
  [streaming feature and RNNT loop](https://github.com/danhansen/parakeet-rs/blob/5d24800e435a48bcca28380cff3876f236cacce9/src/nemotron.rs),
  and [session options](https://github.com/danhansen/parakeet-rs/blob/5d24800e435a48bcca28380cff3876f236cacce9/src/execution.rs).
- [Historical optimization experiments](optimization-experiments.md), with
  hardware/model/context caveats noted above.

The saved Markdown was read back and checked for source and measurement
consistency. A rendered document preview was not available.
