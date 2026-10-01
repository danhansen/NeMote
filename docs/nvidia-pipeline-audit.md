# NVIDIA C++ streaming pipeline audit

The NVIDIA runtime is useful both as a benchmark candidate and as an independent
implementation to compare against NeMo. The comparison has already identified
two feature-alignment discrepancies in Wordpipe's parakeet-rs frontend. These
should be resolved through feature parity tests and accuracy ablations before
attributing a backend's recognition differences solely to quantization.

This audit does not recommend switching backends yet. Native GGUF speed and
forty-speaker accuracy screening are complete. No
production frontend or backend setting has been changed by this investigation.

## Sources and reproducibility

The inspected NVIDIA checkout is `NVIDIA/NeMo-Speech.cpp` revision
`4c101bc7113f49101a3e11d2c994c519f41939f6`, with ggml revision
`c03b4e2bcece5134827881af90242086daf75be5`. Local sources and diagnostic outputs
are retained in `build/performance-audit/`.

The official model is NVIDIA's
`nemotron-speech-streaming-en-0.6b.q8_0.gguf`, from Hugging Face revision
`ebe59e5a817142986528bbbee5dba8db7b38ed50`. Its verified SHA256 is
`d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d`.
The artifact explicitly specifies centered STFT windows, symmetric Hann windows,
and invalid-frame masking. These settings were read from its actual metadata,
not inferred from the latest converter defaults.

The NeMo checkpoint's preprocessor specifies a 16 kHz sample rate, 400-sample
window, 160-sample hop, 512-point FFT, 128 mel bands, and no normalization.
The installed NeMo `FilterbankFeatures.stft` uses `torch.stft` with a symmetric
Hann window, `center=True`, and constant padding. Torch centers a shorter window
within the FFT frame.

Primary implementation references are NVIDIA's
[feature extractor](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4c101bc7113f49101a3e11d2c994c519f41939f6/src/asr/features/fe.cpp),
[stream runner](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4c101bc7113f49101a3e11d2c994c519f41939f6/src/asr/runner.cpp), and
[RNNT decoder](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4c101bc7113f49101a3e11d2c994c519f41939f6/src/asr/decoders/rnnt_greedy_decoder.cpp).

## Confirmed feature alignment discrepancies

### STFT window placement

Our `audio.rs::stft_with_plan` reads the first 400 samples of each 512-sample
frame, applies the window there, and leaves the last 112 positions zero. NeMo
and the official GGUF use a centered window: 56 zero positions, 400 windowed
samples, then 56 zero positions. This changes which audio samples contribute,
not just the phase of the FFT output.

A diagnostic binary compiled the actual Rust audio source and extracted its
power spectrogram for the existing 5.855-second speech fixture. A separate
Torch calculation used the checkpoint's NeMo STFT geometry. Both outputs had
shape 257 by 586.

| Comparison | Relative L2 difference | Maximum absolute power difference |
| --- | ---: | ---: |
| Rust versus NeMo geometry | 0.531133 | 103.415764 |
| Rust versus left-aligned Torch reproduction | 0.000000214 | 0.000045776 |

The second comparison isolates window placement from ordinary FFT rounding.
The first value is a spectrogram difference, not a word-error rate or an
estimated accuracy loss. Evidence is in `frontend-parity.json`; its source and
audio hashes bind the measurement to the tested implementation and input.

The Rust mel basis matches NeMo's librosa Slaney construction within a relative
L2 difference of 4.75e-8. After mel projection and the natural logarithm, Rust
versus centered NeMo geometry differs by 0.0665 in relative L2, whereas the
left-aligned reproduction differs by 1.05e-6. This corroborates window geometry,
rather than filterbank construction, as the source of the measured mismatch.

A window-placement fix also needs a streaming boundary test. With centered
400-sample windows, the final requested mel frame in a full audio chunk needs
roughly 40 samples beyond that chunk. Computing it immediately with artificial
right padding differs from computing the same frame once real samples arrive.
NVIDIA defers incremental frames until the complete FFT-frame right edge is
available. Changing only the window indexing would therefore not establish
streaming parity or justify claiming the frontend is fixed.

### Frame grid after streaming buffer trimming

Our streaming buffer retains
`(9 + chunk_mel_frames) * 160 + 400` samples. That count is 80 modulo the
160-sample hop. Removing the excess prefix can therefore move the buffer origin
by half a hop. The next frame index is obtained by integer-dividing the remaining
processed-sample count by 160, rather than preserving a global frame cursor.

Reproducing this exact index arithmetic with full audio chunks gives a next-frame
center 80 samples, or 5 ms, earlier than the original global grid after the first
trim, at all four supported chunk sizes. The first trims occur after 5, 4, 3,
and 3 chunks for 80, 160, 560, and 1120 ms respectively. This is a frame-grid
discrepancy; the calculation does not establish its word-error impact or imply
that drift accumulates without bound.

NVIDIA retains `audio_base_` and `total_mel_frames_produced_` independently of
buffer storage. Trimming does not restart the feature grid. This is the property
to replicate, regardless of whether we adopt its backend.

## Pipeline differences relevant to performance

| Area | Current Wordpipe path | NVIDIA path | What remains to measure |
| --- | --- | --- | --- |
| Feature extraction | Recompute a bounded overlapping audio buffer and rebuild its FFT plan | Produce new feature frames incrementally using global positions | Frontend stage share and end-to-end gain after parity is established |
| Predictor | Combined predictor and joint graph runs for each visited encoder frame or emitted label | Cache predictor output and candidate state across blanks | Decoder stage share and a split-graph implementation's overhead |
| Encoder joint projection | Combined decoder graph repeats projection work per decoder call | Project a chunk before decoding | Savings versus additional graph/session dispatch |
| Blank runs | One frame at a time | Evaluate remaining frames with the unchanged predictor and accept only the leading blank run | Benefit at concurrency one and small chunk sizes |
| State storage | ORT outputs copied into application-owned cache arrays | Backend-owned stream state and recurrent state banks | CPU copy cost separately from GPU transfer savings |
| Finalization | Worker pads a partial audio chunk and feeds three silence chunks | Runner pads a feature tail based on chunk geometry and drains it | Tail correctness, padding semantics, and total finalization cost |

Predictor caching is valid because a blank does not commit the candidate
recurrent state or change the preceding accepted token. NVIDIA commits a state
bank only after a nonblank token; our decoder likewise only updates its accepted
state on nonblank output. However, our combined ONNX graph prevents simply
skipping the predictor while retaining joint evaluation. A split graph requires
its own numerical and end-to-end tests.

NVIDIA's leading-blank evaluation is not batching independent requests. It
evaluates multiple encoder frames from one stream using the same predictor
state, then discards speculative results after the first emitted token. It can
therefore benefit concurrency one. Its opportunity is inherently limited when
a chunk contains just one encoder frame.

GPU-resident state is not evidence of the same magnitude of benefit on CPU.
Similarly, a cleaner graph does not prove lower latency after added dispatch
and allocation costs. Each proposed change needs an isolated ablation followed
by an actual worker benchmark.

NVIDIA also combines Q/K/V weights into one projection and uses strided output
views. Our earlier ORT QKV experiment was bit-exact but did not show a uniform
kernel win across chunks, and its estimated total gain was below five percent.
The native implementation is evidence that the structure is feasible, not new
evidence that adopting it in ORT would be profitable. NVIDIA precomputes relative
position projections; our current export already folds those projections, so
this is not additional headroom. Its ring-cache path is GPU-gated and should
not be counted as a demonstrated CPU optimization.

Several frontend choices already agree: constant rather than reflected edge
padding, symmetric rather than periodic Hann windows, preemphasis coefficient
0.97, and the additive log guard of 2^-24. NVIDIA's parameter named
`reflect_left` does not mean its current NeMo frontend actually uses reflected
padding. These should not be reported as discrepancies merely because names or
comments suggest otherwise.

The official artifact also includes a serialized FP32 `preprocessor.fb` tensor
with shape 128 by 257, and the runtime uses it. Its analytical fallback is not
the frontend used by this artifact. Treating that fallback's misleading
Slaney-style comment as evidence of a filterbank error would be incorrect.

## Correctness checks before any adoption

First establish frontend parity against NeMo, not against whichever quantized
backend produces the preferred transcript. Test impulses, tones, silence, short
clips, chunk boundaries, and long streams that cross multiple buffer trims.
Compare valid mel frames on their original global grid and explicitly test
preemphasis continuity and right-edge padding.

Then compare token/state traces for predictor caching and joint projection
changes, including blanks, symbol-cap exhaustion, reset, and finalization.
Check that token ties and exceptional nonfinite values have defined behavior.
Current source inspection is not a certification that NVIDIA's decoder is a
ground-truth implementation.

For acoustic accuracy, compare each backend or optimization against an
unquantized reference. Report the original quantization penalty and candidate
penalty on the same utterances; transcript equality with the original quantized
model is only supplementary evidence. A frontend fix changes the reference
pipeline too, so its accuracy ablation must be distinguished from the existing
kernel-only ablation.

## Native benchmark requirements

Use CPU execution, two threads, concurrency one, and all four supported right
contexts. NVIDIA's CLI benchmark explicitly sets batching off at concurrency
one. Load and warmup are outside its measured work; model startup must be
reported separately from transcription if compared with our worker.

Measure complete utterance processing including finalization and report
per-chunk latency separately. Do not compare a tail-excluding kernel number with
our worker's three-flush total. Repeat on identical audio, treat differences
within five percent as noise, and avoid overlapping timed runs with builds or
other inference processes.

The official NVIDIA benchmark uses an eight-thread desktop CPU and different
evaluation audio. Its published speed and error rates motivate a local test;
they are not performance predictions for our two-thread laptop configuration.

## Initial native measurements

On this host, the official Q8_0 model processes the existing 5.855-second clip
in real time at every supported chunk size. These measurements use two CPU
threads, concurrency one, three measured repetitions after warmup, and native
stream finalization. The transcript has zero errors against the 17-word
reference at every chunk size, but one utterance cannot establish accuracy.

| Chunk size | Mean complete utterance time | Real time factor | Mean push and drain time |
| --- | ---: | ---: | ---: |
| 80 ms | 3.168 s | 0.541 | 41.179 ms |
| 160 ms | 2.133 s | 0.364 | 53.331 ms |
| 560 ms | 1.411 s | 0.241 | 108.123 ms |
| 1120 ms | 1.213 s | 0.207 | 146.468 ms |

The complete-utterance column includes finalization; the push-and-drain column
does not. Some pushes do no encoder work while the frontend waits for a complete
window. Report `gguf-screening.json` contains the full distributions, commands,
binary and model hashes, and separate load/warmup times. This is an alternative
pipeline measurement, not an isolated ORT-versus-ggml kernel comparison. There
is no established universal speed win over our compact ORT model.

Native load reporting was approximately 25 ms, but warmup took 2.5 to 5.9 seconds
in this screening, including the configured warmup utterance. The load field
alone is not time until the engine is ready for a latency-sensitive request.

The downloaded GGUF contains 272 Q8_0 tensors, 30 FP16 tensors, and 351 FP32
tensors. Its pointwise convolution weights are Q8_0 and depthwise convolution
weights are FP16. Our exact depthwise optimization instead preserves the
original quantized arithmetic through an integer-valued FP32 kernel; these
approaches must not be described as numerically equivalent.

Five selected upstream tests passed: shared utilities, endpoint policy, ASR
postprocessing, decoders, and live transcript behavior. These are smoke checks
of the build and supporting logic, not NeMo model-parity certification.

## Accuracy screening results

The same forty-speaker dev-clean subset used for the optimization ablation
contains 326.525 seconds of audio and 914 reference words. Each backend resets
between utterances. These are total word errors, including substitutions,
deletions, and insertions; all columns use the same reference texts and scorer.

| Chunk size | Existing FP32 errors | Existing quantized errors | NVIDIA GGUF errors | GGUF within original error budget |
| --- | ---: | ---: | ---: | --- |
| 80 ms | 23 | 22 | 25 | No |
| 160 ms | 21 | 20 | 23 | No |
| 560 ms | 23 | 22 | 20 | Yes |
| 1120 ms | 23 | 22 | 25 | No |

The criterion compares each candidate's penalty relative to the existing
unquantized model with the original quantization penalty, rather than requiring
matching quantized transcripts. GGUF meets that criterion only at 560 ms on
this sample. Differences are small and this is not a population-level accuracy
conclusion. Both aggregate and mean utterance WER are retained in
`gguf-accuracy.json` with artifact hashes and commands.

The comparison includes different frontend and finalization behavior. It is
therefore an end-to-end backend screening, not evidence that Q8_0 quantization
alone causes these differences. The current FP32 path itself has the frontend
discrepancies documented above. A NeMo-faithful unquantized pipeline is needed
to attribute the remaining accuracy differences by component.

Native GGUF remains realtime on the full subset at every chunk size, with
audio-to-compute ratios of approximately 1.85, 2.70, 4.33, and 5.39 respectively.
These are single-pass corpus results, not a counterbalanced performance A/B.
One small frontend-diagnostic build overlapped part of this accuracy run; use
the separate three-repetition screening for uncontended native speed evidence.

## Chunk composability

The same downloaded GGUF was used for all four chunk sizes without conversion.
NVIDIA derives execution geometry from the right-context setting. However, its
encoder rejects changing that setting after its session has been built, so
one artifact does not imply an existing stream can switch context arbitrarily.
Backend adoption would need an explicit session lifecycle for context changes,
just as our ORT shape-specialized sessions have separate cache keys. Neither
approach requires the user to export models locally or publish per-chunk weights.

## Recommended next experiments

Establish centered-window and global-grid feature parity first, including the
lookahead and tail behavior. These are runtime/frontend changes and do not by
themselves require republishing model weights. Measure their accuracy impact as
separate ablations, then together, in both FP32 and compact precision.

After feature parity is established, profile the decoder and frontend stages
on the same corpus to decide whether predictor reuse, chunk-level encoder
projection, or incremental features has the largest remaining end-to-end
opportunity. Preserve the existing exact depthwise candidate as a separate
kernel-only change; its successful accuracy screening does not validate a
frontend rewrite. Do not bundle those effects into one unexplained speed or
accuracy result.
