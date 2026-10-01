# Nemotron frontend parity

The local parakeet-rs candidate now reproduces the checkpoint's NeMo frontend
and cache-aware feature chunking, including packet boundaries and end-of-stream
lengths. Frame counts, masked padding, and chunk lengths agree exactly. Features
are bit-identical across Rust packetizations; comparisons with NeMo retain small
cross-library FP32 FFT/reduction differences documented below.

This is a frontend result, not a certification of every acoustic-model or
decoder operation. The forty-speaker recognition ablation is complete; it does
not establish an accuracy improvement. A native NeMo comparison of the clip
with the missing final word reproduces that omission at the same chunk sizes.
The application still pins the published fork; the candidate and its new explicit
finish API are local and have not been published or enabled in production.

## Reference and tested contract

The reference is `AudioToMelSpectrogramPreprocessor` instantiated directly from
the existing English `.nemo` checkpoint's configuration and put into evaluation
mode. No learned acoustic-model weights are loaded for feature verification.
The chunk reference uses NeMo's actual `CacheAwareStreamingAudioBuffer` iterator,
with geometry derived by NeMo's `ConformerEncoder.setup_streaming_params` and
causal `ConvSubsampling` methods, without constructing learned layers.

| Detail | Tested behavior |
| --- | --- |
| Audio | Mono FP32, 16000 Hz; no clipping, resampling, or normalization in the frontend |
| Evaluation dither | Disabled by NeMo despite the checkpoint's training value of 1e-5 |
| Preemphasis | First real sample unchanged; subsequent samples use x[t] minus 0.97 times x[t-1] |
| Preemphasis across packets | Previous raw sample retained; packet boundaries do not reset the filter |
| True audio length | Raw samples beyond the declared length are zeroed before STFT |
| FFT | 512-point transform, power spectrum, constant edge padding |
| Window | Symmetric Hann, 400 samples, centered at offset 56 inside the FFT frame |
| Window arithmetic | Torch's rounded angular increment multiplied by each FP32 index |
| Frame grid | Global centers spaced 160 samples apart; buffer trimming never restarts it |
| Streaming availability | Emit a frame only once its 400-sample window has real right context |
| Mel basis | 128 Slaney bands, Slaney normalization, 0 to 8000 Hz |
| Log | Natural logarithm after adding 2^-24, not clipping or log10 |
| Normalization and splicing | Checkpoint normalization NA, frame splicing one |
| Valid length | Floor of real sample count divided by 160 |
| Final masked frame | One extra STFT frame, set to literal zero after feature extraction |
| Pre-encode cache | Nine feature frames; missing initial history is literal zero |
| Supported contexts | One implementation supports 1, 2, 7, and 14 encoder frames |
| Chunk lengths | Actual feature length is separate from the fixed transport tensor shape |
| Short final slices | Match NeMo's causal subsampling thresholds of one frame initially and eight thereafter |
| Reset and finish | Reset clears all stream state; finish is idempotent; new audio after finish requires reset |

The worker's existing input contract already requests mono capture at 16000 Hz,
rejects WAV files with a different rate, converts PCM16 by dividing by 32768,
and averages WAV channels. This work does not introduce a resampler or claim
parity with arbitrary external file-decoding/resampling libraries.

## Changes in the local runtime

`nemotron_frontend.rs` extracts features incrementally using global sample and
frame positions. It keeps the raw predecessor needed for preemphasis when
trimming storage. Across the test fixtures, retained raw storage peaks at 398
samples after a push; memory does not grow with stream duration. The FFT plan,
window, and scratch arrays are reused.

`NemotronMelChunker` retains nine feature frames between encoder calls, including
the partially populated history at the second 80 ms chunk. The old offline
transcription loop omitted that history when its processed frame index was less
than nine; offline transcription now uses the same streaming frontend and
chunker as streaming transcription.

The local `Nemotron::finish` drains the true feature tail. It does not convert a
partially filled audio packet into a full valid silence packet or synthesize
three extra audio chunks. Feature padding is not equivalent to silent audio:
literal zero features differ from the log-mel value of raw silence.

The diagnostic corpus caller supplies each final audio slice at its real length
and invokes finish. Application adoption still needs the equivalent worker
caller change together with a new fork pin; the existing production worker is
not being described as fixed merely because the library candidate passes.

## Verification coverage

The feature gate exercises 54 signals across six packetization patterns,
producing 324 comparisons. Cases include empty streams, boundaries around the
hop/window/FFT and all four chunk sizes, impulses at boundary positions, silence,
DC, low and high tones, Nyquist input, amplitude extremes, twenty seconds of
seeded noise, and the existing speech fixture. Additional checks inject nonzero
junk beyond the true length in padded NeMo input storage.

All feature counts, terminal masking, and chunk-length values match their
references. Rust features are bit-identical for the same waveform under every
tested packetization. The chunk gate compares the actual runtime chunker's
outputs against NeMo's iterator, including its partial final slices and literal
transport padding.

Rust unit tests cover arbitrary packetization, bounded raw storage, reset,
idempotent finish, short pre-encode history, invalid feature widths, and transport
lengths. Python unit tests prove that the numerical gate does not apply its
spectral-null exception to ordinary features or to excessive energy/log errors.

## Numerical differences and acceptance

Matching Torch's window arithmetic reduced the maximum coefficient discrepancy
from 3.87e-7 to 5.96e-8. The source of this arithmetic order is Torch's
[Hann and Hamming implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorFactories.cpp).

The final candidate's speech fixture has a maximum absolute log-mel difference
of 0.000437 from NeMo. The largest tested difference, 0.002756, occurs for a
full-amplitude Nyquist signal in a nearly empty spectral band. That comparison
involves approximately 2.44e-8 reference mel energy and only 2.32e-10 absolute
energy difference.

Supplying NeMo's exact window and mel coefficients to the Rust diagnostic leaves
the Nyquist and full-amplitude 4 kHz discrepancies unchanged. This isolates the
remaining differences from window/filterbank generation; they come from the
different FP32 FFT/reduction implementations. It is not evidence of a remaining
frame shift, nor a claim of bitwise equality between libraries.

The acceptance gate keeps an absolute log-mel tolerance of 0.001 for ordinary
features. A measured rounding exception requires all three conditions:

- Guarded reference energy no greater than 16 times 2^-24.
- Absolute guarded-energy difference no greater than 5e-10.
- Absolute log difference no greater than 0.005.

Only 24 cells across the 324 repeated comparisons use this exception. Padding,
frame counts, lengths, and packetization equivalence have no tolerance exception.

## Reproduction and evidence

The verifier is `scripts/verify_nemotron_frontend.py`. It requires the existing
NeMo development environment, not an application user's machine. Its JSONL
diagnostic reads the local fork's actual frontend and chunker implementation.
Run it against the checkpoint, compiled probe, frontend source, and an optional
16 kHz PCM16 speech fixture. The report binds checkpoint, source, binary,
verifier, and oracle files by SHA256.

Detailed results are retained in
`build/performance-audit/frontend-parity-complete.json`. The separate numerical
control report is `frontend-numerical-corners.json`; its exploratory measurement
limit is not the acceptance criterion. The incomplete accuracy run using the
earlier window arithmetic is retained as `frontend-accuracy-unscaled-window.json`
and explicitly marked superseded.

The frontend change needs no model-weight export or user-local NeMo installation.
Both current generic encoder precisions already expose the real-length input
needed for short final chunks. No claim is made yet about end-to-end speed
improvement. Recognition results and their limits follow.

## Recognition screening

The frozen candidate was evaluated on the same forty-speaker, 914-word sample
as the earlier baseline, with two CPU threads, concurrency one, identical model
files, and no kernel or pool changes. The worker, frontend, and runtime hashes
still match their recorded values after the run. Feature extraction and
finalization changed together; this is not a quantization-only experiment.

| Chunk size | Old FP32 errors | Candidate FP32 errors | Old quantized errors | Candidate quantized errors |
| --- | --- | --- | --- | --- |
| 80 ms | 23 | 26 | 22 | 26 |
| 160 ms | 21 | 24 | 20 | 26 |
| 560 ms | 23 | 22 | 22 | 24 |
| 1120 ms | 23 | 25 | 22 | 23 |

The empirical quantization penalty relative to each frontend's FP32 baseline
does not meet the original penalty at 80, 160, or 560 ms; it meets it at 1120 ms.
These small error counts do not establish population-level improvement or
regression. They do not justify promotion under the current screening gate.
The full report is `build/performance-audit/frontend-accuracy.json`.

## Native NeMo final word comparison

The affected utterance, `6313-66129-0007`, ends with “I am hungry too.” The
existing checkpoint was restored into native NeMo FP32, retaining its
`greedy_batch` decoder and ten-symbol limit. All four contexts were tested with
the actual cache-aware streaming iterator and persistent RNNT hypotheses.

| Chunk size | Native NeMo with initial padding | Current FP32 runtime |
| --- | --- | --- |
| 80 ms | I am afraid to admit that I am hungry | Same |
| 160 ms | I am afraid to admit that I am hungry | Same |
| 560 ms | I am free to admit that I am hungry too. | Same |
| 1120 ms | I am free to admit that I am hungry too. | Same |

The native iterator with initial padding, its fixed-transport equivalent, and a
control disabling final `keep_all_outputs` all produce the same transcript for
each context. Without initial padding, native NeMo still omits “too” at 80 and
160 ms; the 160 ms prefix changes from “afraid” to “free.” Initial padding is an
explicit NeMo-supported policy choice, not a claim that every streaming policy
must have the same transcript.

At 80 and 160 ms, the padded iterator leaves six feature frames below its
eight-frame threshold; its last emitted chunk does not set `keep_all_outputs`.
At 560 and 1120 ms, the final slice has 31 input frames including history,
produces three valid encoder frames, and sets `keep_all_outputs`. The unpadded
160 ms control consumes its tail but still omits the word, so leftover frames
alone are not a demonstrated explanation.

Native features were replayed through the existing generic FP32 ONNX encoder.
Encoder shapes and valid lengths match exactly at every step. Maximum encoder
differences across the four contexts range from 5.4e-7 to 1.3e-6. The ONNX
decoder produces exactly the native RNNT token sequence when given either
native or ONNX encoder outputs. Thus this particular missing word is reproduced
by the complete native pipeline, with no observed encoder or decoder port
mismatch. This is a one-utterance result, not complete model certification.

Reproduction scripts are `scripts/trace_nemotron_native_stream.py` and
`scripts/compare_nemotron_native_trace.py`. Detailed per-step lengths, native
hypotheses, token decisions, hashes, and replay results are retained under
`build/performance-audit/native-tail-trace/`.

A separate native control restores the old worker's raw-audio tail policy:
pad the final packet to its full chunk size and append three silent packets.
This adds 4240 samples at 80 ms and 9360 at 160 ms for this utterance. Native
NeMo then emits “I am afraid to admit that I am hungry too” at both sizes,
where the same native padded streaming path without extra audio omits “too.”
The report is `build/performance-audit/native-tail-flush-control/native.json`.

That intervention establishes that the silence-tail policy changes the last
word in this clip. It does not establish how much silence is necessary, whether
other utterances benefit, or whether a different finalization policy should be
adopted. True-EOF parity and preserving the old worker's recognition behavior
are distinct requirements. There is no observed incomplete NeMo-port explanation
for this omission; wider accuracy testing should hold finalization policy fixed
when measuring the frontend alone.
