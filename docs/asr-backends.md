# Wordpipe ASR backends

Wordpipe retains Parakeet as its default backend and adds NeMo-Speech.cpp as a separate C++ worker. The application boundary is the existing newline-delimited JSON worker protocol. No Rust FFI, audio IPC, model conversion, or upstream source fork is needed for NeMo inference.

## Ownership and lifecycle

The Rust service owns configuration, worker lifetime, D-Bus signals, and text insertion coordination. Each worker owns microphone capture, its audio queue, model loading, decoder state, and EOF handling. The service launches only one selected worker, and each worker runs one recognition session at a time.

The C++ worker uses upstream `Recognizer` and `RecognitionStream`, plus miniaudio capture. A recognizer stays loaded across sessions; each start creates a fresh stream. Stop first closes capture, drains the queued real audio, then invokes native `finish()`. It does not append silence or pad the final audio packet. Parakeet's existing frontend and silence-flush behavior remain unchanged.

The upstream C++ API is version-dependent. Headers and libraries must be built together from revision `4c101bc7113f49101a3e11d2c994c519f41939f6`. Wordpipe does not promise binary compatibility with an arbitrary separately installed C++ SDK.

## Worker protocol

Commands on stdin are JSON objects with a `command` field: `start`, `stop`, `shutdown`, and `set_language`. The last command also requires `language` and is accepted only while idle. Stdin EOF stops and joins any session before releasing the model.

Stdout contains only JSON events. Startup emits `loading_model`, `model_loaded`, then `ready`. A live session emits `listening`, complete `partial` transcript snapshots, periodic `stats`, an optional nonempty `commit`, then `stopped`. Failures emit `error` with a `message`. Diagnostics and library logs belong on stderr.

Partial events carry complete transcript snapshots rather than insertion deltas. The C++ worker combines completed upstream utterances with the current interim snapshot and accounts for late punctuation. Automatic punctuation is enabled to retain model-native formatting in final results.

The supported Nemotron greedy RNN-T models accumulate tokens and append their decoded text. Native live text insertion therefore remains available through the existing `TextDelta` interface. The service extracts only a suffix of the previous snapshot; an unexpected non-prefix native snapshot is not inserted again as a whole transcript. The integration check records any partial or final text that fails this prefix contract. NeMo's general ability to host revising decoders does not imply that these Nemotron models revise tokens.

Both workers accept `--model-dir`, `--num-threads`, `--sample-rate`, `--language`, `--chunk-samples`, `--input-device`, `--list-input-devices`, and `--wav`. Backend-specific arguments are not sent to the other worker. NeMo's `--wav-repeat` diagnostic option exercises fresh streams on one loaded model.

## Models and selection

The `nemo-speech` backend exposes `nemo-q8` and `nemo-q8-english` presets. These use NVIDIA's official Q8_0 GGUFs, not the existing ONNX INT8 weights and not FP32. Model downloads are pinned by repository revision, file size, and SHA256 and are installed atomically only after verification. Users do not need NeMo, PyTorch, or local model export.

The multilingual file is installed at `MODEL_ROOT/nemotron-nemo-q8/model.gguf`; the English file at `MODEL_ROOT/nemotron-nemo-en-q8/model.gguf`. A single file supports 80, 160, 560, and 1120 ms chunks. Chunk size and CPU threads are passed to upstream runtime configuration; batching and GPU execution are disabled in this initial integration.

Changing backends selects a compatible model profile and resets the input device to the system default because device identifiers belong to the capture backend. The service enumerates NeMo devices through its worker rather than reusing CPAL indexes. Existing Parakeet models and settings remain available when switching back.

Download an English model with:

```sh
wordpipe-model-install --profile nemo-q8 --model-family english
```

Then choose NeMo-Speech.cpp in GNOME preferences and select the installed English Q8 model. The same installer accepts `--model-family multilingual` and accepts a local official GGUF via `--source` with the same checksum verification.

## Build and verification

Run `bash scripts/build-nemo-worker`. It fetches the pinned upstream checkout when absent, applies NVIDIA's bundled ggml patch series, builds CPU ASR, and builds `wordpipe-nemo-worker`. CMake, a C++17 compiler, and SentencePiece development files are required. The development image supplies these dependencies. The reproducible build links its system SentencePiece library rather than reusing host-built archives from a research checkout. Existing matching builds can be selected with `WORDPIPE_NEMO_SOURCE`, `WORDPIPE_NEMO_BUILD`, and `WORDPIPE_NEMO_WORKER_BUILD`.

Release packaging includes both workers, the native shared libraries, the SentencePiece runtime dependency, and license notices. The NeMo executable resolves packaged libraries relative to itself. For development, `WORDPIPE_NEMO_WORKER` overrides its executable path without changing the saved Parakeet worker path.

Run `ctest --test-dir build/nemo-worker --output-on-failure` for transcript snapshot, session lifecycle, and CLI smoke tests. Rust service tests cover backend selection, model paths, chunk capabilities, persistence, filtering, and insertion deltas. Python tests cover verified model installation and preservation of a good installation after a failed replacement. Setting `WORDPIPE_NEMO_TEST_WORKER` and `WORDPIPE_NEMO_TEST_MODEL` enables real-SDK protocol tests for language validation, capture-error recovery, shutdown, and stdin EOF.

`scripts/verify_nemo_worker.py` checks the real worker against the recorded upstream GGUF corpus with identical model and audio hashes. It exercises all four contexts and two streams per clip on one recognizer. This is an integration-parity check, not evidence that Q8 equals FP32 accuracy on a wider population. Timing includes native EOF processing; cold and second-session metrics are recorded separately. Live microphone and GNOME interaction still require a desktop smoke test.

The initial 40-clip comparison matched upstream's normalized transcripts in all 160 clip/context cases, with identical outputs across two sessions per case. A subsequent check of the final worker on eight clips per context found no partial or final prefix violations. `scripts/benchmark_nemo_worker.py` separately compares complete streaming request latency with upstream's CLI at concurrency one, two CPU threads, and one warmup request; run it without overlapping builds or inference jobs.

The isolated five-repetition timing screen found worker request latency within 5% of upstream at every context, which is within the accepted noise band. This used the same existing SDK build for both executables; it does not benchmark a fresh portable release build. The multilingual Q8 model also matched upstream on an English smoke clip at all four contexts with both English and default-language selection, without emitted-text prefix violations. Other-language accuracy and live desktop behavior were not measured in this integration check.
