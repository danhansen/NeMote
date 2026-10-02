# Native runtime and protocol

NeMo-Speech.cpp is Wordpipe's sole inference runtime. The SDK and headers are
built together from pinned revision `4c101bc7113f49101a3e11d2c994c519f41939f6`
and ggml revision `c03b4e2bcece5134827881af90242086daf75be5`. Our integration
patches add runtime CPU plugins, strict boosting-tokenizer validation, and
pause-triggered preview normalization. Recognition stays in C++.

## Models

Preferences show **English (lower WER)** and **Multilingual**. Both are NVIDIA's
official Q8_0 GGUFs, with immutable revisions and SHA256 checks in
`src/wordpipe/nemo_models.py`. Users need neither NeMo/PyTorch nor local export.
The internal compatibility profile remains `nemo-q8`. Existing directories
`nemotron-nemo-en-q8` and `nemotron-nemo-q8` are retained to reuse installations.

```sh
wordpipe-model-install --model-family english
wordpipe-model-install --model-family multilingual
```

One file supports 80, 160, 560, and 1120 ms chunks. Two CPU threads and 560 ms
are defaults. CPU plugins are selected by supported instructions, not CPU model
names. CUDA/Vulkan are not enabled in the shipped worker.

## Worker protocol

Commands on stdin are JSON objects: `start`, `stop`, `shutdown`, and
`set_language` (also requires `language`). Stdout carries only JSON events;
library diagnostics go to stderr.

Startup emits `loading_model`, `model_loaded`, and `ready`. A session emits
`listening`, complete `partial` snapshots, `stats`, optional `commit` snapshots,
and finally `stopped`. Errors carry `message`. The service forwards these through
the existing D-Bus interface, preserving session IDs and monotonically increasing
sequence numbers. Incompatible snapshot rewrites do not generate insertion deltas.

Stop closes capture, drains real queued audio, and calls upstream `finish()`.
There are no synthetic silence-flush chunks. With endpoint commit mode, multiple
cumulative commits can occur within one dictation session.

ITN, VAD, and boosting are optional. Preview endpoint mode normalizes the current
full utterance on pauses without resetting recognition; following speech can
replace that provisional formatting. EOF normalizes the full final utterance.
VAD masks features and does not itself imply endpointing. Tokenizer and VAD
companions download automatically unless explicit path overrides are configured.

## Build and checks

`scripts/build-nemo-worker` builds the SDK and worker; `scripts/build-service`
builds the independent GLib/GIO service. `scripts/run-build-pipeline test` runs
native tests, Python downloader tests, JavaScript insertion/preferences tests,
schema/contract checks, and an isolated D-Bus lifecycle test. Set the SDK/model
fixture environment variables described in `tests/test_nemo_worker.py` to run
real model tests. Live GNOME/microphone behavior still requires desktop testing.
