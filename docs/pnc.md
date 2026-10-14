# Optional English punctuation and capitalization

The **Restore Punctuation and Capitalization** switch enables a separate
NeMo-Speech.cpp token classifier. It is experimental and off by default;
it is not an LLM cleanup step. Expand **Text → Restore Punctuation and Capitalization**
to select **BERT** (higher accuracy) or **DistilBERT** (faster).

While dictation is stopped, use the install button for **English PnC — BERT**
or **English PnC — DistilBERT** inside that expandable row. The section can be opened
while restoration is off. Only installed models can be selected; download buttons
remain available for missing models. This explicitly downloads the
chosen checkpoint directly from NVIDIA and converts it locally using the
pinned upstream converter. Download progress and preparation/conversion stages
appear on the model row, in Advanced diagnostics, and in service logs. Installation does not enable PnC or
change the selected model: choose the model and enable the switch when ready.
Dictation never downloads or converts PnC models. If the selected model is
missing, install it in preferences or disable the switch before dictating.

New models are stored under `<model-root>/pnc`, shared between ASR models.
Existing per-ASR PnC downloads are reused in place. Normal starts load the
installed GGUF without invoking the installer or importing PyTorch. The
installer verifies cache receipts and checksums when invoked.

Alternatively, enter a compatible English `pnc` GGUF path in **Advanced →
Companion Model Overrides → Punctuation Model (.gguf)**;
a nonempty override takes precedence over the model choice and bypasses setup.
The model is loaded once per worker, sharing the recognizer's CPU backend.
PnC adds model memory and postprocessing latency only when enabled.

## Timing

Immediate partials remain the acoustic model's output. PnC runs after optional
ITN at the same points as our existing postprocessing:

- Endpointing disabled: on Stop.
- Preview: on pauses, reprocess the accumulated raw session transcript without
  committing it; retain that formatted prefix while subsequent partials arrive.
- Commit: on pauses, postprocess and commit each segment.

ITN and PnC are independent switches. English is required. Auto language
selection is allowed with the English model; with the multilingual model,
select English explicitly. This avoids applying an English classifier to a
mixed-language accumulated preview.

Sentence-edge punctuation is removed before ITN. After ITN, PnC receives a
lowercase classification input with punctuation removed. A one-to-one token map
restores written-form numbers, acronyms/mixed-case units, and symbol-bearing tokens, including currency,
decimal separators, emails, URLs and paths, rather than emitting the stripped
classification input. Unexpected lexical changes are rejected. This safeguard
does not guarantee punctuation quality, and stripped addresses can provide less
useful context to the classifier. PnC predicts commas, periods and question marks;
it does not restore every kind of punctuation that Nemotron can emit.

## Model preparation

No checkpoint weights or converted GGUFs are bundled or rehosted. Explicit
setup downloads `punctuation_en_bert` (406 MB) or `punctuation_en_distilbert`
(245 MB), version `1.0.0rc1`, directly from NVIDIA NGC. Downloaded source bytes
and the encoder configuration must match pinned SHA-256 hashes before they
reach the converter. Resulting Q8 GGUFs are approximately 139 MB and 94 MB,
respectively. Each variant has its own cache receipt and checksum.

First setup needs Python's venv/ensurepip support (`python3-venv` on
Debian/Ubuntu), approximately 1.25 GiB available RAM and at least 2 GiB free on
each filesystem used for models and the conversion environment. This is a
conservative preflight, not a guarantee against every resource limit. On this
host conversion alone took about 5 seconds, with measured peak RSS of 783 MiB
for BERT and 574 MiB for DistilBERT; downloading and environment setup add time.

CPU-only PyTorch, NumPy, GGUF and PyYAML install into a private, versioned venv
under `$XDG_CACHE_HOME/nemote/pnc-export` (default `~/.cache/nemote/pnc-export`),
not the system Python. Tested environments used about 840–977 MB installed.
PyYAML's small pure-Python wheel is bundled by the release builder; clients use
binary wheels only and do not need a compiler, even if the system Python is
newer than PyYAML's published platform wheels.
These dependencies are installed only when explicit PnC conversion is needed;
normal dictation does not require them. The original checkpoint and temporary
conversion files are deleted after setup, including on failure. Existing model
files are not replaced unless conversion and output validation succeed.

For source builds, `scripts/build-nemo-worker` packages the converter code;
it can also be staged explicitly with
`python3 scripts/package_pnc_converter.py --sdk /path/to/NeMo-Speech.cpp`.
Developers may set `NEMOTE_PNC_PYTHON` to an existing interpreter with the exact
conversion dependencies pinned in `src/nemote/pnc_install.py`.

### Distribution status

Publication of our converted checkpoints remains on hold pending confirmation
of redistribution rights. Checked on 2026-10-12: the
official NGC model card for version `1.0.0rc1` identifies the
[NGC Terms of Use](https://ngc.nvidia.com/legal/terms) as its license unless
another agreement is clearly specified. The checkpoint archive contains no
separate license file, and the model metadata does not identify an overriding
agreement.

The legal page's
[published terms asset](https://ngc.nvidia.com/legal/consolidated-tou.json)
(dated April 7, 2025) grants use without sublicensing in section 4 and restricts
modification and redistribution, including publication in public repositories,
without express authorization in section 5(e). The NeMo toolkit's code license
does not establish redistribution rights for these weights. We have not
published the GGUF to Hugging Face. Direct download and local conversion avoid
our redistribution of the weights, but do not establish permission to modify
them: users remain responsible for the applicable model terms. A successful
conversion and upstream conversion documentation are not a legal determination.

### Conversion

Using a NeMo-Speech.cpp checkout with its conversion dependencies installed:

```sh
python3 convert_model.py current-layout-pnc.nemo --architecture pnc \
    --outfile pnc-bert-en.q8_0.gguf --outtype q8_0
```

The official NGC `punctuation_en_bert` version `1.0.0rc1` uses an older archive
layout: it embeds label IDs in YAML and omits the encoder JSON. Our developer
adapter repackages those metadata files without changing BERT weights, then invokes
the upstream converter. For that checkpoint, from the NeMote checkout:

```sh
curl -fL https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f7235943/config.json \
    -o encoder_config.json
python3 scripts/export_nemo_pnc.py punctuation_en_bert.nemo \
    --sdk /path/to/NeMo-Speech.cpp --encoder-config encoder_config.json \
    --outfile pnc-bert-en.q8_0.gguf
```

This requires PyTorch, NumPy, GGUF and PyYAML in the conversion environment, but
does not require the Python NeMo toolkit or any changes to the upstream runtime.

The unmodified upstream converter handles BERT tensor names. For the supported
NVIDIA DistilBERT checkpoint, our adapter maps its six attention/FFN blocks to
the same post-LayerNorm SDK graph, preserving learned tensors and adding a
zero segment-embedding tensor (DistilBERT has no segment embeddings). No SDK
runtime patch is needed. This is an integration-specific compatibility adapter,
not a claim that upstream supports arbitrary DistilBERT checkpoints. The
existing SDK uses LayerNorm epsilon `1e-5`, versus `1e-12` in the source BERT and
DistilBERT implementations; quantization and ggml operations also prevent a
bit-exact inference claim. A small five-sentence adapter smoke test preserved
every learned DistilBERT tensor and matched the unquantized source checkpoint's
punctuation/capitalization labels on real word tokens. It is not a broad accuracy
evaluation. Both variants passed the WAV integration checks in all three
endpoint modes, with and without ITN, preserving transcript words and immediate
partials.

The initial local smoke-test export uses the pinned SDK revision
`4c101bc7113f49101a3e11d2c994c519f41939f6`. Its source checkpoint SHA-256 is
`d69e87388e9162f86840cadc2d4efee45588de4eaf92b542d994f0eccd73e7ae`.
The Q8 GGUF is 138,875,840 bytes, with SHA-256
`131255fe1628be6368af1f1a82fe61ea214218c76be0684c8a0f72afceac011b`.
Weights are kept out of Git; the developer test artifact is in
`build/pnc/pnc-bert-en.q8_0.gguf`.

A JFK WAV smoke test passed in all three endpoint modes, with and without ITN:
initial partials matched the native output, and final transcript words were
preserved. Whole-session processing produced sensible sentence punctuation;
pause-and-commit processing fragmented the sample more aggressively. This is
integration evidence, not a demonstration that BERT improves Nemotron's
punctuation accuracy in general.

The model must use the SDK's `pnc` architecture and its BERT/two-head format.
Other runtimes' punctuation GGUFs are not interchangeable merely because their
file extension is the same.

For a WAV-driven check with a built worker:

```sh
nemote-nemo-worker --model-dir /path/to/asr/model.gguf --wav sample.wav \
    --language en-US --pnc --pnc-model /path/to/pnc-bert-en.q8_0.gguf
```

Add `--itn` to test the combined chain; use `--endpoint-mode preview` or
`--endpoint-mode commit` to test pause processing.
