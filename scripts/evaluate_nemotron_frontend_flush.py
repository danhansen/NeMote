#!/usr/bin/env python3
"""Frontend plus silence-flush accuracy and three-pipeline timing comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path('build/performance-audit')
ORT_LIB = '/home/dhansen/.local/libexec/wordpipe/python-venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.28.0'
MODELS = {'fp32': Path('build/dynamic-chunk-validation/fast-generic'),
          'compact': Path('build/dynamic-chunk-validation/compact-pointwise-generic')}
OLD = Path('target/container/release/wordpipe-parakeet-worker')
NEW = Path('target/container/release/wordpipe-frontend-flush-worker')
CORPUS = Path('target/container/release/wordpipe-corpus-flush-probe')
NATIVE = ROOT / 'nemo-speech-cpp/build/cpu-asr/bin/nemo-speech'
GGUF = ROOT / 'nemotron-speech-streaming-en-0.6b.q8_0.gguf'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifacts():
    paths = [OLD, NEW, CORPUS, NATIVE, GGUF, Path(__file__),
             Path('build/parakeet-rs-publish/src/nemotron_frontend.rs'),
             Path('build/parakeet-rs-publish/src/nemotron.rs'),
             ROOT / 'worker-probe/src/main.rs', ROOT / 'worker-probe/src/corpus.rs',
             ROOT / 'accuracy-manifest.json', ROOT / 'accuracy-ablation.json',
             ROOT / 'gguf-accuracy.json', Path(ORT_LIB)]
    for model in MODELS.values():
        config = json.loads((model / 'config.json').read_text())
        paths.extend(model / f for f in ['encoder.onnx', 'decoder_joint.onnx',
                                         'config.json', 'tokenizer.model',
                                         *config.get('shared_weight_files', [])])
    return {str(path): sha(path) for path in paths}


def save(report, output):
    output.write_text(json.dumps(report, indent=2) + '\n')


def accuracy(report, output):
    from score_benchmark_wer import load_eval_helpers
    helpers = load_eval_helpers()
    manifest = json.loads((ROOT / 'accuracy-manifest.json').read_text())
    baseline = json.loads((ROOT / 'accuracy-ablation.json').read_text())
    gguf = json.loads((ROOT / 'gguf-accuracy.json').read_text())
    assert baseline['status'] == gguf['status'] == 'complete'
    assert gguf['manifest_sha256'] == sha(ROOT / 'accuracy-manifest.json')
    references = {r['utt_id']: r for r in manifest}
    for c in (1, 2, 7, 14):
        for label, model in MODELS.items():
            command = [str(CORPUS), '--model-dir', str(model), '--wav',
                       str(ROOT / 'accuracy-manifest.json'), '--num-threads', '2',
                       '--chunk-samples', str(c*1280), '--flush-chunks', '3']
            env = dict(os.environ, ORT_DYLIB_PATH=ORT_LIB, WORDPIPE_POOL_EXPERIMENT='',
                       WORDPIPE_FRONTEND_TAIL='silence')
            env.pop('PARAKEET_STAGE_TRACE', None)
            with subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE) as process:
                for line in process.stdout:
                    row = json.loads(line)
                    reference = references[row['clip']]
                    assert sha(reference['audio']) == reference['sha256']
                    edits, words, wer = helpers.wer_stats(reference['reference'], row['text'])
                    row.update(label=label, chunk_frames=c, edits=edits, words=words,
                               wer=wer, reference=reference['reference'])
                    report['results'].append(row)
                    save(report, output)
                    print(f'accuracy c={c} {label} {row["clip"]}: {edits}/{words}', flush=True)
                stderr = process.stderr.read()
                if process.wait():
                    raise RuntimeError(stderr)
        current = {label: sum(r['edits'] for r in report['results']
                              if r['label'] == label and r['chunk_frames'] == c)
                   for label in MODELS}
        old = next(r for r in baseline['summary'] if r['chunk_frames'] == c)
        native = next(r for r in gguf['summary'] if r['chunk_frames'] == c)
        words = native['words']
        row = dict(chunk_frames=c, words=words, old_fp32_errors=old['models']['fp32']['edits'],
                   old_compact_errors=old['models']['original-quantized']['edits'],
                   new_fp32_errors=current['fp32'], new_compact_errors=current['compact'],
                   gguf_errors=native['edits'],
                   original_quantization_penalty=old['original_quantization_penalty'],
                   new_quantization_penalty=(current['compact']-current['fp32'])/words,
                   within_original_quantization_budget=
                   current['compact']-current['fp32'] <=
                   old['models']['original-quantized']['edits']-old['models']['fp32']['edits'])
        report['summary'].append(row)
        save(report, output)
        print('SUMMARY', json.dumps(row), flush=True)


def speed(report, output, repetitions):
    import benchmark_parakeet_variant as bench
    audio = Path('build/dynamic-chunk-validation/audio/1272-128104-0000.wav')
    report['audio_sha256'] = sha(audio)
    # Full process startup is separate from timed transcription. Native warmup
    # is reported, not silently included in its model-load field.
    labels = ['old-ort', 'new-ort-flush', 'nvidia-gguf']
    os.environ['WORDPIPE_POOL_EXPERIMENT'] = ''
    os.environ.pop('PARAKEET_STAGE_TRACE', None)
    for c in (1, 2, 7, 14):
        for repetition in range(repetitions):
            order = labels[repetition % 3:] + labels[:repetition % 3]
            if (repetition // 3) % 2:
                order.reverse()
            for label in order:
                if label == 'nvidia-gguf':
                    command = [str(NATIVE), 'bench', 'asr', str(audio), '--model', str(GGUF),
                               '--mode', 'stream', '--device', 'cpu', '--asr.backend.threads', '2',
                               '--asr.streaming.rnnt_right_context', str(c-1), '--chunk-ms', str(c*80),
                               '-c', '1', '-n', '1', '--warmup', '1', '--json']
                    process = subprocess.run(command, check=True, capture_output=True, text=True)
                    result = json.loads(process.stdout)
                    run = result['runs'][0]
                    row = dict(label=label, chunk_frames=c, repetition=repetition,
                               decode_seconds=run['latency_ms']['mean']/1000,
                               load_seconds=result['load_ms']/1000,
                               warmup_seconds=result['warmup_ms']/1000,
                               details=result, command=command)
                else:
                    sys.argv = ['benchmark', str(MODELS['compact']), '--wav', str(audio),
                                '--worker', str(OLD if label == 'old-ort' else NEW),
                                '--num-threads', '2', '--chunk-samples', str(c*1280),
                                '--flush-chunks', '3', '--min-mem-available-gb', '3',
                                '--ort-dylib', ORT_LIB]
                    settings = bench.parse_args()
                    result = bench.run_once(settings, label, MODELS['compact'], repetition)
                    row = dict(label=label, chunk_frames=c, repetition=repetition,
                               decode_seconds=result['metrics']['decode_seconds'],
                               load_seconds=result['load_seconds'], details=result)
                report['results'].append(row)
                save(report, output)
                print(f'speed c={c} rep={repetition} {label}: {row["decode_seconds"]:.4f}s', flush=True)
        rows = [r for r in report['results'] if r['chunk_frames'] == c]
        medians = {label: statistics.median(r['decode_seconds'] for r in rows if r['label'] == label)
                   for label in labels}
        paired = {label: [next(r['decode_seconds'] for r in rows
                               if r['label'] == label and r['repetition'] == i) /
                         next(r['decode_seconds'] for r in rows
                              if r['label'] == 'old-ort' and r['repetition'] == i)
                         for i in range(repetitions)] for label in labels[1:]}
        summary = dict(chunk_frames=c, medians=medians, paired_ratios_vs_old=paired,
                       paired_median_ratios_vs_old={k: statistics.median(v) for k,v in paired.items()})
        report['summary'].append(summary)
        save(report, output)
        print('SUMMARY', json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['accuracy', 'speed'], required=True)
    p.add_argument('--repetitions', type=int, default=6)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.repetitions < 1:
        p.error('repetitions must be positive')
    report = dict(status='running', phase=args.phase, artifacts=artifacts(),
                  concurrency=1, threads=2, flush_chunks=3, results=[], summary=[],
                  limitations=['Accuracy reuses artifact-bound old ORT and GGUF evaluations on the identical 40-speaker sample.',
                               'Different backend quantizations and EOF policies; not an isolated kernel comparison.',
                               'Speed screen uses one utterance and excludes startup; differences within 5% are noise.'])
    save(report, args.output)
    if args.phase == 'accuracy':
        accuracy(report, args.output)
    else:
        speed(report, args.output, args.repetitions)
    for path, fingerprint in report['artifacts'].items():
        assert sha(path) == fingerprint, f'artifact changed: {path}'
    report['status'] = 'complete'
    save(report, args.output)


if __name__ == '__main__':
    main()
