#!/usr/bin/env python3
"""Isolated CPU build A/B: same GGUF, audio, threads, contexts, and session lifecycle."""
import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def sdk_identity(worker):
    cache = worker.parent / 'CMakeCache.txt'
    if not cache.is_file():
        raise SystemExit(f'Matching build metadata missing: {cache}')
    key = 'NEMO_SPEECH_LIBRARY_DIR:PATH='
    directory = next((Path(line[len(key):]) for line in cache.read_text().splitlines()
                      if line.startswith(key)), None)
    if directory is None:
        raise SystemExit(f'SDK library directory missing: {cache}')
    return {str(path): sha(path) for path in sorted(directory.glob('*.so*'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='append', required=True, help='LABEL=PATH; first is the baseline')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 99:
        parser.error('repetitions must be in [1,99]')
    workers = {}
    for item in args.worker:
        label, path = item.split('=', 1)
        if not label or label in workers:
            parser.error('worker labels must be nonempty and unique')
        workers[label] = Path(path)
    report = {'status': 'running', 'threads': 2, 'concurrency': 1, 'batching': False,
              'warmup': 1, 'repetitions': args.repetitions, 'compute_backend': 'cpu',
              'model_sha256': sha(args.model), 'wav_sha256': sha(args.wav),
              'workers': {label: {'path': str(path), 'sha256': sha(path),
                                  'sdk_libraries': sdk_identity(path)} for label, path in workers.items()},
              'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    save()
    for context in (1, 2, 7, 14):
        labels = list(workers)
        if context in (2, 14):
            labels.reverse()
        row = {'chunk_ms': context * 80, 'builds': {}}
        for label in labels:
            path = workers[label]
            if sha(path) != report['workers'][label]['sha256']:
                raise SystemExit('Worker changed during measurement')
            if sdk_identity(path) != report['workers'][label]['sdk_libraries']:
                raise SystemExit('SDK libraries changed during measurement')
            command = [str(path), '--model-dir', str(args.model), '--wav', str(args.wav),
                       '--device', 'cpu', '--num-threads', '2', '--chunk-samples', str(context * 1280),
                       '--wav-repeat', str(args.repetitions + 1)]
            proc = subprocess.run(command, capture_output=True, text=True, check=True, timeout=240)
            events = [json.loads(line) for line in proc.stdout.splitlines()]
            commits = [e for e in events if e['event'] == 'commit']
            if len(commits) != args.repetitions + 1 or len({e['text'] for e in commits}) != 1:
                raise SystemExit('Repeated-stream transcript changed')
            timing = [e['data']['elapsed_seconds'] for e in commits[1:]]
            row['builds'][label] = {'seconds': timing, 'median_seconds': statistics.median(timing),
                                    'text': commits[0]['text'], 'command': command,
                                    'model_loaded': next(e['data'] for e in events if e['event'] == 'model_loaded')}
        baseline = row['builds'][next(iter(workers))]['median_seconds']
        for label, measured in row['builds'].items():
            measured['delta_percent'] = (measured['median_seconds'] / baseline - 1) * 100
            print(f"{context * 80} ms {label}: {measured['median_seconds']:.3f}s "
                  f"({measured['delta_percent']:+.1f}%)", flush=True)
        row['same_transcript'] = len({r['text'] for r in row['builds'].values()}) == 1
        report['results'].append(row)
        save()
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
