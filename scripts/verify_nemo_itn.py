#!/usr/bin/env python3
"""Verify ITN affects finals, not recognition partials, at every chunk size."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('worker', 'model', 'wav', 'itn-tester', 'grammars', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 10:
        parser.error('repetitions must be in [1,10]')
    def sha(path):
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    report = {'status': 'running', 'threads': 2, 'concurrency': 1,
              'artifacts': {str(path): sha(path) for path in (args.worker, args.model, args.wav, args.itn_tester)},
              'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    save()
    for context in (1, 2, 7, 14):
        builds = {}
        for enabled in (False, True):
            command = [str(args.worker), '--model-dir', str(args.model), '--wav', str(args.wav),
                       '--device', 'cpu', '--num-threads', '2', '--chunk-samples', str(context * 1280),
                       '--wav-repeat', str(args.repetitions)]
            if enabled:
                command.append('--itn')
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=240)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            loaded = next(event['data'] for event in events if event['event'] == 'model_loaded')
            assert loaded['itn'] == enabled
            finals = [event for event in events if event['event'] == 'commit']
            assert len(finals) == args.repetitions and len({event['text'] for event in finals}) == 1
            builds['on' if enabled else 'off'] = {
                'text': finals[0]['text'], 'model_loaded': loaded,
                'partial_texts': [event['text'] for event in events if event['event'] == 'partial'],
                'elapsed_seconds': [event['data']['elapsed_seconds'] for event in finals],
                'decode_seconds': [event['data']['decode_seconds'] for event in finals],
            }
        expected = subprocess.run([str(args.itn_tester), str(args.grammars), builds['off']['text']],
                                  capture_output=True, text=True, check=True, timeout=60).stdout.rstrip('\n')
        assert builds['on']['text'] == expected, (builds['on']['text'], expected)
        assert builds['on']['partial_texts'] == builds['off']['partial_texts'], 'ITN changed raw partials'
        row = {'chunk_ms': context * 80, 'same_raw_partials': True, 'final_matches_normalizer': True, 'builds': builds}
        row['warm_itn_overhead_ms'] = 1000 * (statistics.median(builds['on']['elapsed_seconds'][-1:]) -
                                             statistics.median(builds['off']['elapsed_seconds'][-1:]))
        report['results'].append(row)
        save()
        print(f"{context * 80} ms: raw partials identical; ITN final matches normalizer; "
              f"warm elapsed delta {row['warm_itn_overhead_ms']:+.1f} ms", flush=True)
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
