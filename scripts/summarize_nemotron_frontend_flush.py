#!/usr/bin/env python3
"""Summarize frozen frontend-flush evaluations with paired speaker bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def paired_difference(candidate, baseline, words, draws):
    """Each clip is a distinct speaker; resample paired clips, never words."""
    changes = np.asarray(candidate, dtype=np.int64) - np.asarray(baseline, dtype=np.int64)
    words = np.asarray(words, dtype=np.int64)
    if changes.shape != words.shape or words.sum() <= 0:
        raise ValueError('inconsistent paired word counts')
    samples = changes[draws].sum(axis=1) / words[draws].sum(axis=1)
    return dict(difference_percentage_points=float(changes.sum()/words.sum()*100),
                bootstrap_95_percent_interval_percentage_points=(
                    np.quantile(samples, [0.025, 0.975])*100).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=Path('build/performance-audit'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    paths = {name: args.directory / filename for name, filename in dict(
        candidate='frontend-flush-accuracy.json', speed='frontend-flush-speed.json',
        old='accuracy-ablation.json', gguf='gguf-accuracy.json', manifest='accuracy-manifest.json').items()}
    data = {k: json.loads(v.read_text()) for k,v in paths.items()}
    assert all(data[k]['status'] == 'complete' for k in ('candidate', 'speed', 'old', 'gguf'))
    manifest = data['manifest']
    assert len({r['speaker'] for r in manifest}) == len(manifest)
    ids = [r['utt_id'] for r in manifest]
    rng = np.random.default_rng(20261001)
    draws = rng.integers(len(ids), size=(20000, len(ids)))
    report = dict(status='complete', bootstrap_repetitions=20000, seed=20261001,
                  sources={k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
                  accuracy=[], speed=data['speed']['summary'],
                  limitations=['Paired bootstrap screening on 40 speakers, not a population non-inferiority proof.',
                               'No equivalence margin specified; intervals do not certify equivalence.',
                               'Timing is a one-utterance screen with separate pipeline finalization policies.'])
    for c in (1, 2, 7, 14):
        rows = {
            'old_fp32': [r for r in data['old']['results'] if r['label']=='fp32' and r['chunk_frames']==c],
            'old_compact': [r for r in data['old']['results'] if r['label']=='original-quantized' and r['chunk_frames']==c],
            'new_fp32': [r for r in data['candidate']['results'] if r['label']=='fp32' and r['chunk_frames']==c],
            'new_compact': [r for r in data['candidate']['results'] if r['label']=='compact' and r['chunk_frames']==c],
            'gguf': [r for r in data['gguf']['results'] if r['chunk_frames']==c]}
        errors = {}
        words = None
        for label, items in rows.items():
            by_id = {r['clip']: r for r in items}
            assert set(by_id) == set(ids)
            current_words = [by_id[i]['words'] for i in ids]
            if words is None:
                words = current_words
            else:
                assert words == current_words
            errors[label] = np.array([by_id[i]['edits'] for i in ids])
        values = {label: dict(errors=int(e.sum()), words=sum(words), wer_percent=float(e.sum()/sum(words)*100))
                  for label,e in errors.items()}
        differences = {label: paired_difference(errors[a], errors[b], words, draws)
                       for label,a,b in [('frontend_fp32', 'new_fp32', 'old_fp32'),
                                         ('frontend_compact', 'new_compact', 'old_compact'),
                                         ('new_compact_vs_gguf', 'new_compact', 'gguf')]}
        differences['change_in_quantization_penalty'] = paired_difference(
            errors['new_compact']-errors['new_fp32'],
            errors['old_compact']-errors['old_fp32'], words, draws)
        report['accuracy'].append(dict(chunk_frames=c, configurations=values, paired_differences=differences))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['accuracy'], indent=2))


if __name__ == '__main__':
    main()
