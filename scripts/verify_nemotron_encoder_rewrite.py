#!/usr/bin/env python3
"""Verify an exact encoder rewrite against its pre-rewrite quantized graph.

This is an equivalence gate, not an acoustic accuracy ablation. Accuracy
ablations should score reference transcripts with FP32 and quantized baselines.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(baseline, candidate, fixtures, *, threads=2):
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    if manifest.get('format') != 1:
        raise ValueError('unsupported fixture format')
    modes = sorted({r['chunk_frames'] for r in manifest['records']})
    hashes = {'baseline_sha256': sha256(baseline), 'candidate_sha256': sha256(candidate)}
    records = []
    for mode in ('dynamic', *modes):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        if mode != 'dynamic':
            for name, size in dict(batch=1,
                                  mel_frames=mode*manifest['subsampling_factor']+manifest['mel_frames_overhead'],
                                  encoder_frames=mode, current_frames=mode).items():
                options.add_free_dimension_override_by_name(name, size)
        a = ort.InferenceSession(str(baseline), options)
        b = ort.InferenceSession(str(candidate), options)
        if [v.name for v in a.get_inputs()] != [v.name for v in b.get_inputs()]:
            raise ValueError('input ABI changed')
        if [v.name for v in a.get_outputs()] != [v.name for v in b.get_outputs()]:
            raise ValueError('output ABI changed')
        for row in manifest['records']:
            if mode != 'dynamic' and row['chunk_frames'] != mode:
                continue
            filename = row['fixture']
            if Path(filename).name != filename:
                raise ValueError('unsafe fixture name')
            with np.load(fixtures / filename, allow_pickle=False) as data:
                rng = np.random.default_rng(42)
                feeds = {}
                for value in a.get_inputs():
                    if value.name in data:
                        feeds[value.name] = data[value.name]
                    elif value.name.startswith(('cache_key_layer_', 'cache_value_layer_')):
                        # Existing projected cache ABI dimensions are static,
                        # even when the current chunk remains dynamic.
                        shape = [1 if dim == 'batch' else dim for dim in value.shape]
                        feeds[value.name] = rng.normal(size=shape).astype(np.float32)
                    else:
                        raise ValueError(f'fixture lacks {value.name}')
                for value, expected, actual in zip(a.get_outputs(), a.run(None, feeds), b.run(None, feeds), strict=True):
                    if np.issubdtype(actual.dtype, np.floating) and not np.isfinite(actual).all():
                        raise ValueError(f'nonfinite output: {value.name}')
                    np.testing.assert_array_equal(actual, expected, err_msg=value.name)
            record = dict(session=mode, chunk_frames=row['chunk_frames'], valid_cache=row['valid_cache'],
                          fixture_sha256=sha256(fixtures / filename), exact=True)
            records.append(record)
            print(json.dumps(record), flush=True)
        del a, b, feeds
        gc.collect()
    if sha256(baseline) != hashes['baseline_sha256'] or sha256(candidate) != hashes['candidate_sha256']:
        raise ValueError('artifact changed during verification')
    return dict(format=1, passed=True, exact=True, ort_version=ort.__version__, threads=threads,
                verifier_sha256=sha256(Path(__file__)), records=records, **hashes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.threads < 1:
        p.error('--threads must be positive')
    report = verify(args.baseline, args.candidate, args.fixtures, threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
