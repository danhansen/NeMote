#!/usr/bin/env python3
"""Replay native frontend chunks through the shipped FP32 ONNX ABI."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(session, encodings, blank):
    state1 = np.zeros((2, 1, 640), np.float32)
    state2 = state1.copy()
    token = blank
    ids, traces = [], []
    for step, (encoded, length) in enumerate(encodings):
        for frame in range(length):
            for symbol in range(10):
                out = dict(zip([v.name for v in session.get_outputs()], session.run(None, {
                    'encoder_outputs': encoded[:, :, frame:frame+1],
                    'targets': np.array([[token]], np.int32),
                    'target_length': np.array([1], np.int32),
                    'input_states_1': state1, 'input_states_2': state2})))
                logits = out['outputs'].reshape(-1)
                best = int(np.argmax(logits))
                traces.append(dict(step=step, frame=frame, symbol=symbol, token=best,
                                   input_token=token, blank_margin=float(logits[best] - logits[blank])))
                if best == blank:
                    break
                ids.append(best)
                token = best
                state1, state2 = out['output_states_1'], out['output_states_2']
    return dict(tokens=ids, decisions=traces)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    args = p.parse_args()
    native = json.loads((args.trace_dir / 'native.json').read_text())
    assert native['status'] == 'complete'
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    decoder = ort.InferenceSession(str(args.model_dir / 'decoder_joint.onnx'), options)
    config = json.loads((args.model_dir / 'config.json').read_text())
    report = dict(status='running', ort_version=ort.__version__,
                  native_trace_sha256=sha(args.trace_dir / 'native.json'),
                  script_sha256=sha(Path(__file__)),
                  model_files={v: sha(args.model_dir / v) for v in
                               ['encoder.onnx', 'decoder_joint.onnx', 'config.json',
                                *config.get('shared_weight_files', [])]}, results=[])
    for context in (1, 2, 7, 14):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        for name, size in dict(batch=1, mel_frames=9+8*context,
                               encoder_frames=context, current_frames=context).items():
            options.add_free_dimension_override_by_name(name, size)
        encoder = ort.InferenceSession(str(args.model_dir / 'encoder.onnx'), options)
        reference = next(r for r in native['results'] if r['chunk_frames'] == context
                         and r['policy'] == 'fixed-padded')
        with np.load(args.trace_dir / f'native-c{context}.npz') as data:
            feeds = dict(cache_last_channel=data['initial_channel'],
                         cache_last_time=data['initial_time'],
                         cache_last_channel_len=data['initial_cache_len'])
            for v in encoder.get_inputs():
                if v.name.startswith(('cache_key_layer_', 'cache_value_layer_')):
                    feeds[v.name] = np.zeros(v.shape, np.float32)
            records, native_enc, ort_enc = [], [], []
            for step, trace in enumerate(reference['traces']):
                feeds['processed_signal'] = data[f'features_{step}']
                feeds['processed_signal_length'] = data[f'length_{step}']
                out = dict(zip([v.name for v in encoder.get_outputs()], encoder.run(None, feeds)))
                expected = data[f'encoded_{step}']
                length = int(out['encoded_len'][0])
                assert length == trace['encoded_length'], (context, step, length, trace)
                assert out['encoded'].shape == expected.shape
                difference = np.abs(out['encoded'] - expected)
                records.append(dict(step=step, input_length=trace['input_length'],
                                    encoded_length=length, shape=list(expected.shape),
                                    max_absolute_error=float(difference.max()),
                                    mean_absolute_error=float(difference.mean())))
                native_enc.append((expected.copy(), length))
                ort_enc.append((out['encoded'], length))
                for name in ('cache_last_channel', 'cache_last_time', 'cache_last_channel_len'):
                    feeds[name] = out[name + '_next']
                for v in encoder.get_inputs():
                    if v.name.startswith(('cache_key_layer_', 'cache_value_layer_')):
                        suffix = v.name.removeprefix('cache_')
                        current = out['projected_current_' + suffix]
                        feeds[v.name] = np.concatenate((feeds[v.name], current), axis=1)[:, -70:, :]
            a = decode(decoder, native_enc, config['blank_id'])
            b = decode(decoder, ort_enc, config['blank_id'])
            native_ids = reference['traces'][-1]['tokens']
            item = dict(chunk_frames=context, encoder=records,
                        native_tokens=native_ids, ort_decoder_native_encoder=a,
                        ort_decoder_ort_encoder=b,
                        decoder_matches_native=a['tokens'] == native_ids,
                        encoder_changes_tokens=a['tokens'] != b['tokens'])
            report['results'].append(item)
            print(json.dumps({k: v for k, v in item.items() if k not in
                              ('encoder', 'ort_decoder_native_encoder', 'ort_decoder_ort_encoder')}), flush=True)
        del encoder
        (args.trace_dir / 'onnx-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'complete'
    (args.trace_dir / 'onnx-comparison.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
