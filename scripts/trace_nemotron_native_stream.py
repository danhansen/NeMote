#!/usr/bin/env python3
"""Compare native NeMo streaming policies; retain real encoder/decoder traces.

Run in the existing NeMo environment. No export or runtime source changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--audio', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--contexts', type=int, nargs='+', default=[1, 2, 7, 14],
                   choices=[1, 2, 7, 14])
    p.add_argument('--policies', nargs='+', default=[
        'native-unpadded', 'native-padded', 'fixed-padded', 'fixed-truncated'],
        choices=['native-unpadded', 'native-padded', 'fixed-padded',
                 'fixed-truncated', 'native-padded-silence'])
    args = p.parse_args()
    import torch
    from export_nemotron_parakeet_optimized import load_model
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with wave.open(args.audio) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', checkpoint_sha256=sha(args.checkpoint),
                  audio_sha256=sha(args.audio), script_sha256=sha(__file__),
                  torch_version=torch.__version__, results=[])
    model = load_model(args.checkpoint, torch.device('cpu')).eval()
    # Preserve the checkpoint's greedy_batch decoding settings, including max_symbols=10.
    report['model_class'] = type(model).__name__
    report['decoding_class'] = type(model.decoding.decoding).__name__
    original_step = model.encoder.cache_aware_stream_step
    captured = []

    def traced_step(*a, **kw):
        result = original_step(*a, **kw)
        captured.append(dict(encoded=result[0].detach().numpy().copy(),
                             encoded_len=result[1].detach().numpy().copy(),
                             keep_all_outputs=kw['keep_all_outputs'],
                             drop_extra=kw['drop_extra_pre_encoded']))
        return result

    model.encoder.cache_aware_stream_step = traced_step
    with torch.inference_mode():
        for context in args.contexts:
            model.encoder.set_default_att_context_size([70, context - 1])
            for policy in args.policies:
                buffer = CacheAwareStreamingAudioBuffer(model, online_normalization=False,
                                                        pad_and_drop_preencoded=policy != 'native-unpadded')
                synthetic_samples = 0
                if policy == 'native-padded-silence':
                    # Exact old corpus caller policy: pad its final raw packet,
                    # then process three full raw silence packets.
                    packet_samples = context * 1280
                    synthetic_samples = (-len(audio)) % packet_samples + 3 * packet_samples
                buffer.append_audio(np.pad(audio, (0, synthetic_samples)))
                channel, time, cache_len = model.encoder.get_initial_cache_state(batch_size=1)
                fixtures = dict(initial_channel=channel.numpy(), initial_time=time.numpy(),
                                initial_cache_len=cache_len.numpy())
                hypothesis = None
                predictions = None
                traces = []
                captured.clear()
                for step, (features, length) in enumerate(buffer):
                    variable_width = features.shape[-1]
                    if policy.startswith('fixed-'):
                        features = torch.nn.functional.pad(features, (0, 9 + context * 8 - variable_width))
                    keep = buffer.is_buffer_empty() and policy != 'fixed-truncated'
                    drop = model.encoder.streaming_cfg.drop_extra_pre_encoded
                    if policy == 'native-unpadded' and step == 0:
                        drop = 0
                    result = model.conformer_stream_step(
                        processed_signal=features, processed_signal_length=length,
                        cache_last_channel=channel, cache_last_time=time,
                        cache_last_channel_len=cache_len, keep_all_outputs=keep,
                        drop_extra_pre_encoded=drop, previous_hypotheses=hypothesis,
                        previous_pred_out=predictions, return_transcription=True)
                    predictions, texts, channel, time, cache_len, hypothesis = result
                    enc = captured[-1]
                    ids = hypothesis[0].y_sequence.tolist()
                    trace = dict(step=step, input_width=features.shape[-1],
                                 variable_width=variable_width, input_length=int(length[0]),
                                 encoded_width=enc['encoded'].shape[-1],
                                 encoded_length=int(enc['encoded_len'][0]),
                                 keep_all_outputs=keep, drop_extra=drop,
                                 tokens=ids, text=hypothesis[0].text)
                    traces.append(trace)
                    if policy == 'fixed-padded':
                        fixtures[f'features_{step}'] = features.numpy().copy()
                        fixtures[f'length_{step}'] = length.numpy().copy()
                        fixtures[f'encoded_{step}'] = enc['encoded']
                        fixtures[f'encoded_len_{step}'] = enc['encoded_len']
                item = dict(chunk_frames=context, policy=policy, traces=traces,
                            text=hypothesis[0].text if hypothesis else '',
                            synthetic_samples=synthetic_samples,
                            remaining_features=max(0, buffer.buffer.shape[-1] - buffer.buffer_idx))
                report['results'].append(item)
                if policy == 'fixed-padded':
                    np.savez(args.output_dir / f'native-c{context}.npz', **fixtures)
                print(json.dumps({k: v for k, v in item.items() if k != 'traces'}), flush=True)
                (args.output_dir / 'native.json').write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'complete'
    (args.output_dir / 'native.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
