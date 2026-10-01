#!/usr/bin/env python3
"""Compare a JSONL Rust frontend probe to the checkpoint's actual NeMo frontend.

Run in the NeMo export environment. No acoustic model weights are loaded.
Tolerance covers cross-library FP32 FFT/reduction rounding, not frame shifts.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import tarfile
import wave

import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def packets(length, pattern):
    if length < 0 or not pattern or any(size <= 0 for size in pattern):
        raise ValueError('invalid audio length or packet pattern')
    result = []
    consumed = 0
    while consumed < length:
        size = min(pattern[len(result) % len(pattern)], length-consumed)
        result.append(size)
        consumed += size
    return result or [0]


def numerical_match(actual, reference, tolerance):
    delta = np.abs(actual-reference)
    a = np.exp(actual.astype(np.float64))
    b = np.exp(reference.astype(np.float64))
    # Controlled runs with NeMo's exact coefficients still show cross-library
    # FP32 FFT/reduction residuals in spectral nulls. Bound that exception in
    # BOTH energy space and log space; do not loosen ordinary feature tolerance.
    quiet = (b <= 16*2**-24) & (np.abs(a-b) <= 5e-10) & (delta <= .005)
    return bool(np.all((delta <= tolerance) | quiet)), int(np.sum((delta > tolerance) & quiet))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--absolute-tolerance', type=float, default=0.001)
    args = parser.parse_args()
    import torch
    import yaml
    from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
    from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
    torch.set_num_threads(2)
    with tarfile.open(args.checkpoint) as archive:
        member = next(m for m in archive.getmembers() if m.name.removeprefix('./') == 'model_config.yaml')
        checkpoint_config = yaml.safe_load(archive.extractfile(member))
        config = checkpoint_config['preprocessor']
    config.pop('_target_')
    oracle = AudioToMelSpectrogramPreprocessor(**config).eval()
    fe = oracle.featurizer
    contract = dict(sample_rate=fe.sample_rate, window=fe.win_length, hop=fe.hop_length,
                    n_fft=fe.n_fft, features=fe.nfilt, preemphasis=fe.preemph,
                    normalization=fe.normalize, dither=fe.dither, training=fe.training,
                    log_guard_type=fe.log_zero_guard_type, log_guard_value=fe.log_zero_guard_value,
                    magnitude_power=fe.mag_power, exact_pad=fe.exact_pad,
                    window_function=config.get('window','hann'),
                    use_grads=fe.use_grads,log=fe.log,
                    dtype=str(fe.fb.dtype),
                    frame_splicing=fe.frame_splicing, pad_to=fe.pad_to, pad_value=fe.pad_value)
    expected = dict(sample_rate=16000, window=400, hop=160, n_fft=512, features=128,
                    preemphasis=.97, normalization='NA', training=False, log_guard_type='add',
                    log_guard_value=2**-24, magnitude_power=2., exact_pad=False,
                    window_function='hann',use_grads=False,log=True,dtype='torch.float32',
                    frame_splicing=1, pad_to=0, pad_value=0.)
    for key, value in expected.items():
        if contract[key] != value:
            raise ValueError(f'unsupported checkpoint frontend {key}={contract[key]!r}')
    rng = np.random.default_rng(20261001)
    enc_config = checkpoint_config['encoder']
    assert enc_config['subsampling']=='dw_striding' and enc_config['causal_downsampling']
    assert enc_config['subsampling_factor']==8 and enc_config['att_context_style']=='chunked_limited'
    contexts = enc_config['att_context_size']
    if not isinstance(contexts[0],list): contexts = [contexts]
    left_context = contexts[0][0]
    assert {ctx[1]+1 for ctx in contexts if ctx[0]==left_context}=={1,2,7,14}
    # Exercise NeMo's configuration methods without constructing learned layers.
    sampling = ConvSubsampling.__new__(ConvSubsampling)
    torch.nn.Module.__init__(sampling)
    sampling.subsampling_factor = enc_config['subsampling_factor']
    configurations = {}
    for size in (8,16,56,112):
        encoder = ConformerEncoder.__new__(ConformerEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.att_context_size = [left_context,size//8-1]
        encoder.att_context_style = enc_config['att_context_style']
        encoder.subsampling_factor = enc_config['subsampling_factor']
        encoder.pre_encode = sampling
        encoder.layers = torch.nn.ModuleList()
        encoder.setup_streaming_params()
        configurations[size] = encoder.streaming_cfg
    cases = []
    for length in (0,1,159,160,161,199,200,201,255,256,257,399,400,401,
                   1279,1280,1281,2559,2560,2561,8959,8960,8961,17919,17920,17921):
        cases.append((f'noise-{length}', rng.uniform(-.2,.2,length).astype(np.float32)))
    for length in (160,1281,17921):
        cases.append((f'silence-{length}', np.zeros(length, np.float32)))
        cases.append((f'dc-{length}', np.full(length,.25,np.float32)))
    for index in (0,1,159,160,199,200,255,256,399,400,1279,1280,2559):
        impulse = np.zeros(2561,np.float32)
        impulse[index] = 1
        cases.append((f'impulse-{index}', impulse))
    for frequency in (50,1000,7900):
        signal = (.3*np.sin(2*np.pi*frequency*np.arange(32001)/16000)).astype(np.float32)
        cases.append((f'tone-{frequency}',signal))
    for amplitude in (1.,1e-4,1e-8):
        signal = (amplitude*np.sin(2*np.pi*4000*np.arange(3201)/16000)).astype(np.float32)
        cases.append((f'dynamic-range-{amplitude}',signal))
    cases.append(('nyquist',np.where(np.arange(3201)%2,.9,-.9).astype(np.float32)))
    cases.append(('long-noise',rng.uniform(-.1,.1,320001).astype(np.float32)))
    if args.audio:
        with wave.open(str(args.audio)) as wav:
            assert wav.getnchannels()==1 and wav.getsampwidth()==2 and wav.getframerate()==16000
            signal = np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype(np.float32)/32768
        cases.append(('speech',signal))
    report = dict(format=1,status='running',contract=contract,torch_version=torch.__version__,
                  verifier_sha256=digest(__file__),
                  checkpoint_sha256=digest(args.checkpoint),probe_sha256=digest(args.probe),
                  source_sha256=digest(args.source),oracle_source_sha256=digest(inspect.getfile(type(fe))),
                  chunk_oracle_source_sha256=digest(inspect.getfile(CacheAwareStreamingAudioBuffer)),
                  absolute_tolerance=args.absolute_tolerance,records=[])
    report['streaming_contract'] = dict(sampling_frames=sampling.get_sampling_frames(),
        modes={str(size//8):dict(chunk_size=cfg.chunk_size,shift_size=cfg.shift_size,
            pre_encode_cache_size=cfg.pre_encode_cache_size,
            drop_extra_pre_encoded=cfg.drop_extra_pre_encoded,
            cache_drop_size=cfg.cache_drop_size) for size,cfg in configurations.items()})
    report['quiet_rounding_exception'] = dict(max_guarded_energy=16*2**-24,
        max_energy_difference=5e-10,max_log_difference=.005,
        evidence='exact-coefficient controls isolate cross-library FP32 FFT/reduction rounding')
    process = subprocess.Popen([str(args.probe.resolve())],stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        for name, signal in cases:
            # Empty streams have no valid features. NeMo's documented streaming
            # convention supplies one padded input sample with valid length zero.
            tensor = torch.from_numpy(signal if len(signal) else np.zeros(1,np.float32)).unsqueeze(0)
            length = torch.tensor([len(signal)],dtype=torch.long)
            with torch.inference_mode():
                features, valid = oracle(input_signal=tensor.clone(),length=length)
                again, again_valid = oracle(input_signal=tensor.clone(),length=length)
            assert torch.equal(features,again) and torch.equal(valid,again_valid), 'eval dither is not deterministic'
            reference = features[0].T.numpy()
            # Batched storage may extend past the true length, even with junk
            # samples there. NeMo masks raw preemphasis past that length before
            # STFT and masks all invalid feature frames afterwards.
            extra = rng.uniform(-.8,.8,257).astype(np.float32)
            stored = torch.from_numpy(np.concatenate((signal,extra))).unsqueeze(0)
            with torch.inference_mode():
                padded_features,padded_valid = oracle(input_signal=stored,length=length)
            assert torch.equal(valid,padded_valid)
            padded_error = float(np.abs(padded_features[0,:,:len(reference)].T.numpy()-reference).max(initial=0))
            assert padded_error <= args.absolute_tolerance
            assert torch.count_nonzero(padded_features[:,:,int(valid[0]):])==0
            previous = None
            patterns = [[max(1,len(signal))],[1,159,257,13,400], [1280],[2560],[8960],[17920]]
            for pattern, chunk_size in zip(patterns, [8,8,8,16,56,112]):
                sizes = packets(len(signal),pattern)
                process.stdin.write(json.dumps(dict(samples=signal.tolist(),packets=sizes,
                                                     chunk_frames=chunk_size))+'\n')
                process.stdin.flush()
                line = process.stdout.readline()
                if not line: raise RuntimeError(process.stderr.read())
                row = json.loads(line)
                actual = np.asarray(row['frames'],np.float32)
                assert actual.shape==reference.shape, (name,pattern,actual.shape,reference.shape)
                assert row['sample_count']==len(signal)
                assert row['max_buffered_samples'] <= 560
                assert len(actual)==len(signal)//160+1 and int(valid[0])==len(signal)//160
                assert np.array_equal(actual[int(valid[0]):],np.zeros_like(actual[int(valid[0]):]))
                if previous is not None: assert np.array_equal(actual,previous), (name,'packetization mismatch')
                previous = actual
                difference = np.abs(actual-reference)
                error = float(difference.max(initial=0))
                worst = np.unravel_index(difference.argmax(),difference.shape)
                guarded_reference_energy = float(np.exp(np.float64(reference[worst])))
                guarded_actual_energy = float(np.exp(np.float64(actual[worst])))
                # Exercise NeMo's actual iterator without allocating the acoustic
                # model. This checkpoint uses causal dw_striding with sampling
                # frames [1,8], pre-encode cache [0,9], and cache drop zero.
                buffer = CacheAwareStreamingAudioBuffer.__new__(CacheAwareStreamingAudioBuffer)
                buffer.buffer = features
                buffer.buffer_idx = 0
                buffer.step = 0
                buffer.input_features = 128
                buffer.pad_and_drop_preencoded = True
                buffer.online_normalization = False
                buffer.sampling_frames = sampling.get_sampling_frames()
                buffer.streaming_cfg = configurations[chunk_size]
                buffer.streams_length = torch.tensor([reference.shape[0]])
                expected_chunks = list(buffer)
                assert len(row['chunks'])==len(expected_chunks), (name,chunk_size,'chunk count')
                chunk_error = 0.
                for actual_chunk, (expected_chunk, expected_length) in zip(row['chunks'],expected_chunks):
                    assert actual_chunk['input_frames']==int(expected_length[0]), (name,'input length')
                    padded = torch.nn.functional.pad(expected_chunk,(0,chunk_size+9-expected_chunk.shape[-1]))
                    actual_chunk = np.asarray(actual_chunk['features'],np.float32).reshape(1,128,chunk_size+9)
                    chunk_error = max(chunk_error,float(np.abs(actual_chunk-padded.numpy()).max(initial=0)))
                    matched,_ = numerical_match(actual_chunk,padded.numpy(),args.absolute_tolerance)
                    assert matched, (name,'chunk values',chunk_error)
                matched,quiet_cells = numerical_match(actual,reference,args.absolute_tolerance)
                record = dict(case=name,pattern=pattern,frames=len(actual),valid_frames=int(valid[0]),
                              encoder_chunk_frames=chunk_size//8,chunk_count=len(expected_chunks),
                              chunk_max_absolute_error=chunk_error,chunk_lengths_match=True,
                              padded_storage_max_absolute_error=padded_error,
                              quiet_rounding_cells=quiet_cells,
                              max_absolute_error=error,mean_absolute_error=float(difference.mean()),
                              worst_frame=int(worst[0]),worst_mel_band=int(worst[1]),
                              reference_energy_at_worst=guarded_reference_energy-2**-24,
                              energy_error_at_worst=abs(guarded_actual_energy-guarded_reference_energy),
                              rms_error=float(np.sqrt(np.mean(difference**2))),
                              bit_exact_packetization=True,max_buffered_samples=row['max_buffered_samples'])
                report['records'].append(record)
                if not matched:
                    raise AssertionError(f'{name} pattern={pattern} max error {error} exceeds tolerance')
            print(f'{name}: max_error={report["records"][-1]["max_absolute_error"]:.8g}',flush=True)
            if name in ('dynamic-range-1.0','nyquist','speech'):
                request=dict(samples=signal.tolist(),packets=[len(signal)],chunk_frames=8,
                             window=fe.window.tolist(),mel_basis=fe.fb[0].flatten().tolist())
                process.stdin.write(json.dumps(request)+'\n')
                process.stdin.flush()
                control=np.asarray(json.loads(process.stdout.readline())['frames'],np.float32)
                delta=np.abs(control-reference)
                worst=np.unravel_index(delta.argmax(),delta.shape)
                ref_energy=float(np.exp(np.float64(reference[worst])))
                control_energy=float(np.exp(np.float64(control[worst])))
                report.setdefault('exact_coefficient_controls',[]).append(dict(case=name,
                    max_absolute_log_error=float(delta.max()),
                    worst_reference_energy=ref_energy-2**-24,
                    energy_error_at_worst=abs(control_energy-ref_energy)))
        report.update(status='complete',passed=True,
                      max_absolute_error=max(r['max_absolute_error'] for r in report['records']))
    except Exception:
        report.update(status='failed',passed=False)
        raise
    finally:
        process.stdin.close()
        if report['status']=='failed': process.terminate()
        if process.wait(timeout=15)!=0 and report['status']=='complete':
            raise RuntimeError(process.stderr.read())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__': main()
