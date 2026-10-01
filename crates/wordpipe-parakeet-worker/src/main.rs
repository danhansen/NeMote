use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use crossbeam_channel::{bounded, select, Receiver, Sender};
use parakeet_rs::{ExecutionConfig, GraphOptimization, Nemotron, NemotronChunkTrace, NemotronMode};
use serde::Deserialize;
use serde_json::{json, Value};

const NEMOTRON_CHUNK_SAMPLES: usize = 8960;

#[derive(Debug, Parser, Clone)]
struct Args {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long)]
    list_input_devices: bool,
    #[arg(long, default_value_t = 2)]
    num_threads: usize,
    #[arg(long, default_value_t = 16000)]
    sample_rate: u32,
    #[arg(long)]
    input_device: Option<String>,
    #[arg(long, default_value = "en-US")]
    language: String,
    #[arg(long, default_value_t = 10.0)]
    queue_seconds: f32,
    #[arg(long, default_value_t = 1.0)]
    stats_interval_seconds: f32,
    #[arg(skip = NEMOTRON_CHUNK_SAMPLES)]
    chunk_samples: usize,
    /// Select a context for dynamic exports; fixed exports must match their context.
    #[arg(long = "chunk-samples")]
    chunk_samples_override: Option<usize>,
    #[arg(long, default_value_t = 3)]
    flush_chunks: usize,
    #[arg(long)]
    wav: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = CliGraphOptimization::All)]
    graph_optimization: CliGraphOptimization,
    #[arg(long, value_enum, default_value_t = CliBoolOverride::Auto)]
    ort_memory_pattern: CliBoolOverride,
    #[arg(long)]
    ort_parallel_execution: bool,
    #[arg(long, value_enum, default_value_t = CliBoolOverride::Auto)]
    ort_cpu_arena: CliBoolOverride,
    #[arg(long)]
    ort_optimized_model_cache_dir: Option<PathBuf>,
    #[arg(long)]
    trace_token_decisions: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliGraphOptimization {
    Disable,
    Level1,
    Level2,
    Level3,
    All,
}

impl From<CliGraphOptimization> for GraphOptimization {
    fn from(value: CliGraphOptimization) -> Self {
        match value {
            CliGraphOptimization::Disable => GraphOptimization::Disable,
            CliGraphOptimization::Level1 => GraphOptimization::Level1,
            CliGraphOptimization::Level2 => GraphOptimization::Level2,
            CliGraphOptimization::Level3 => GraphOptimization::Level3,
            CliGraphOptimization::All => GraphOptimization::All,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliBoolOverride {
    Auto,
    Enable,
    Disable,
}

impl CliBoolOverride {
    fn option(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::Enable => Some(true),
            Self::Disable => Some(false),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Command {
    command: String,
    language: Option<String>,
}

struct JsonEmitter {
    output: Mutex<io::Stdout>,
}

impl JsonEmitter {
    fn new() -> Self {
        Self {
            output: Mutex::new(io::stdout()),
        }
    }

    fn emit(&self, value: Value) {
        if let Ok(mut output) = self.output.lock() {
            let _ = serde_json::to_writer(&mut *output, &value);
            let _ = writeln!(output);
            let _ = output.flush();
        }
    }
}

struct RunningSession {
    stop_tx: Sender<()>,
    join: thread::JoinHandle<Nemotron>,
}

fn main() -> Result<()> {
    let mut args = Arc::new(Args::parse());
    validate_args(&args)?;
    let emitter = Arc::new(JsonEmitter::new());

    if args.list_input_devices {
        return list_input_devices(emitter);
    }

    if args.wav.is_some() {
        return run_wav_file(args, emitter);
    }

    emitter
        .emit(json!({"event": "loading_model", "data": {"model_dir": required_model_dir(&args)?}}));
    let load_started = Instant::now();
    let mut model = Some(load_model(&args)?);
    Arc::make_mut(&mut args).chunk_samples = model.as_ref().unwrap().chunk_samples();
    emitter.emit(json!({
        "event": "model_loaded",
        "data": {"load_seconds": round3(load_started.elapsed().as_secs_f64())}
    }));
    emitter.emit(json!({"event": "ready"}));

    let stdin = io::stdin();
    let mut session: Option<RunningSession> = None;
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let command: Command = match serde_json::from_str(&line) {
            Ok(command) => command,
            Err(err) => {
                emitter.emit(
                    json!({"event": "error", "message": format!("invalid JSON command: {err}")}),
                );
                continue;
            }
        };
        match command.command.as_str() {
            "start" => {
                if session.is_some() {
                    continue;
                }
                let Some(mut session_model) = model.take() else {
                    emitter
                        .emit(json!({"event": "error", "message": "ASR model is not available"}));
                    continue;
                };
                session_model.reset();
                let (stop_tx, stop_rx) = bounded::<()>(1);
                let worker_args = Arc::clone(&args);
                let worker_emitter = Arc::clone(&emitter);
                let join = thread::spawn(move || {
                    if let Err(err) = run_session(
                        worker_args,
                        worker_emitter.clone(),
                        stop_rx,
                        &mut session_model,
                    ) {
                        worker_emitter
                            .emit(json!({"event": "error", "message": format!("{err:#}")}));
                    }
                    worker_emitter.emit(json!({"event": "stopped"}));
                    session_model
                });
                session = Some(RunningSession { stop_tx, join });
            }
            "stop" => {
                if let Some(running) = session.take() {
                    let _ = running.stop_tx.send(());
                    match running.join.join() {
                        Ok(session_model) => model = Some(session_model),
                        Err(_) => return Err(anyhow!("ASR session thread panicked")),
                    }
                } else {
                    emitter.emit(json!({"event": "stopped"}));
                }
            }
            "set_language" => {
                let Some(language) = command.language.as_deref() else {
                    emitter.emit(
                        json!({"event": "error", "message": "set_language requires language"}),
                    );
                    continue;
                };
                let Some(session_model) = model.as_mut() else {
                    emitter.emit(json!({
                        "event": "error",
                        "message": "cannot change language while recognition is active"
                    }));
                    continue;
                };
                match apply_language(session_model, language) {
                    Ok(()) => emitter.emit(json!({
                        "event": "language_changed",
                        "data": {"language": language},
                    })),
                    Err(err) => emitter.emit(json!({
                        "event": "error",
                        "message": format!("{err:#}"),
                    })),
                }
            }
            "shutdown" => {
                if let Some(running) = session.take() {
                    let _ = running.stop_tx.send(());
                    let _ = running.join.join();
                }
                break;
            }
            other => {
                emitter.emit(
                    json!({"event": "error", "message": format!("unknown command: {other}")}),
                );
            }
        }
    }
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if args.model_dir.is_none() && !args.list_input_devices {
        return Err(anyhow!(
            "--model-dir is required unless --list-input-devices is used"
        ));
    }
    if args.num_threads == 0 {
        return Err(anyhow!("--num-threads must be at least 1"));
    }
    if args.sample_rate == 0 {
        return Err(anyhow!("--sample-rate must be at least 1"));
    }
    if args.chunk_samples == 0 || args.chunk_samples_override == Some(0) {
        return Err(anyhow!("--chunk-samples must be at least 1"));
    }
    if !args.queue_seconds.is_finite() || args.queue_seconds <= 0.0 {
        return Err(anyhow!("--queue-seconds must be a positive finite number"));
    }
    if !args.stats_interval_seconds.is_finite() || args.stats_interval_seconds <= 0.0 {
        return Err(anyhow!(
            "--stats-interval-seconds must be a positive finite number"
        ));
    }
    Ok(())
}

fn list_input_devices(emitter: Arc<JsonEmitter>) -> Result<()> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    for (index, device) in host
        .input_devices()
        .context("failed to enumerate input devices")?
        .enumerate()
    {
        let name = device.name().unwrap_or_else(|_| "unknown".to_string());
        emitter.emit(json!({
            "event": "input_device",
            "data": {
                "index": index,
                "name": name,
                "is_default": default_name.as_ref() == Some(&name),
            }
        }));
    }
    Ok(())
}

fn run_session(
    args: Arc<Args>,
    emitter: Arc<JsonEmitter>,
    stop_rx: Receiver<()>,
    model: &mut Nemotron,
) -> Result<()> {
    let host = cpal::default_host();
    let device = select_input_device(&host, args.input_device.as_deref())?;
    let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
    let stream_config = StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(args.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let queue_chunks = ((args.queue_seconds * 100.0).ceil() as usize).max(64);
    let (audio_tx, audio_rx) = bounded::<Vec<f32>>(queue_chunks);
    let (audio_pool_tx, audio_pool_rx) = bounded::<Vec<f32>>(queue_chunks + 4);
    let dropped_chunks = Arc::new(AtomicUsize::new(0));
    let stream = build_input_stream(
        &device,
        &stream_config,
        audio_tx,
        audio_pool_rx,
        Arc::clone(&dropped_chunks),
    )?;
    stream.play().context("failed to start input stream")?;
    let mut stream = Some(stream);

    emitter.emit(json!({
        "event": "listening",
        "data": {"input_device": {"name": device_name, "requested": args.input_device}}
    }));

    let started = Instant::now();
    let mut last_stats = Instant::now();
    let mut pending = Vec::<f32>::with_capacity(args.chunk_samples * 2);
    let mut chunk_buf = vec![0.0f32; args.chunk_samples];
    let silence = vec![0.0f32; args.chunk_samples];
    let mut transcript = String::new();
    let mut accepted_samples = 0usize;
    let mut processed_samples = 0usize;
    let mut decode_seconds = 0.0f64;
    let mut decode_calls = 0usize;
    let mut last_rms = 0.0f32;
    let mut peak_rms = 0.0f32;

    loop {
        let stopping = select! {
            recv(stop_rx) -> _ => {
                drop(stream.take());
                drain_captured_audio(&audio_rx, &mut pending, &audio_pool_tx,
                    &mut accepted_samples, &mut last_rms, &mut peak_rms);
                true
            },
            recv(audio_rx) -> msg => {
                match msg {
                    Ok(samples) => {
                        accepted_samples += samples.len();
                        let (rms, peak) = audio_level(&samples);
                        last_rms = rms;
                        peak_rms = peak_rms.max(peak);
                        pending.extend_from_slice(&samples);
                        recycle_audio_buffer(samples, &audio_pool_tx);
                        false
                    },
                    Err(_) => {
                        drop(stream.take());
                        true
                    }
                }
            },
            default(Duration::from_millis(20)) => false,
        };
        while pending.len() >= args.chunk_samples {
            chunk_buf.copy_from_slice(&pending[..args.chunk_samples]);
            pending.drain(..args.chunk_samples);
            processed_samples += chunk_buf.len();
            let decoded = decode_chunk(
                model,
                &chunk_buf,
                &mut decode_seconds,
                &mut decode_calls,
                args.trace_token_decisions,
            )?;
            emit_token_trace(&emitter, &decoded);
            let decoded = decoded.text;
            if !decoded.is_empty() {
                transcript.push_str(&decoded);
                emitter.emit(json!({
                    "event": "partial",
                    "text": transcript,
                    "data": metrics(started, accepted_samples, processed_samples, args.sample_rate, decode_seconds, decode_calls, dropped_chunks.load(Ordering::Relaxed), last_rms, peak_rms),
                }));
            }
        }
        if stopping {
            break;
        }

        if last_stats.elapsed() >= Duration::from_secs_f32(args.stats_interval_seconds.max(0.1)) {
            last_stats = Instant::now();
            emitter.emit(json!({
                "event": "stats",
                "text": transcript,
                "data": metrics(started, accepted_samples, processed_samples, args.sample_rate, decode_seconds, decode_calls, dropped_chunks.load(Ordering::Relaxed), last_rms, peak_rms),
            }));
        }
    }

    if !pending.is_empty() {
        chunk_buf.fill(0.0);
        chunk_buf[..pending.len()].copy_from_slice(&pending);
        processed_samples += chunk_buf.len();
        let decoded = decode_chunk(
            model,
            &chunk_buf,
            &mut decode_seconds,
            &mut decode_calls,
            args.trace_token_decisions,
        )?;
        emit_token_trace(&emitter, &decoded);
        let decoded = decoded.text;
        if !decoded.is_empty() {
            transcript.push_str(&decoded);
            emitter.emit(json!({
                "event": "partial",
                "text": transcript,
                "data": metrics(started, accepted_samples, processed_samples, args.sample_rate, decode_seconds, decode_calls, dropped_chunks.load(Ordering::Relaxed), last_rms, peak_rms),
            }));
        }
    }

    for _ in 0..args.flush_chunks {
        processed_samples += silence.len();
        let decoded = decode_chunk(
            model,
            &silence,
            &mut decode_seconds,
            &mut decode_calls,
            args.trace_token_decisions,
        )?;
        emit_token_trace(&emitter, &decoded);
        let decoded = decoded.text;
        if !decoded.is_empty() {
            transcript.push_str(&decoded);
            emitter.emit(json!({
                "event": "partial",
                "text": transcript,
                "data": metrics(started, accepted_samples, processed_samples, args.sample_rate, decode_seconds, decode_calls, dropped_chunks.load(Ordering::Relaxed), last_rms, peak_rms),
            }));
        }
    }

    let committed = transcript.trim();
    if !committed.is_empty() {
        emitter.emit(json!({
            "event": "commit",
            "text": committed,
            "data": metrics(started, accepted_samples, processed_samples, args.sample_rate, decode_seconds, decode_calls, dropped_chunks.load(Ordering::Relaxed), last_rms, peak_rms),
        }));
    }
    drop(stream);
    Ok(())
}

fn run_wav_file(mut args: Arc<Args>, emitter: Arc<JsonEmitter>) -> Result<()> {
    let wav_path = args
        .wav
        .as_ref()
        .ok_or_else(|| anyhow!("--wav is required for WAV mode"))?;
    emitter.emit(json!({"event": "loading_wav", "data": {"path": wav_path}}));
    let samples = read_wav_mono_float32(wav_path, args.sample_rate)?;
    emitter
        .emit(json!({"event": "loading_model", "data": {"model_dir": required_model_dir(&args)?}}));
    let load_started = Instant::now();
    let mut model = load_model(&args)?;
    Arc::make_mut(&mut args).chunk_samples = model.chunk_samples();
    emitter.emit(json!({
        "event": "model_loaded",
        "data": {"load_seconds": round3(load_started.elapsed().as_secs_f64())}
    }));
    let started = Instant::now();
    let mut transcript = String::new();
    let mut decode_seconds = 0.0f64;
    let mut decode_calls = 0usize;
    let mut peak_rms = 0.0f32;
    let mut accepted_samples = 0usize;
    let mut processed_samples = 0usize;
    let mut chunk_buf = vec![0.0f32; args.chunk_samples];
    let silence = vec![0.0f32; args.chunk_samples];

    for chunk in samples.chunks(args.chunk_samples) {
        let chunk_index = decode_calls;
        chunk_buf.fill(0.0);
        chunk_buf[..chunk.len()].copy_from_slice(chunk);
        accepted_samples += chunk.len();
        processed_samples += chunk_buf.len();
        let (last_rms, peak) = audio_level(chunk);
        peak_rms = peak_rms.max(peak);
        emitter.emit(json!({
            "event": "decoding_chunk",
                "data": {
                    "chunk_index": chunk_index,
                    "samples": chunk.len(),
                    "processed_samples": chunk_buf.len(),
                    "synthetic_samples": chunk_buf.len().saturating_sub(chunk.len()),
                },
        }));
        let decoded = decode_chunk(
            &mut model,
            &chunk_buf,
            &mut decode_seconds,
            &mut decode_calls,
            args.trace_token_decisions,
        )?;
        emit_token_trace(&emitter, &decoded);
        let decoded = decoded.text;
        if !decoded.is_empty() {
            transcript.push_str(&decoded);
            emitter.emit(json!({
                "event": "partial",
                "text": transcript,
                "data": metrics(started, accepted_samples, processed_samples, args.sample_rate, decode_seconds, decode_calls, 0, last_rms, peak_rms),
            }));
        }
        emitter.emit(json!({
            "event": "stats",
            "text": transcript,
            "data": metrics(started, accepted_samples, processed_samples, args.sample_rate, decode_seconds, decode_calls, 0, last_rms, peak_rms),
        }));
    }

    let last_rms = 0.0f32;
    for _ in 0..args.flush_chunks {
        let chunk_index = decode_calls;
        emitter.emit(json!({
            "event": "decoding_flush_chunk",
            "data": {
                "chunk_index": chunk_index,
                "samples": args.chunk_samples,
                "processed_samples": args.chunk_samples,
                "synthetic_samples": args.chunk_samples,
            },
        }));
        processed_samples += silence.len();
        let decoded = decode_chunk(
            &mut model,
            &silence,
            &mut decode_seconds,
            &mut decode_calls,
            args.trace_token_decisions,
        )?;
        emit_token_trace(&emitter, &decoded);
        let decoded = decoded.text;
        if !decoded.is_empty() {
            transcript.push_str(&decoded);
            emitter.emit(json!({
                "event": "partial",
                "text": transcript,
                "data": metrics(started, accepted_samples, processed_samples, args.sample_rate, decode_seconds, decode_calls, 0, last_rms, peak_rms),
            }));
        }
    }

    let committed = transcript.trim();
    if !committed.is_empty() {
        emitter.emit(json!({
            "event": "commit",
            "text": committed,
            "data": metrics(started, accepted_samples, processed_samples, args.sample_rate, decode_seconds, decode_calls, 0, last_rms, peak_rms),
        }));
    }
    Ok(())
}

fn load_model(args: &Args) -> Result<Nemotron> {
    let model_dir = required_model_dir(args)?;
    let config = ExecutionConfig::new()
        .with_intra_threads(args.num_threads)
        .with_inter_threads(1)
        .with_graph_optimization(args.graph_optimization.into())
        .with_memory_pattern(args.ort_memory_pattern.option())
        .with_parallel_execution(args.ort_parallel_execution)
        .with_cpu_arena(args.ort_cpu_arena.option());
    let config = if let Some(cache_dir) = args.ort_optimized_model_cache_dir.as_ref() {
        config.with_ort_optimized_model_cache_dir(cache_dir)
    } else {
        config
    };
    let config = match args.chunk_samples_override {
        Some(samples) if samples > 0 && samples % (8 * 160) == 0 => {
            config.with_streaming_chunk_frames(samples / (8 * 160))
        }
        _ => config,
    };
    let mut model = Nemotron::from_pretrained(model_dir.to_string_lossy().as_ref(), Some(config))
        .with_context(|| {
        format!("failed to load Nemotron model from {}", model_dir.display())
    })?;
    if args
        .chunk_samples_override
        .is_some_and(|samples| model.chunk_samples() != samples)
    {
        return Err(anyhow!(
            "model requires {} samples per chunk, but worker configured for {}",
            model.chunk_samples(),
            args.chunk_samples_override.unwrap()
        ));
    }
    apply_language(&mut model, &args.language)?;
    Ok(model)
}

fn apply_language(model: &mut Nemotron, language: &str) -> Result<()> {
    let language = language.trim();
    if model.mode() == NemotronMode::Multilingual {
        model
            .set_target_lang(language)
            .with_context(|| format!("failed to set target language to {language:?}"))?;
    } else if !language.is_empty()
        && language != "auto"
        && language != "en"
        && !language.to_ascii_lowercase().starts_with("en-")
    {
        return Err(anyhow!(
            "language {language:?} requires a multilingual Nemotron model"
        ));
    }
    Ok(())
}

fn required_model_dir(args: &Args) -> Result<&PathBuf> {
    args.model_dir
        .as_ref()
        .ok_or_else(|| anyhow!("--model-dir is required"))
}

fn decode_chunk(
    model: &mut Nemotron,
    chunk: &[f32],
    decode_seconds: &mut f64,
    decode_calls: &mut usize,
    trace_tokens: bool,
) -> Result<NemotronChunkTrace> {
    let started = Instant::now();
    let trace = if trace_tokens {
        model
            .transcribe_chunk_with_trace(chunk)
            .context("Nemotron chunk decode failed")?
    } else {
        NemotronChunkTrace {
            text: model
                .transcribe_chunk(chunk)
                .context("Nemotron chunk decode failed")?,
            decisions: Vec::new(),
        }
    };
    *decode_seconds += started.elapsed().as_secs_f64();
    *decode_calls += 1;
    Ok(trace)
}

fn emit_token_trace(emitter: &JsonEmitter, trace: &NemotronChunkTrace) {
    for decision in &trace.decisions {
        emitter.emit(json!({
            "event": "token_decision",
            "data": {
                "chunk_index": decision.chunk_index,
                "frame_index": decision.frame_index,
                "symbol_index": decision.symbol_index,
                "input_token_id": decision.input_token_id,
                "token_id": decision.token_id,
                "piece": decision.piece,
                "logit": decision.logit,
                "blank_logit": decision.blank_logit,
                "margin": decision.margin,
                "top": decision.top.iter().map(|top| {
                    json!({
                        "id": top.id,
                        "piece": top.piece,
                        "logit": top.logit,
                    })
                }).collect::<Vec<_>>(),
            }
        }));
    }
}

fn select_input_device(host: &cpal::Host, requested: Option<&str>) -> Result<cpal::Device> {
    let devices = host
        .input_devices()
        .context("failed to enumerate input devices")?;
    if let Some(requested) = requested {
        if let Ok(index) = requested.parse::<usize>() {
            return devices
                .into_iter()
                .nth(index)
                .ok_or_else(|| anyhow!("input device index not found: {index}"));
        }

        for device in devices {
            let name = device.name().unwrap_or_default();
            if name == requested || name.contains(requested) {
                return Ok(device);
            }
        }
        return Err(anyhow!("input device not found: {requested}"));
    }

    host.default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))
}

fn build_input_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    audio_tx: Sender<Vec<f32>>,
    audio_pool_rx: Receiver<Vec<f32>>,
    dropped_chunks: Arc<AtomicUsize>,
) -> Result<cpal::Stream> {
    let err_fn = |err| eprintln!("audio input stream error: {err}");
    let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
    let sample_format = device
        .default_input_config()
        .with_context(|| {
            format!(
                "failed to read default input config for device '{device_name}' at requested {} Hz",
                config.sample_rate.0
            )
        })?
        .sample_format();

    match sample_format {
        SampleFormat::F32 => device
            .build_input_stream(
                config,
                move |data: &[f32], _| {
                    let mut samples = take_audio_buffer(&audio_pool_rx, data.len());
                    samples.extend_from_slice(data);
                    send_audio(samples, &audio_tx, &dropped_chunks);
                },
                err_fn,
                None,
            )
            .with_context(|| {
                format!(
                    "failed to build f32 input stream for device '{device_name}' at {} Hz",
                    config.sample_rate.0
                )
            }),
        SampleFormat::I16 => device
            .build_input_stream(
                config,
                move |data: &[i16], _| {
                    let mut samples = take_audio_buffer(&audio_pool_rx, data.len());
                    samples.extend(data.iter().map(|sample| *sample as f32 / 32768.0));
                    send_audio(samples, &audio_tx, &dropped_chunks);
                },
                err_fn,
                None,
            )
            .with_context(|| {
                format!(
                    "failed to build i16 input stream for device '{device_name}' at {} Hz",
                    config.sample_rate.0
                )
            }),
        SampleFormat::U16 => device
            .build_input_stream(
                config,
                move |data: &[u16], _| {
                    let mut samples = take_audio_buffer(&audio_pool_rx, data.len());
                    samples.extend(
                        data.iter()
                            .map(|sample| (*sample as f32 - 32768.0) / 32768.0),
                    );
                    send_audio(samples, &audio_tx, &dropped_chunks);
                },
                err_fn,
                None,
            )
            .with_context(|| {
                format!(
                    "failed to build u16 input stream for device '{device_name}' at {} Hz",
                    config.sample_rate.0
                )
            }),
        other => Err(anyhow!("unsupported input sample format: {other:?}")),
    }
}

fn read_wav_mono_float32(path: &PathBuf, expected_sample_rate: u32) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV file {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != expected_sample_rate {
        return Err(anyhow!(
            "expected {} Hz WAV, got {} Hz",
            expected_sample_rate,
            spec.sample_rate
        ));
    }

    let mut samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("failed to read float WAV samples")?,
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|sample| sample.map(|sample| sample as f32 / 32768.0))
                .collect::<Result<Vec<_>, _>>()
                .context("failed to read i16 WAV samples")?,
            other => return Err(anyhow!("unsupported integer WAV bit depth: {other}")),
        },
    };

    if spec.channels > 1 {
        samples = samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }
    Ok(samples)
}

fn send_audio(samples: Vec<f32>, audio_tx: &Sender<Vec<f32>>, dropped_chunks: &AtomicUsize) {
    if audio_tx.try_send(samples).is_err() {
        dropped_chunks.fetch_add(1, Ordering::Relaxed);
    }
}

fn take_audio_buffer(audio_pool_rx: &Receiver<Vec<f32>>, requested_len: usize) -> Vec<f32> {
    let mut samples = audio_pool_rx
        .try_recv()
        .unwrap_or_else(|_| Vec::with_capacity(requested_len));
    samples.clear();
    if samples.capacity() < requested_len {
        samples.reserve(requested_len - samples.capacity());
    }
    samples
}

fn recycle_audio_buffer(mut samples: Vec<f32>, audio_pool_tx: &Sender<Vec<f32>>) {
    samples.clear();
    let _ = audio_pool_tx.try_send(samples);
}

fn drain_captured_audio(
    audio_rx: &Receiver<Vec<f32>>,
    pending: &mut Vec<f32>,
    pool_tx: &Sender<Vec<f32>>,
    accepted_samples: &mut usize,
    last_rms: &mut f32,
    peak_rms: &mut f32,
) {
    while let Ok(samples) = audio_rx.try_recv() {
        *accepted_samples += samples.len();
        let (rms, peak) = audio_level(&samples);
        *last_rms = rms;
        *peak_rms = peak_rms.max(peak);
        pending.extend_from_slice(&samples);
        recycle_audio_buffer(samples, pool_tx);
    }
}

fn audio_level(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum = 0.0f32;
    let mut peak = 0.0f32;
    for sample in samples {
        sum += sample * sample;
        peak = peak.max(sample.abs());
    }
    ((sum / samples.len() as f32).sqrt(), peak)
}

fn metrics(
    started: Instant,
    accepted_samples: usize,
    processed_samples: usize,
    sample_rate: u32,
    decode_seconds: f64,
    decode_calls: usize,
    dropped_audio_chunks: usize,
    last_rms: f32,
    peak_rms: f32,
) -> Value {
    let audio_seconds = accepted_samples as f64 / sample_rate as f64;
    let processed_audio_seconds = processed_samples as f64 / sample_rate as f64;
    json!({
        "audio_seconds": round3(audio_seconds),
        "processed_audio_seconds": round3(processed_audio_seconds),
        "synthetic_audio_seconds": round3((processed_audio_seconds - audio_seconds).max(0.0)),
        "elapsed_seconds": round3(started.elapsed().as_secs_f64()),
        "decode_seconds": round3(decode_seconds),
        "decode_calls": decode_calls,
        "dropped_audio_chunks": dropped_audio_chunks,
        "last_rms": round5(last_rms as f64),
        "peak_rms": round5(peak_rms as f64),
        "real_time_factor": if processed_audio_seconds > 0.0 { round3(decode_seconds / processed_audio_seconds) } else { 0.0 },
        "real_audio_real_time_factor": if audio_seconds > 0.0 { round3(decode_seconds / audio_seconds) } else { 0.0 },
    })
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn round5(value: f64) -> f64 {
    (value * 100000.0).round() / 100000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_args() -> Args {
        Args {
            model_dir: Some(PathBuf::from("/models/nemotron")),
            list_input_devices: false,
            num_threads: 2,
            sample_rate: 16_000,
            input_device: None,
            language: "en-US".to_string(),
            queue_seconds: 10.0,
            stats_interval_seconds: 1.0,
            chunk_samples: NEMOTRON_CHUNK_SAMPLES,
            chunk_samples_override: None,
            flush_chunks: 3,
            wav: None,
            graph_optimization: CliGraphOptimization::All,
            ort_memory_pattern: CliBoolOverride::Auto,
            ort_parallel_execution: false,
            ort_cpu_arena: CliBoolOverride::Auto,
            ort_optimized_model_cache_dir: None,
            trace_token_decisions: false,
        }
    }

    #[test]
    fn validate_args_accepts_defaults() {
        validate_args(&default_args()).unwrap();
    }

    #[test]
    fn stopping_preserves_all_queued_audio_in_capture_order() {
        let (tx, rx) = bounded(4);
        let (pool_tx, pool_rx) = bounded(4);
        tx.send(vec![0.25; 5]).unwrap();
        tx.send(vec![0.5; 7]).unwrap();
        drop(tx);
        let mut pending = vec![0.75; 3];
        let mut accepted = 3;
        let mut rms = 0.75;
        let mut peak = 0.75;
        drain_captured_audio(
            &rx,
            &mut pending,
            &pool_tx,
            &mut accepted,
            &mut rms,
            &mut peak,
        );
        assert_eq!(accepted, 15);
        assert_eq!(
            pending,
            [vec![0.75; 3], vec![0.25; 5], vec![0.5; 7]].concat()
        );
        assert_eq!(rms, 0.5);
        assert_eq!(peak, 0.75);
        assert_eq!(pool_rx.len(), 2);
    }

    #[test]
    fn validate_args_rejects_zero_chunk_samples() {
        let mut args = default_args();
        args.chunk_samples = 0;

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(error.contains("--chunk-samples"));
    }

    #[test]
    fn validate_args_rejects_zero_sample_rate() {
        let mut args = default_args();
        args.sample_rate = 0;

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(error.contains("--sample-rate"));
    }

    #[test]
    fn validate_args_rejects_non_positive_queue_seconds() {
        let mut args = default_args();
        args.queue_seconds = 0.0;

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(error.contains("--queue-seconds"));
    }

    #[test]
    fn validate_args_allows_device_listing_without_model_dir() {
        let mut args = default_args();
        args.model_dir = None;
        args.list_input_devices = true;

        validate_args(&args).unwrap();
    }

    #[test]
    fn validate_args_rejects_missing_model_dir_for_recognition() {
        let mut args = default_args();
        args.model_dir = None;

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(error.contains("--model-dir"));
    }
}
