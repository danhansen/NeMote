use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use wordpipe_protocol::{
    is_backend, is_english_language, is_language, is_model_family, is_model_profile,
    language_available_for_model_family, model_preset, model_preset_id, ModelPresetSpec, BACKENDS,
    BUS_NAME, DEFAULT_BACKEND, DEFAULT_LANGUAGE, DEFAULT_MODEL_FAMILY, DEFAULT_MODEL_PROFILE,
    DEFAULT_NUM_THREADS, DEFAULT_SAMPLE_RATE, DEFAULT_SHORTCUT, MODEL_PRESETS, MODEL_PROFILES,
    OBJECT_PATH,
};
use zbus::object_server::SignalEmitter;
use zbus::zvariant::{OwnedValue, Value};
use zbus::{connection, interface};

type VariantMap = HashMap<String, OwnedValue>;
const PROFILE_COMPLETION_MARKER: &str = ".wordpipe-profile.json";

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    replace: bool,
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ServiceConfig {
    backend: String,
    model_profile: String,
    model_family: String,
    input_device: String,
    language: String,
    shortcut: String,
    model_root: String,
    worker_path: String,
    model_installer_path: String,
    sample_rate: u32,
    num_threads: u32,
    spoken_punctuation: bool,
    insert_partials: bool,
    stream_insert_delay_ms: u32,
    show_overlay: bool,
}

#[derive(Default, Deserialize, Serialize)]
struct PersistedConfig {
    backend: Option<String>,
    model_profile: Option<String>,
    model_family: Option<String>,
    input_device: Option<String>,
    language: Option<String>,
    shortcut: Option<String>,
    model_root: Option<String>,
    worker_path: Option<String>,
    model_installer_path: Option<String>,
    sample_rate: Option<u32>,
    num_threads: Option<u32>,
    spoken_punctuation: Option<bool>,
    insert_partials: Option<bool>,
    stream_insert_delay_ms: Option<u32>,
    show_overlay: Option<bool>,
}

impl From<&ServiceConfig> for PersistedConfig {
    fn from(config: &ServiceConfig) -> Self {
        Self {
            backend: Some(config.backend.clone()),
            model_profile: Some(config.model_profile.clone()),
            model_family: Some(config.model_family.clone()),
            input_device: Some(config.input_device.clone()),
            language: Some(config.language.clone()),
            shortcut: Some(config.shortcut.clone()),
            model_root: Some(config.model_root.clone()),
            worker_path: Some(config.worker_path.clone()),
            model_installer_path: Some(config.model_installer_path.clone()),
            sample_rate: Some(config.sample_rate),
            num_threads: Some(config.num_threads),
            spoken_punctuation: Some(config.spoken_punctuation),
            insert_partials: Some(config.insert_partials),
            stream_insert_delay_ms: Some(config.stream_insert_delay_ms),
            show_overlay: Some(config.show_overlay),
        }
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            backend: DEFAULT_BACKEND.to_string(),
            model_profile: DEFAULT_MODEL_PROFILE.to_string(),
            model_family: DEFAULT_MODEL_FAMILY.to_string(),
            input_device: String::new(),
            language: DEFAULT_LANGUAGE.to_string(),
            shortcut: DEFAULT_SHORTCUT.to_string(),
            model_root: default_model_root(),
            worker_path: default_worker_path(),
            model_installer_path: default_model_installer_path(),
            sample_rate: DEFAULT_SAMPLE_RATE,
            num_threads: DEFAULT_NUM_THREADS,
            spoken_punctuation: true,
            insert_partials: true,
            stream_insert_delay_ms: 0,
            show_overlay: true,
        }
    }
}

struct ServiceData {
    config: ServiceConfig,
    listening: bool,
    stopping: bool,
    installing: bool,
    installing_profile: String,
    loading_model: bool,
    model_loaded: bool,
    session_id: u64,
    seq: u64,
    partial_text: String,
    last_commit_text: String,
    last_error: String,
    last_metrics: VariantMap,
    last_install_progress: VariantMap,
    worker: Option<WorkerProcess>,
}

impl Default for ServiceData {
    fn default() -> Self {
        Self {
            config: ServiceConfig::default(),
            listening: false,
            stopping: false,
            installing: false,
            installing_profile: String::new(),
            loading_model: false,
            model_loaded: false,
            session_id: 0,
            seq: 0,
            partial_text: String::new(),
            last_commit_text: String::new(),
            last_error: String::new(),
            last_metrics: VariantMap::new(),
            last_install_progress: VariantMap::new(),
            worker: None,
        }
    }
}

struct WorkerProcess {
    stdin: Arc<Mutex<ChildStdin>>,
    child: Child,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToggleAction {
    Start,
    Stop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WorkerExit {
    session_id: u64,
    was_active: bool,
    was_expected_shutdown: bool,
}

#[derive(Clone, Default)]
struct WordpipeService {
    data: Arc<Mutex<ServiceData>>,
    emitter: Arc<Mutex<Option<SignalEmitter<'static>>>>,
    config_path: PathBuf,
}

impl WordpipeService {
    fn new(config_path: PathBuf, config: ServiceConfig) -> Self {
        Self {
            data: Arc::new(Mutex::new(ServiceData {
                config,
                ..ServiceData::default()
            })),
            emitter: Arc::default(),
            config_path,
        }
    }
}

#[interface(interface = "dev.wordpipe.Service1")]
impl WordpipeService {
    async fn start(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let is_stopping = { self.lock_data()?.stopping };
        if is_stopping {
            let message = "dictation is stopping".to_string();
            self.record_error(&emitter, &message).await?;
            return Err(zbus::fdo::Error::Failed(message));
        }
        if let Err(err) = self.ensure_worker(emitter.to_owned()).await {
            let message = fdo_error_message(&err);
            self.record_error(&emitter, &message).await?;
            return Err(err);
        }
        let is_stopping = { self.lock_data()?.stopping };
        if is_stopping {
            let message = "dictation is stopping".to_string();
            self.record_error(&emitter, &message).await?;
            return Err(zbus::fdo::Error::Failed(message));
        }
        let (state, session_id, stdin, language) = {
            let mut data = self.lock_data()?;
            if !data.listening {
                data.listening = true;
                data.stopping = false;
                data.session_id = next_session_id(data.session_id);
                data.seq = 0;
                data.partial_text.clear();
                data.last_commit_text.clear();
            }
            let stdin = data
                .worker
                .as_ref()
                .map(|worker| Arc::clone(&worker.stdin))
                .ok_or_else(|| zbus::fdo::Error::Failed("ASR worker is not running".to_string()))?;
            (
                state_map(&data),
                data.session_id,
                stdin,
                data.config.language.clone(),
            )
        };
        send_worker_language_command(&stdin, &language).map_err(fdo_failed)?;
        send_worker_command(&stdin, "start").map_err(fdo_failed)?;
        Self::session_started(&emitter, session_id).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn stop(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (state, stopped_session, stdin, emit_stopped) = {
            let mut data = self.lock_data()?;
            let was_stopping = data.stopping;
            let stopped_session = data.session_id;
            let stdin = if was_stopping {
                None
            } else {
                data.listening = false;
                data.stopping = data.worker.is_some();
                data.seq = data.seq.saturating_add(1);
                data.worker.as_ref().map(|worker| Arc::clone(&worker.stdin))
            };
            let emit_stopped = !was_stopping && stdin.is_none();
            (state_map(&data), stopped_session, stdin, emit_stopped)
        };
        if let Some(stdin) = stdin {
            send_worker_command(&stdin, "stop").map_err(fdo_failed)?;
        }
        if emit_stopped {
            Self::session_stopped(&emitter, stopped_session).await?;
        }
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn toggle(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let action = {
            let data = self.lock_data()?;
            toggle_action(&data)
        };
        match action {
            ToggleAction::Stop => self.stop(emitter).await,
            ToggleAction::Start => self.start(emitter).await,
        }
    }

    fn get_state(&self) -> zbus::fdo::Result<VariantMap> {
        let data = self.lock_data()?;
        Ok(state_map(&data))
    }

    fn get_config(&self) -> zbus::fdo::Result<VariantMap> {
        let data = self.lock_data()?;
        Ok(config_map(&data.config))
    }

    fn list_backends(&self) -> Vec<VariantMap> {
        BACKENDS
            .iter()
            .map(|backend| {
                let mut item = VariantMap::new();
                insert_str(&mut item, "id", backend.id);
                insert_str(&mut item, "title", backend.title);
                insert_str(&mut item, "description", backend.description);
                item
            })
            .collect()
    }

    fn list_model_profiles(&self) -> Vec<VariantMap> {
        let model_root = self
            .data
            .lock()
            .map(|data| data.config.model_root.clone())
            .unwrap_or_default();
        MODEL_PRESETS
            .iter()
            .filter_map(|preset| {
                let profile = MODEL_PROFILES
                    .iter()
                    .find(|profile| profile.id == preset.model_profile)?;
                let english = preset.model_family == "english";
                let output_name = if english {
                    profile.english_output_name
                } else {
                    profile.output_name
                };
                let prebuilt_repo = if english {
                    profile.english_prebuilt_repo
                } else {
                    profile.prebuilt_repo
                };
                let runtime_dir = profile_runtime_dir(&model_root, output_name, profile.ort_format);
                let mut item = VariantMap::new();
                insert_str(&mut item, "id", preset.id);
                insert_str(&mut item, "title", preset.title);
                insert_str(&mut item, "description", profile.description);
                insert_str(&mut item, "model_profile", preset.model_profile);
                insert_str(&mut item, "model_family", preset.model_family);
                insert_str(&mut item, "build_profile", profile.build_profile);
                insert_str(&mut item, "output_name", output_name);
                insert_str(&mut item, "prebuilt_repo", prebuilt_repo);
                insert_bool(&mut item, "ort_format", profile.ort_format);
                insert_str(&mut item, "runtime_dir", &runtime_dir);
                insert_bool(
                    &mut item,
                    "installed",
                    profile_installed(&runtime_dir, profile.id),
                );
                Some(item)
            })
            .collect()
    }

    fn list_input_devices(&self) -> zbus::fdo::Result<Vec<VariantMap>> {
        enumerate_input_devices().map_err(fdo_failed)
    }

    async fn set_backend(
        &self,
        backend: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        if !is_backend(backend) {
            return Err(zbus::fdo::Error::InvalidArgs(format!(
                "unknown backend: {backend}"
            )));
        }
        let (config_data, config, state) = {
            let mut data = self.lock_data()?;
            data.config.backend = backend.to_string();
            shutdown_worker(&mut data);
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            let state = state_map(&data);
            (config_data, config, state)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn set_model_profile(
        &self,
        preset_id: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let Some(preset) = model_preset(preset_id) else {
            return Err(zbus::fdo::Error::InvalidArgs(format!(
                "unknown model preset: {preset_id}"
            )));
        };
        let (config_data, config, state) = {
            let mut data = self.lock_data()?;
            if !model_preset_installed(&data.config, preset) {
                return Err(zbus::fdo::Error::Failed(format!(
                    "model preset '{preset_id}' is not installed; install it before selecting it"
                )));
            }
            if apply_model_preset(&mut data.config, preset) {
                shutdown_worker(&mut data);
            }
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            let state = state_map(&data);
            (config_data, config, state)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn set_input_device(
        &self,
        selector: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (config_data, config, state) = {
            let mut data = self.lock_data()?;
            if data.config.input_device != selector {
                data.config.input_device = selector.to_string();
                shutdown_worker(&mut data);
            }
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            let state = state_map(&data);
            (config_data, config, state)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn set_shortcut(
        &self,
        accelerator: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (config_data, config, state) = {
            let mut data = self.lock_data()?;
            data.config.shortcut = accelerator.to_string();
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            let state = state_map(&data);
            (config_data, config, state)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn set_insertion_options(
        &self,
        options: VariantMap,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (config_data, config, state) = {
            let mut data = self.lock_data()?;
            if let Some(value) = get_bool(&options, "spoken_punctuation") {
                data.config.spoken_punctuation = value;
            }
            if let Some(value) = get_bool(&options, "insert_partials") {
                data.config.insert_partials = value;
            }
            if let Some(value) = get_u32(&options, "stream_insert_delay_ms") {
                data.config.stream_insert_delay_ms = value;
            }
            if let Some(value) = get_bool(&options, "show_overlay") {
                data.config.show_overlay = value;
            }
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            let state = state_map(&data);
            (config_data, config, state)
        };
        self.persist_config(&config_data)?;
        Self::config_changed(&emitter, config).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn set_runtime_options(
        &self,
        options: VariantMap,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let (config_data, config, state, language_update) = {
            let mut data = self.lock_data()?;
            let mut restart_worker = false;
            let mut language_update = None;
            let mut model_root_changed = false;
            if let Some(value) = get_string(&options, "model_root") {
                let value = normalize_model_root(value);
                model_root_changed = data.config.model_root != value;
                restart_worker |= model_root_changed;
                data.config.model_root = value;
            }
            if let Some(value) = get_string(&options, "language") {
                let value = normalize_language(value);
                if !is_language(&value) {
                    return Err(zbus::fdo::Error::InvalidArgs(format!(
                        "unknown language: {value}"
                    )));
                }
                if !language_available_for_model_family(&data.config.model_family, &value) {
                    return Err(zbus::fdo::Error::InvalidArgs(format!(
                        "language {value} is not available for the {} model family",
                        data.config.model_family
                    )));
                }
                if data.config.language != value {
                    if !data.listening && !data.stopping {
                        language_update = data
                            .worker
                            .as_ref()
                            .map(|worker| (Arc::clone(&worker.stdin), value.clone()));
                    }
                }
                data.config.language = value;
            }
            if let Some(value) = get_string(&options, "worker_path") {
                let value = normalize_worker_path(value);
                restart_worker |= data.config.worker_path != value;
                data.config.worker_path = value;
            }
            if let Some(value) = get_string(&options, "model_installer_path") {
                let value = normalize_model_installer_path(value);
                data.config.model_installer_path = value;
            }
            if let Some(value) = get_u32(&options, "sample_rate") {
                if value == 0 {
                    return Err(zbus::fdo::Error::InvalidArgs(
                        "sample_rate must be positive".to_string(),
                    ));
                }
                restart_worker |= data.config.sample_rate != value;
                data.config.sample_rate = value;
            }
            if let Some(value) = get_u32(&options, "num_threads") {
                if value == 0 {
                    return Err(zbus::fdo::Error::InvalidArgs(
                        "num_threads must be positive".to_string(),
                    ));
                }
                restart_worker |= data.config.num_threads != value;
                data.config.num_threads = value;
            }
            if model_root_changed {
                let previous_profile = data.config.model_profile.clone();
                select_installed_model_profile(&mut data.config);
                restart_worker |= data.config.model_profile != previous_profile;
            }
            if restart_worker {
                language_update = None;
                shutdown_worker(&mut data);
            }
            let config_data = data.config.clone();
            let config = config_map(&data.config);
            let state = state_map(&data);
            (config_data, config, state, language_update)
        };
        self.persist_config(&config_data)?;
        if let Some((stdin, language)) = language_update {
            send_worker_language_command(&stdin, &language).map_err(fdo_failed)?;
        }
        Self::config_changed(&emitter, config).await?;
        Self::state_changed(&emitter, state).await?;
        Ok(())
    }

    async fn install_model(
        &self,
        preset_id: &str,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let Some(preset) = model_preset(preset_id) else {
            return Err(zbus::fdo::Error::InvalidArgs(format!(
                "unknown model preset: {preset_id}"
            )));
        };
        let mut progress = VariantMap::new();
        insert_str(&mut progress, "profile", preset.id);
        insert_str(&mut progress, "phase", "starting");
        insert_str(&mut progress, "message", "starting model installer");
        insert_f64(&mut progress, "fraction", 0.0);
        let (installer_path, model_root, state) = {
            let mut data = self.lock_data()?;
            if data.installing {
                return Err(zbus::fdo::Error::Failed(
                    "model installation is already running".to_string(),
                ));
            }
            data.installing = true;
            data.installing_profile = preset.id.to_string();
            data.last_install_progress = progress.clone();
            (
                data.config.model_installer_path.clone(),
                data.config.model_root.clone(),
                state_map(&data),
            )
        };
        Self::install_progress(&emitter, preset.id, progress).await?;
        Self::state_changed(&emitter, state).await?;

        let service = self.clone();
        let profile = preset.model_profile.to_string();
        let family = preset.model_family.to_string();
        let preset_id = preset.id.to_string();
        let emitter = emitter.to_owned();
        std::thread::Builder::new()
            .name("wordpipe-model-install".to_string())
            .spawn(move || {
                service.run_model_installer(
                    installer_path,
                    model_root,
                    profile,
                    family,
                    preset_id,
                    emitter,
                );
            })
            .map_err(|err| zbus::fdo::Error::Failed(err.to_string()))?;
        Ok(())
    }

    async fn shutdown(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        let state = {
            let mut data = self.lock_data()?;
            data.listening = false;
            data.stopping = false;
            shutdown_worker(&mut data);
            state_map(&data)
        };
        Self::state_changed(&emitter, state).await?;
        std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            std::process::exit(0);
        });
        Ok(())
    }

    #[zbus(signal)]
    async fn state_changed(emitter: &SignalEmitter<'_>, state: VariantMap) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn config_changed(emitter: &SignalEmitter<'_>, config: VariantMap) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn session_started(emitter: &SignalEmitter<'_>, session_id: u64) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn text_delta(
        emitter: &SignalEmitter<'_>,
        session_id: u64,
        seq: u64,
        text: &str,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn partial(
        emitter: &SignalEmitter<'_>,
        session_id: u64,
        seq: u64,
        full_text: &str,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn commit(
        emitter: &SignalEmitter<'_>,
        session_id: u64,
        seq: u64,
        text: &str,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn session_stopped(emitter: &SignalEmitter<'_>, session_id: u64) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn install_progress(
        emitter: &SignalEmitter<'_>,
        profile: &str,
        progress: VariantMap,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn metrics(emitter: &SignalEmitter<'_>, metrics: VariantMap) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn error(emitter: &SignalEmitter<'_>, message: &str) -> zbus::Result<()>;
}

impl WordpipeService {
    fn lock_data(&self) -> zbus::fdo::Result<std::sync::MutexGuard<'_, ServiceData>> {
        self.data
            .lock()
            .map_err(|_| zbus::fdo::Error::Failed("service state lock poisoned".to_string()))
    }

    fn persist_config(&self, config: &ServiceConfig) -> zbus::fdo::Result<()> {
        save_service_config(&self.config_path, config).map_err(fdo_failed)
    }

    async fn record_error(
        &self,
        emitter: &SignalEmitter<'_>,
        message: &str,
    ) -> zbus::fdo::Result<()> {
        let state = {
            let mut data = self.lock_data()?;
            data.last_error = message.to_string();
            data.loading_model = false;
            data.model_loaded = false;
            state_map(&data)
        };
        Self::error(emitter, message).await?;
        Self::state_changed(emitter, state).await?;
        Ok(())
    }

    async fn ensure_worker(&self, emitter: SignalEmitter<'static>) -> zbus::fdo::Result<()> {
        let spawn = {
            let data = self.lock_data()?;
            data.worker.is_none()
        };
        if !spawn {
            return Ok(());
        }

        let (worker, stdout) = {
            let mut data = self.lock_data()?;
            let config = data.config.clone();
            let runtime_dir = selected_runtime_dir(&config);
            if !profile_installed(&runtime_dir, &config.model_profile) {
                return Err(zbus::fdo::Error::Failed(format!(
                    "model profile '{}' is not installed at {runtime_dir}",
                    config.model_profile
                )));
            }
            data.loading_model = true;
            data.model_loaded = false;
            data.last_error.clear();
            match spawn_worker(&config, &runtime_dir) {
                Ok(worker) => worker,
                Err(err) => {
                    data.loading_model = false;
                    data.model_loaded = false;
                    data.last_error = err.to_string();
                    return Err(fdo_failed(err));
                }
            }
        };

        {
            let mut data = self.lock_data()?;
            data.worker = Some(worker);
        }

        let service = self.clone();
        std::thread::Builder::new()
            .name("wordpipe-asr-events".to_string())
            .spawn(move || service.read_worker_events(stdout, emitter))
            .map_err(|err| zbus::fdo::Error::Failed(err.to_string()))?;
        Ok(())
    }

    fn read_worker_events(
        &self,
        stdout: std::process::ChildStdout,
        emitter: SignalEmitter<'static>,
    ) {
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<JsonValue>(&line) {
                Ok(value) => self.handle_worker_event(value, &emitter),
                Err(err) => {
                    let _ = zbus::block_on(Self::error(
                        &emitter,
                        &format!("invalid ASR worker event: {err}"),
                    ));
                }
            }
        }
        let (state, exit) = {
            let mut data = match self.data.lock() {
                Ok(data) => data,
                Err(_) => return,
            };
            let exit = apply_worker_exit(&mut data);
            (state_map(&data), exit)
        };
        if exit.was_active {
            let _ = zbus::block_on(Self::session_stopped(&emitter, exit.session_id));
        }
        if exit.was_active && !exit.was_expected_shutdown {
            let _ = zbus::block_on(Self::error(&emitter, "ASR worker exited unexpectedly"));
        }
        let _ = zbus::block_on(Self::state_changed(&emitter, state));
    }

    fn handle_worker_event(&self, value: JsonValue, emitter: &SignalEmitter<'static>) {
        let event = value
            .get("event")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        match event {
            "loading_model" => {
                let state = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.loading_model = true;
                    data.model_loaded = false;
                    state_map(&data)
                };
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            "model_loaded" => {
                let (state, metrics) = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.loading_model = false;
                    data.model_loaded = true;
                    let metrics = value
                        .get("data")
                        .map(json_to_variant_map)
                        .unwrap_or_default();
                    data.last_metrics = metrics.clone();
                    (state_map(&data), metrics)
                };
                let _ = zbus::block_on(Self::state_changed(emitter, state));
                let _ = zbus::block_on(Self::metrics(emitter, metrics));
            }
            "ready" => {
                let state = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.loading_model = false;
                    data.model_loaded = true;
                    state_map(&data)
                };
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            "listening" => {
                let state = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.listening = true;
                    data.stopping = false;
                    state_map(&data)
                };
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            "partial" => self.forward_partial(value, emitter),
            "commit" => self.forward_commit(value, emitter),
            "stats" => {
                let metrics = value
                    .get("data")
                    .map(json_to_variant_map)
                    .unwrap_or_default();
                if let Ok(mut data) = self.data.lock() {
                    data.last_metrics = metrics.clone();
                }
                let _ = zbus::block_on(Self::metrics(emitter, metrics));
            }
            "stopped" => {
                let (state, session_id) = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.listening = false;
                    data.stopping = false;
                    (state_map(&data), data.session_id)
                };
                let _ = zbus::block_on(Self::session_stopped(emitter, session_id));
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            "error" => {
                let message = value
                    .get("message")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("ASR worker error")
                    .to_string();
                let state = {
                    let mut data = match self.data.lock() {
                        Ok(data) => data,
                        Err(_) => return,
                    };
                    data.last_error = message.clone();
                    state_map(&data)
                };
                let _ = zbus::block_on(Self::error(emitter, &message));
                let _ = zbus::block_on(Self::state_changed(emitter, state));
            }
            _ => {}
        }
    }

    fn forward_partial(&self, value: JsonValue, emitter: &SignalEmitter<'static>) {
        let text = value
            .get("text")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string();
        let metrics = value
            .get("data")
            .map(json_to_variant_map)
            .unwrap_or_default();
        let (session_id, seq, delta, text) = {
            let mut data = match self.data.lock() {
                Ok(data) => data,
                Err(_) => return,
            };
            let text = if data.config.spoken_punctuation {
                normalize_spoken_punctuation_partial(&text)
            } else {
                text
            };
            data.seq = data.seq.saturating_add(1);
            let delta = text
                .strip_prefix(&data.partial_text)
                .unwrap_or(&text)
                .to_string();
            data.partial_text = text.clone();
            data.last_metrics = metrics.clone();
            (data.session_id, data.seq, delta, text)
        };
        let _ = zbus::block_on(Self::partial(emitter, session_id, seq, &text));
        if !delta.is_empty() {
            let _ = zbus::block_on(Self::text_delta(emitter, session_id, seq, &delta));
        }
        let _ = zbus::block_on(Self::metrics(emitter, metrics));
    }

    fn forward_commit(&self, value: JsonValue, emitter: &SignalEmitter<'static>) {
        let text = value
            .get("text")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string();
        let metrics = value
            .get("data")
            .map(json_to_variant_map)
            .unwrap_or_default();
        let (session_id, seq, text) = {
            let mut data = match self.data.lock() {
                Ok(data) => data,
                Err(_) => return,
            };
            let text = if data.config.spoken_punctuation {
                normalize_spoken_punctuation(&text)
            } else {
                text
            };
            data.seq = data.seq.saturating_add(1);
            data.last_metrics = metrics.clone();
            data.last_commit_text = text.clone();
            (data.session_id, data.seq, text)
        };
        if !text.is_empty() {
            let _ = zbus::block_on(Self::commit(emitter, session_id, seq, &text));
        }
        let _ = zbus::block_on(Self::metrics(emitter, metrics));
    }

    fn run_model_installer(
        &self,
        installer_path: String,
        model_root: String,
        profile: String,
        family: String,
        preset_id: String,
        emitter: SignalEmitter<'static>,
    ) {
        let command = model_installer_command(&installer_path, &model_root, &profile, &family);

        let result = run_progress_command(command, &preset_id, &emitter, Arc::clone(&self.data));
        let state = {
            let mut data = match self.data.lock() {
                Ok(data) => data,
                Err(_) => return,
            };
            data.installing = false;
            data.installing_profile.clear();
            match result {
                Ok(()) => {
                    data.last_error.clear();
                    let mut progress = VariantMap::new();
                    insert_str(&mut progress, "profile", &preset_id);
                    insert_str(&mut progress, "phase", "complete");
                    insert_str(&mut progress, "message", "model profile installed");
                    insert_f64(&mut progress, "fraction", 1.0);
                    data.last_install_progress = progress.clone();
                    let _ = zbus::block_on(Self::install_progress(&emitter, &preset_id, progress));
                }
                Err(err) => {
                    data.last_error = err.to_string();
                    let mut progress = VariantMap::new();
                    insert_str(&mut progress, "profile", &preset_id);
                    insert_str(&mut progress, "phase", "error");
                    insert_str(&mut progress, "message", &err.to_string());
                    insert_f64(&mut progress, "fraction", 0.0);
                    data.last_install_progress = progress.clone();
                    let _ = zbus::block_on(Self::install_progress(&emitter, &preset_id, progress));
                    let _ = zbus::block_on(Self::error(&emitter, &err.to_string()));
                }
            }
            state_map(&data)
        };
        let _ = zbus::block_on(Self::state_changed(&emitter, state));
    }
}

fn model_installer_command(
    installer_path: &str,
    model_root: &str,
    profile: &str,
    family: &str,
) -> Command {
    let mut command = Command::new(installer_path);
    command
        .arg("--profile")
        .arg(profile)
        .arg("--model-family")
        .arg(family)
        .arg("--model-root")
        .arg(model_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn apply_model_preset(config: &mut ServiceConfig, preset: &ModelPresetSpec) -> bool {
    let changed =
        config.model_profile != preset.model_profile || config.model_family != preset.model_family;
    if !changed {
        return false;
    }
    config.model_profile = preset.model_profile.to_string();
    config.model_family = preset.model_family.to_string();
    if !language_available_for_model_family(&config.model_family, &config.language) {
        config.language = DEFAULT_LANGUAGE.to_string();
    }
    true
}

fn main() -> Result<()> {
    let args = Args::parse();
    zbus::block_on(run(args))
}

async fn run(args: Args) -> Result<()> {
    if args.replace {
        eprintln!("wordpipe-service: --replace requested; D-Bus will replace an existing name if the bus allows it");
    }
    let config_path = args.config.unwrap_or_else(default_config_path);
    let config = load_service_config(&config_path)
        .with_context(|| format!("failed to load service config {}", config_path.display()))?;
    let service = WordpipeService::new(config_path, config);
    let service_handle = service.clone();
    let _connection = connection::Builder::session()?
        .serve_at(OBJECT_PATH, service)?
        .name(BUS_NAME)?
        .build()
        .await
        .with_context(|| format!("failed to own D-Bus name {BUS_NAME}"))?;
    let emitter = SignalEmitter::new(&_connection, OBJECT_PATH)?.to_owned();
    *service_handle
        .emitter
        .lock()
        .expect("service emitter lock should not be poisoned") = Some(emitter);
    eprintln!("wordpipe-service: listening on {BUS_NAME} {OBJECT_PATH}");
    std::future::pending::<()>().await;
    Ok(())
}

fn state_map(data: &ServiceData) -> VariantMap {
    let mut map = VariantMap::new();
    let runtime_dir = selected_runtime_dir(&data.config);
    insert_bool(&mut map, "listening", data.listening);
    insert_bool(&mut map, "stopping", data.stopping);
    insert_bool(&mut map, "installing", data.installing);
    insert_str(&mut map, "installing_profile", &data.installing_profile);
    insert_bool(&mut map, "loading_model", data.loading_model);
    insert_bool(&mut map, "model_loaded", data.model_loaded);
    insert_u64(&mut map, "session_id", data.session_id);
    insert_u64(&mut map, "seq", data.seq);
    insert_str(&mut map, "backend", &data.config.backend);
    insert_str(&mut map, "model_profile", &data.config.model_profile);
    insert_str(&mut map, "model_family", &data.config.model_family);
    if let Some(preset) = model_preset_id(&data.config.model_profile, &data.config.model_family) {
        insert_str(&mut map, "model_preset", preset);
    }
    insert_str(&mut map, "input_device", &data.config.input_device);
    insert_str(&mut map, "partial_text", &data.partial_text);
    insert_str(&mut map, "last_commit_text", &data.last_commit_text);
    insert_str(&mut map, "selected_runtime_dir", &runtime_dir);
    insert_bool(
        &mut map,
        "selected_model_installed",
        profile_installed(&runtime_dir, &data.config.model_profile),
    );
    insert_str(&mut map, "last_error", &data.last_error);
    insert_map(&mut map, "last_metrics", &data.last_metrics);
    insert_map(
        &mut map,
        "last_install_progress",
        &data.last_install_progress,
    );
    map
}

fn config_map(config: &ServiceConfig) -> VariantMap {
    let mut map = VariantMap::new();
    insert_str(&mut map, "backend", &config.backend);
    insert_str(&mut map, "model_profile", &config.model_profile);
    insert_str(&mut map, "model_family", &config.model_family);
    if let Some(preset) = model_preset_id(&config.model_profile, &config.model_family) {
        insert_str(&mut map, "model_preset", preset);
    }
    insert_str(&mut map, "input_device", &config.input_device);
    insert_str(&mut map, "language", &config.language);
    insert_str(&mut map, "shortcut", &config.shortcut);
    insert_str(&mut map, "model_root", &config.model_root);
    insert_str(&mut map, "worker_path", &config.worker_path);
    insert_str(
        &mut map,
        "model_installer_path",
        &config.model_installer_path,
    );
    insert_u32(&mut map, "sample_rate", config.sample_rate);
    insert_u32(&mut map, "num_threads", config.num_threads);
    insert_bool(&mut map, "spoken_punctuation", config.spoken_punctuation);
    insert_bool(&mut map, "insert_partials", config.insert_partials);
    insert_u32(
        &mut map,
        "stream_insert_delay_ms",
        config.stream_insert_delay_ms,
    );
    insert_bool(&mut map, "show_overlay", config.show_overlay);
    map
}

fn default_config_path() -> PathBuf {
    if let Some(value) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(value).join("wordpipe").join("service.json")
    } else if let Some(value) = std::env::var_os("HOME") {
        PathBuf::from(value)
            .join(".config")
            .join("wordpipe")
            .join("service.json")
    } else {
        PathBuf::from("wordpipe-service.json")
    }
}

fn load_service_config(path: &Path) -> Result<ServiceConfig> {
    if !path.exists() {
        let mut config = ServiceConfig::default();
        select_installed_model_profile(&mut config);
        return Ok(config);
    }
    let persisted: PersistedConfig = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let mut config = apply_persisted_config(ServiceConfig::default(), persisted)?;
    select_installed_model_profile(&mut config);
    Ok(config)
}

fn apply_persisted_config(
    mut config: ServiceConfig,
    persisted: PersistedConfig,
) -> Result<ServiceConfig> {
    let migrate_family_from_language = persisted.model_family.is_none();
    if let Some(value) = persisted.backend {
        if !is_backend(&value) {
            return Err(anyhow!("unknown backend in service config: {value}"));
        }
        config.backend = value;
    }
    if let Some(value) = persisted.model_profile {
        if !is_model_profile(&value) {
            return Err(anyhow!("unknown model profile in service config: {value}"));
        }
        config.model_profile = value;
    }
    if let Some(value) = persisted.model_family {
        if !is_model_family(&value) {
            return Err(anyhow!("unknown model family in service config: {value}"));
        }
        config.model_family = value;
    }
    if let Some(value) = persisted.input_device {
        config.input_device = value;
    }
    if let Some(value) = persisted.language {
        let value = normalize_language(value);
        if !is_language(&value) {
            return Err(anyhow!("unknown language in service config: {value}"));
        }
        config.language = value;
    }
    if migrate_family_from_language {
        config.model_family = if is_english_language(&config.language) {
            "english"
        } else {
            "multilingual"
        }
        .to_string();
    }
    if !language_available_for_model_family(&config.model_family, &config.language) {
        config.language = DEFAULT_LANGUAGE.to_string();
    }
    if let Some(value) = persisted.shortcut {
        config.shortcut = value;
    }
    if let Some(value) = persisted.model_root {
        config.model_root = normalize_model_root(value);
    }
    if let Some(value) = persisted.worker_path {
        config.worker_path = normalize_worker_path(value);
    }
    if let Some(value) = persisted.model_installer_path {
        config.model_installer_path = normalize_model_installer_path(value);
    }
    if let Some(value) = persisted.sample_rate {
        if value == 0 {
            return Err(anyhow!("sample_rate must be positive"));
        }
        config.sample_rate = value;
    }
    if let Some(value) = persisted.num_threads {
        if value == 0 {
            return Err(anyhow!("num_threads must be positive"));
        }
        config.num_threads = value;
    }
    if let Some(value) = persisted.spoken_punctuation {
        config.spoken_punctuation = value;
    }
    if let Some(value) = persisted.insert_partials {
        config.insert_partials = value;
    }
    if let Some(value) = persisted.stream_insert_delay_ms {
        config.stream_insert_delay_ms = value;
    }
    if let Some(value) = persisted.show_overlay {
        config.show_overlay = value;
    }
    Ok(config)
}

fn save_service_config(path: &Path, config: &ServiceConfig) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("config path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    let temporary = path.with_file_name(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("service.json"),
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(&PersistedConfig::from(config))?;
    fs::write(&temporary, bytes)
        .with_context(|| format!("failed to write {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| {
        format!(
            "failed to replace {} with {}",
            path.display(),
            temporary.display()
        )
    })?;
    Ok(())
}

fn enumerate_input_devices() -> Result<Vec<VariantMap>> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let mut devices = Vec::new();
    for (index, device) in host
        .input_devices()
        .context("failed to enumerate input devices")?
        .enumerate()
    {
        let name = device.name().unwrap_or_else(|_| "unknown".to_string());
        devices.push(input_device_map(
            index as u32,
            &name,
            default_name.as_ref() == Some(&name),
        ));
    }
    Ok(devices)
}

fn input_device_map(index: u32, name: &str, is_default: bool) -> VariantMap {
    let mut item = VariantMap::new();
    insert_u32(&mut item, "index", index);
    insert_str(&mut item, "name", name);
    insert_str(&mut item, "selector", name);
    insert_bool(&mut item, "is_default", is_default);
    item
}

fn default_model_root() -> String {
    if let Some(value) = std::env::var_os("XDG_DATA_HOME") {
        format!("{}/wordpipe/models", value.to_string_lossy())
    } else if let Some(value) = std::env::var_os("HOME") {
        format!("{}/.local/share/wordpipe/models", value.to_string_lossy())
    } else {
        "wordpipe/models".to_string()
    }
}

fn normalize_model_root(value: String) -> String {
    if value.trim().is_empty() {
        default_model_root()
    } else {
        value
    }
}

fn normalize_worker_path(value: String) -> String {
    normalize_executable_path(value, default_worker_path)
}

fn normalize_language(value: String) -> String {
    let value = value.trim();
    if value.is_empty() {
        DEFAULT_LANGUAGE.to_string()
    } else {
        value.to_string()
    }
}

fn normalize_model_installer_path(value: String) -> String {
    normalize_executable_path(value, default_model_installer_path)
}

fn normalize_executable_path(value: String, default_path: impl FnOnce() -> String) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return default_path();
    }
    if Path::new(trimmed).exists() {
        trimmed.to_string()
    } else {
        let fallback = default_path();
        if Path::new(&fallback).exists() {
            fallback
        } else {
            trimmed.to_string()
        }
    }
}

fn graph_file_valid(runtime_dir: &Path, stem: &str) -> bool {
    if runtime_dir.join(format!("{stem}.ort")).exists() {
        return true;
    }
    let onnx_path = runtime_dir.join(format!("{stem}.onnx"));
    if !onnx_path.exists() {
        return false;
    }
    let data_path = runtime_dir.join(format!("{stem}.onnx.data"));
    if data_path.exists() {
        return true;
    }
    match fs::metadata(&onnx_path) {
        Ok(metadata) if metadata.len() > 128 * 1024 * 1024 => true,
        Ok(_) => !onnx_references_external_data(&onnx_path, &format!("{stem}.onnx.data")),
        Err(_) => false,
    }
}

fn onnx_references_external_data(onnx_path: &Path, marker: &str) -> bool {
    let marker = marker.as_bytes();
    let mut file = match fs::File::open(onnx_path) {
        Ok(file) => file,
        Err(_) => return true,
    };
    let mut buffer = [0_u8; 1024 * 1024];
    let mut carry = Vec::<u8>::new();
    loop {
        let count = match file.read(&mut buffer) {
            Ok(0) => return false,
            Ok(count) => count,
            Err(_) => return true,
        };
        let mut chunk = Vec::with_capacity(carry.len() + count);
        chunk.extend_from_slice(&carry);
        chunk.extend_from_slice(&buffer[..count]);
        if chunk.windows(marker.len()).any(|window| window == marker) {
            return true;
        }
        let carry_len = marker.len().saturating_sub(1).min(chunk.len());
        carry.clear();
        carry.extend_from_slice(&chunk[chunk.len() - carry_len..]);
    }
}

fn model_runtime_dir_valid(runtime_dir: &Path) -> bool {
    model_runtime_structure_valid(runtime_dir)
        && profile_completion_marker_valid_if_present(runtime_dir)
}

fn model_profile_metadata_valid_if_present(runtime_dir: &Path, profile: &str) -> bool {
    let config_path = runtime_dir.join("config.json");
    if !config_path.exists() {
        return true;
    }
    let payload = match fs::read(&config_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<JsonValue>(&bytes).ok())
    {
        Some(payload) => payload,
        None => return false,
    };
    let Some(fixed) = payload.get("fixed_streaming_shapes") else {
        return false;
    };
    let cache_len = match payload.get("model_family").and_then(JsonValue::as_str) {
        Some("english") => 70,
        Some("multilingual") | None => 56,
        Some(_) => return false,
    };
    let expected = [
        ("input_frames", 65_u64),
        ("output_frames", 7),
        ("num_layers", 24),
        ("cache_len", cache_len),
        ("hidden_dim", 1024),
        ("conv_context", 8),
    ];
    if expected
        .iter()
        .any(|(key, value)| fixed.get(*key).and_then(JsonValue::as_u64) != Some(*value))
    {
        return false;
    }
    if payload.get("projected_cache").and_then(JsonValue::as_bool) != Some(true) {
        return false;
    }
    let quantized = payload
        .get("dynamic_quint8_quantization")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    match profile {
        "fast" => !quantized,
        "compact" => quantized,
        _ => false,
    }
}

fn model_runtime_structure_valid(runtime_dir: &Path) -> bool {
    runtime_dir.join("tokenizer.model").exists()
        && graph_file_valid(runtime_dir, "encoder")
        && graph_file_valid(runtime_dir, "decoder_joint")
}

fn profile_completion_marker_valid_if_present(runtime_dir: &Path) -> bool {
    let marker = runtime_dir.join(PROFILE_COMPLETION_MARKER);
    if !marker.exists() {
        return true;
    }
    let payload = match fs::read(&marker)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<JsonValue>(&bytes).ok())
    {
        Some(payload) => payload,
        None => return false,
    };
    if payload.get("format").and_then(JsonValue::as_u64) != Some(1) {
        return false;
    }
    let Some(files) = payload.get("files").and_then(JsonValue::as_array) else {
        return false;
    };
    if files.is_empty() {
        return false;
    }
    files
        .iter()
        .all(|item| marker_file_valid(runtime_dir, item))
}

fn marker_file_valid(runtime_dir: &Path, item: &JsonValue) -> bool {
    let Some(relative) = item.get("path").and_then(JsonValue::as_str) else {
        return false;
    };
    let Some(expected_size) = item.get("size").and_then(JsonValue::as_u64) else {
        return false;
    };
    if !safe_relative_path(relative) {
        return false;
    }
    let path = runtime_dir.join(relative);
    fs::metadata(path)
        .map(|metadata| metadata.len() == expected_size)
        .unwrap_or(false)
}

fn safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !value.is_empty()
        && !path.is_absolute()
        && !path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
}

fn legacy_encoder_graph_exists(runtime_dir: &Path) -> bool {
    runtime_dir.join("encoder.encoder.onnx").exists()
        || runtime_dir.join("encoder.encoder.ort").exists()
}

fn legacy_model_runtime_dir_valid(runtime_dir: &Path) -> bool {
    runtime_dir.join("tokenizer.model").exists() && legacy_encoder_graph_exists(runtime_dir)
}

fn profile_installed(runtime_dir: &str, profile: &str) -> bool {
    let path = Path::new(runtime_dir);
    if model_runtime_dir_valid(path) {
        model_profile_metadata_valid_if_present(path, profile)
    } else {
        legacy_model_runtime_dir_valid(path)
    }
}

fn profile_runtime_dir(model_root: &str, output_name: &str, ort_format: bool) -> String {
    if ort_format {
        format!("{model_root}/{output_name}-ort-format")
    } else {
        format!("{model_root}/{output_name}")
    }
}

fn selected_runtime_dir(config: &ServiceConfig) -> String {
    let Some(profile) = MODEL_PROFILES
        .iter()
        .find(|profile| profile.id == config.model_profile)
    else {
        return config.model_root.clone();
    };
    let output_name = if config.model_family == "english" {
        profile.english_output_name
    } else {
        profile.output_name
    };
    profile_runtime_dir(&config.model_root, output_name, profile.ort_format)
}

fn model_preset_installed(config: &ServiceConfig, preset: &ModelPresetSpec) -> bool {
    let Some(profile) = MODEL_PROFILES
        .iter()
        .find(|profile| profile.id == preset.model_profile)
    else {
        return false;
    };
    let output_name = if preset.model_family == "english" {
        profile.english_output_name
    } else {
        profile.output_name
    };
    let runtime_dir = profile_runtime_dir(&config.model_root, output_name, profile.ort_format);
    profile_installed(&runtime_dir, profile.id)
}

fn select_installed_model_profile(config: &mut ServiceConfig) {
    if profile_installed(&selected_runtime_dir(config), &config.model_profile) {
        return;
    }
    for profile in MODEL_PROFILES {
        let output_name = if config.model_family == "english" {
            profile.english_output_name
        } else {
            profile.output_name
        };
        let runtime_dir = profile_runtime_dir(&config.model_root, output_name, profile.ort_format);
        if profile_installed(&runtime_dir, profile.id) {
            config.model_profile = profile.id.to_string();
            return;
        }
    }
}

fn next_session_id(current: u64) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_micros() as u64)
        .unwrap_or(0);
    now.max(current.saturating_add(1))
}

fn toggle_action(data: &ServiceData) -> ToggleAction {
    if data.listening || data.stopping {
        ToggleAction::Stop
    } else {
        ToggleAction::Start
    }
}

fn apply_worker_exit(data: &mut ServiceData) -> WorkerExit {
    let event = WorkerExit {
        session_id: data.session_id,
        was_active: data.listening || data.stopping || data.loading_model,
        was_expected_shutdown: data.stopping,
    };
    if data.listening || data.loading_model {
        data.last_error = "ASR worker exited unexpectedly".to_string();
    }
    data.listening = false;
    data.stopping = false;
    data.loading_model = false;
    data.model_loaded = false;
    data.worker = None;
    event
}

fn insert_str(map: &mut VariantMap, key: &str, value: &str) {
    map.insert(key.to_string(), owned(Value::from(value.to_string())));
}

fn insert_bool(map: &mut VariantMap, key: &str, value: bool) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn insert_u32(map: &mut VariantMap, key: &str, value: u32) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn insert_u64(map: &mut VariantMap, key: &str, value: u64) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn insert_f64(map: &mut VariantMap, key: &str, value: f64) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn insert_i64(map: &mut VariantMap, key: &str, value: i64) {
    map.insert(key.to_string(), owned(Value::from(value)));
}

fn insert_map(map: &mut VariantMap, key: &str, value: &VariantMap) {
    map.insert(key.to_string(), owned(Value::from(value.clone())));
}

fn owned(value: Value<'_>) -> OwnedValue {
    OwnedValue::try_from(value).expect("D-Bus variant value should be valid")
}

fn get_bool(map: &VariantMap, key: &str) -> Option<bool> {
    map.get(key)
        .and_then(|value| bool::try_from(value.clone()).ok())
}

fn get_u32(map: &VariantMap, key: &str) -> Option<u32> {
    map.get(key)
        .and_then(|value| u32::try_from(value.clone()).ok())
}

fn get_string(map: &VariantMap, key: &str) -> Option<String> {
    map.get(key)
        .and_then(|value| String::try_from(value.clone()).ok())
}

fn fdo_failed(err: anyhow::Error) -> zbus::fdo::Error {
    zbus::fdo::Error::Failed(err.to_string())
}

fn fdo_error_message(err: &zbus::fdo::Error) -> String {
    let message = err.to_string();
    message
        .strip_prefix("org.freedesktop.DBus.Error.Failed: ")
        .unwrap_or(&message)
        .to_string()
}

fn default_worker_path() -> String {
    if let Ok(value) = std::env::var("WORDPIPE_WORKER") {
        return value;
    }
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            let sibling = parent.join("wordpipe-parakeet-worker");
            if sibling.exists() {
                return sibling.to_string_lossy().to_string();
            }
        }
    }
    "wordpipe-parakeet-worker".to_string()
}

fn default_model_installer_path() -> String {
    if let Ok(value) = std::env::var("WORDPIPE_MODEL_INSTALLER") {
        return value;
    }
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            let sibling = parent.join("wordpipe-model-install");
            if sibling.exists() {
                return sibling.to_string_lossy().to_string();
            }
        }
    }
    "wordpipe-model-install".to_string()
}

fn default_ort_dylib_path() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os("ORT_DYLIB_PATH") {
        return Some(PathBuf::from(value));
    }

    for path in [
        PathBuf::from("/app/lib/libonnxruntime.so"),
        PathBuf::from("/app/lib/onnxruntime/libonnxruntime.so"),
    ] {
        if path.exists() {
            return Some(path);
        }
    }

    for venv in candidate_release_venvs() {
        if let Some(path) = find_ort_in_venv(&venv) {
            return Some(path);
        }
    }

    for root in candidate_repo_roots() {
        for venv in [".venv", ".venv-nemo-export"] {
            if let Some(path) = find_ort_in_venv(&root.join(venv)) {
                return Some(path);
            }
        }
    }
    None
}

fn candidate_release_venvs() -> Vec<PathBuf> {
    let mut venvs = Vec::new();
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(prefix) = current_exe.parent().and_then(Path::parent) {
            venvs.push(prefix.join("python-venv"));
        }
    }
    venvs
}

fn candidate_repo_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(current_dir) = std::env::current_dir() {
        roots.push(current_dir);
    }
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(target_dir) = current_exe.parent().and_then(Path::parent) {
            if target_dir.file_name().and_then(|value| value.to_str()) == Some("target") {
                if let Some(repo) = target_dir.parent() {
                    roots.push(repo.to_path_buf());
                }
            }
        }
    }
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        let crate_dir = PathBuf::from(manifest_dir);
        if let Some(repo) = crate_dir.parent().and_then(Path::parent) {
            roots.push(repo.to_path_buf());
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

fn find_ort_in_venv(venv: &Path) -> Option<PathBuf> {
    for lib_dir in ["lib", "lib64"] {
        let Ok(python_dirs) = std::fs::read_dir(venv.join(lib_dir)) else {
            continue;
        };
        for python_dir in python_dirs.flatten() {
            let capi_dir = python_dir
                .path()
                .join("site-packages")
                .join("onnxruntime")
                .join("capi");
            let files = std::fs::read_dir(capi_dir).ok()?;
            for file in files.flatten() {
                let path = file.path();
                let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                    continue;
                };
                if name.starts_with("libonnxruntime.so") {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn spawn_worker(
    config: &ServiceConfig,
    runtime_dir: &str,
) -> Result<(WorkerProcess, std::process::ChildStdout)> {
    let mut command = Command::new(&config.worker_path);
    command
        .arg("--model-dir")
        .arg(runtime_dir)
        .arg("--num-threads")
        .arg(config.num_threads.to_string())
        .arg("--sample-rate")
        .arg(config.sample_rate.to_string())
        .arg("--language")
        .arg(&config.language)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if !config.input_device.is_empty() {
        command.arg("--input-device").arg(&config.input_device);
    }
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        if let Some(path) = default_ort_dylib_path() {
            command.env("ORT_DYLIB_PATH", path);
        }
    }
    apply_worker_thread_env(&mut command, config.num_threads);
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to start ASR worker '{}'; build/install wordpipe-parakeet-worker or set WORDPIPE_WORKER",
            config.worker_path
        )
    })?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("ASR worker stdin was not piped"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("ASR worker stdout was not piped"))?;
    Ok((
        WorkerProcess {
            stdin: Arc::new(Mutex::new(stdin)),
            child,
        },
        stdout,
    ))
}

fn apply_worker_thread_env(command: &mut Command, num_threads: u32) {
    let value = num_threads.to_string();
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"] {
        if std::env::var_os(name).is_none() {
            command.env(name, &value);
        }
    }
}

fn send_worker_command(stdin: &Arc<Mutex<ChildStdin>>, command: &str) -> Result<()> {
    send_worker_message(stdin, &json!({ "command": command }))
}

fn send_worker_language_command(stdin: &Arc<Mutex<ChildStdin>>, language: &str) -> Result<()> {
    send_worker_message(
        stdin,
        &json!({ "command": "set_language", "language": language }),
    )
}

fn send_worker_message(stdin: &Arc<Mutex<ChildStdin>>, message: &JsonValue) -> Result<()> {
    let mut stdin = stdin
        .lock()
        .map_err(|_| anyhow!("ASR worker stdin lock poisoned"))?;
    serde_json::to_writer(&mut *stdin, message)?;
    writeln!(stdin)?;
    stdin.flush()?;
    Ok(())
}

fn shutdown_worker(data: &mut ServiceData) {
    if let Some(mut worker) = data.worker.take() {
        let _ = send_worker_command(&worker.stdin, "shutdown");
        let _ = worker.child.kill();
        let _ = worker.child.wait();
    }
    data.listening = false;
    data.stopping = false;
    data.loading_model = false;
    data.model_loaded = false;
    data.partial_text.clear();
}

fn normalize_spoken_punctuation(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    normalize_spoken_words(&words)
}

fn normalize_spoken_punctuation_partial(text: &str) -> String {
    let mut words: Vec<&str> = text.split_whitespace().collect();
    if words
        .last()
        .is_some_and(|word| is_incomplete_command_prefix(word))
    {
        words.pop();
    }
    normalize_spoken_words(&words)
}

fn normalize_spoken_words(words: &[&str]) -> String {
    let mut output = String::new();
    let mut index = 0;
    while index < words.len() {
        if let Some((command, consumed)) = match_spoken_command(&words, index) {
            if matches!(command, "," | "." | "?" | "!" | ":" | ";") {
                while output.ends_with(' ') {
                    output.pop();
                }
                output.push_str(command);
                output.push(' ');
            } else {
                while output.ends_with(' ') {
                    output.pop();
                }
                output.push_str(command);
            }
            index += consumed;
            continue;
        }
        append_spoken_word(&mut output, words[index]);
        index += 1;
    }
    output.trim_end_matches(' ').to_string()
}

fn is_incomplete_command_prefix(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "full" | "question" | "exclamation" | "new"
    )
}

fn match_spoken_command(words: &[&str], index: usize) -> Option<(&'static str, usize)> {
    let remaining = words.len().saturating_sub(index);
    for size in (1..=remaining.min(2)).rev() {
        let first = words[index].to_ascii_lowercase();
        let second = if size == 2 {
            Some(words[index + 1].to_ascii_lowercase())
        } else {
            None
        };
        let command = match (first.as_str(), second.as_deref()) {
            ("comma", None) => Some(","),
            ("period", None) => Some("."),
            ("full", Some("stop")) => Some("."),
            ("question", Some("mark")) => Some("?"),
            ("exclamation", Some("point")) => Some("!"),
            ("exclamation", Some("mark")) => Some("!"),
            ("colon", None) => Some(":"),
            ("semicolon", None) => Some(";"),
            ("new", Some("line")) => Some("\n"),
            ("newline", None) => Some("\n"),
            ("new", Some("paragraph")) => Some("\n\n"),
            _ => None,
        };
        if let Some(command) = command {
            return Some((command, size));
        }
    }
    None
}

fn append_spoken_word(output: &mut String, word: &str) {
    if output.is_empty() || output.ends_with('\n') || output.ends_with(' ') {
        output.push_str(word);
        output.push(' ');
    } else {
        output.push(' ');
        output.push_str(word);
        output.push(' ');
    }
}

fn run_progress_command(
    mut command: Command,
    profile: &str,
    emitter: &SignalEmitter<'static>,
    data: Arc<Mutex<ServiceData>>,
) -> Result<()> {
    eprintln!("wordpipe-service: starting model installer for {profile}: {command:?}");
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to start model installer {:?}", command))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("model installer stdout was not piped"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("model installer stderr was not piped"))?;

    let stdout_profile = profile.to_string();
    let stdout_emitter = emitter.clone();
    let stdout_data = Arc::clone(&data);
    let stdout_thread = std::thread::spawn(move || {
        stream_progress_lines(stdout, &stdout_profile, &stdout_emitter, stdout_data)
    });

    let stderr_profile = profile.to_string();
    let stderr_emitter = emitter.clone();
    let stderr_thread = std::thread::spawn(move || {
        stream_progress_lines(stderr, &stderr_profile, &stderr_emitter, data)
    });

    let status = child.wait()?;
    let _ = stdout_thread.join();
    let _ = stderr_thread.join();
    if status.success() {
        eprintln!("wordpipe-service: model installer for {profile} completed successfully");
        Ok(())
    } else {
        eprintln!("wordpipe-service: model installer for {profile} exited with {status}");
        Err(anyhow!("model installer exited with {status}"))
    }
}

fn stream_progress_lines<R>(
    reader: R,
    profile: &str,
    emitter: &SignalEmitter<'static>,
    data: Arc<Mutex<ServiceData>>,
) where
    R: std::io::Read,
{
    let reader = std::io::BufReader::new(reader);
    for line in reader.lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        eprintln!(
            "wordpipe-service: model installer {profile}: {}",
            line.trim()
        );
        let progress = progress_from_line(profile, line.trim());
        if let Ok(mut data) = data.lock() {
            data.last_install_progress = progress.clone();
        }
        let _ = zbus::block_on(WordpipeService::install_progress(
            emitter, profile, progress,
        ));
    }
}

fn progress_from_line(profile: &str, line: &str) -> VariantMap {
    const PREFIX: &str = "wordpipe-progress ";
    if let Some(payload) = line.strip_prefix(PREFIX) {
        if let Ok(value) = serde_json::from_str::<JsonValue>(payload) {
            let mut progress = json_to_variant_map(&value);
            insert_str(&mut progress, "profile", profile);
            return progress;
        }
    }

    let mut progress = VariantMap::new();
    insert_str(&mut progress, "profile", profile);
    insert_str(&mut progress, "phase", "running");
    insert_str(&mut progress, "message", line);
    insert_f64(&mut progress, "fraction", 0.0);
    progress
}

fn json_to_variant_map(value: &JsonValue) -> VariantMap {
    let mut map = VariantMap::new();
    let Some(object) = value.as_object() else {
        return map;
    };
    for (key, value) in object {
        match value {
            JsonValue::Bool(value) => insert_bool(&mut map, key, *value),
            JsonValue::Number(number) => {
                if let Some(value) = number.as_u64() {
                    insert_u64(&mut map, key, value);
                } else if let Some(value) = number.as_i64() {
                    insert_i64(&mut map, key, value);
                } else if let Some(value) = number.as_f64() {
                    insert_f64(&mut map, key, value);
                }
            }
            JsonValue::String(value) => insert_str(&mut map, key, value),
            _ => {}
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_keys(map: &VariantMap) -> Vec<&str> {
        let mut keys = map.keys().map(String::as_str).collect::<Vec<_>>();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn persisted_config_overrides_defaults() {
        let config = apply_persisted_config(
            ServiceConfig::default(),
            PersistedConfig {
                model_profile: Some("compact".to_string()),
                input_device: Some("pipewire".to_string()),
                language: Some("en-GB".to_string()),
                num_threads: Some(4),
                sample_rate: Some(16_000),
                show_overlay: Some(false),
                ..PersistedConfig::default()
            },
        )
        .unwrap();

        assert_eq!(config.model_profile, "compact");
        assert_eq!(config.model_family, "english");
        assert_eq!(config.input_device, "pipewire");
        assert_eq!(config.language, "en-GB");
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.sample_rate, 16_000);
        assert!(!config.show_overlay);
        assert_eq!(config.backend, DEFAULT_BACKEND);
    }

    #[test]
    fn legacy_config_infers_family_from_saved_language() {
        let english = apply_persisted_config(
            ServiceConfig::default(),
            PersistedConfig {
                language: Some("en-GB".to_string()),
                ..PersistedConfig::default()
            },
        )
        .unwrap();
        let multilingual = apply_persisted_config(
            ServiceConfig::default(),
            PersistedConfig {
                language: Some("fr-FR".to_string()),
                ..PersistedConfig::default()
            },
        )
        .unwrap();

        assert_eq!(english.model_family, "english");
        assert_eq!(multilingual.model_family, "multilingual");
    }

    #[test]
    fn explicit_english_family_rejects_non_english_saved_language() {
        let config = apply_persisted_config(
            ServiceConfig::default(),
            PersistedConfig {
                model_family: Some("english".to_string()),
                language: Some("fr-FR".to_string()),
                ..PersistedConfig::default()
            },
        )
        .unwrap();

        assert_eq!(config.model_family, "english");
        assert_eq!(config.language, DEFAULT_LANGUAGE);
    }

    #[test]
    fn config_map_has_contract_keys() {
        let config = ServiceConfig::default();

        assert_eq!(
            sorted_keys(&config_map(&config)),
            vec![
                "backend",
                "input_device",
                "insert_partials",
                "language",
                "model_family",
                "model_installer_path",
                "model_preset",
                "model_profile",
                "model_root",
                "num_threads",
                "sample_rate",
                "shortcut",
                "show_overlay",
                "spoken_punctuation",
                "stream_insert_delay_ms",
                "worker_path",
            ]
        );
    }

    #[test]
    fn state_map_has_contract_keys() {
        let data = ServiceData::default();

        assert_eq!(
            sorted_keys(&state_map(&data)),
            vec![
                "backend",
                "input_device",
                "installing",
                "installing_profile",
                "last_commit_text",
                "last_error",
                "last_install_progress",
                "last_metrics",
                "listening",
                "loading_model",
                "model_family",
                "model_loaded",
                "model_preset",
                "model_profile",
                "partial_text",
                "selected_model_installed",
                "selected_runtime_dir",
                "seq",
                "session_id",
                "stopping",
            ]
        );
    }

    #[test]
    fn list_backends_has_contract_keys() {
        let service = WordpipeService::new(PathBuf::from("unused.json"), ServiceConfig::default());
        let backends = service.list_backends();

        assert!(!backends.is_empty());
        assert_eq!(
            sorted_keys(&backends[0]),
            vec!["description", "id", "title"]
        );
    }

    #[test]
    fn list_model_profiles_has_contract_keys() {
        let root = unique_temp_dir("profile-contract");
        fs::create_dir_all(&root).unwrap();
        let config = ServiceConfig {
            model_root: root.to_string_lossy().to_string(),
            ..ServiceConfig::default()
        };
        let service = WordpipeService::new(PathBuf::from("unused.json"), config);

        let profiles = service.list_model_profiles();

        assert!(!profiles.is_empty());
        assert_eq!(profiles.len(), 4);
        assert_eq!(
            sorted_keys(&profiles[0]),
            vec![
                "build_profile",
                "description",
                "id",
                "installed",
                "model_family",
                "model_profile",
                "ort_format",
                "output_name",
                "prebuilt_repo",
                "runtime_dir",
                "title",
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn installer_command_downloads_only_the_selected_family() {
        let command = model_installer_command(
            "/usr/bin/wordpipe-model-install",
            "/models",
            "compact",
            "english",
        );
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert_eq!(
            args,
            [
                "--profile",
                "compact",
                "--model-family",
                "english",
                "--model-root",
                "/models",
            ]
        );
    }

    #[test]
    fn selecting_english_preset_keeps_paths_and_limits_language() {
        let mut config = ServiceConfig {
            model_profile: "compact".to_string(),
            model_family: "multilingual".to_string(),
            language: "fr-FR".to_string(),
            model_root: "/models".to_string(),
            ..ServiceConfig::default()
        };

        assert!(apply_model_preset(
            &mut config,
            model_preset("compact-english").unwrap()
        ));

        assert_eq!(config.model_profile, "compact");
        assert_eq!(config.model_family, "english");
        assert_eq!(config.language, DEFAULT_LANGUAGE);
        assert_eq!(
            selected_runtime_dir(&config),
            "/models/nemotron-wordpipe-en-compact-fixed-shape-ort-format"
        );
    }

    #[test]
    fn selecting_multilingual_preset_does_not_change_existing_model_path() {
        let mut config = ServiceConfig {
            model_profile: "compact".to_string(),
            model_family: "english".to_string(),
            language: "en-US".to_string(),
            model_root: "/models".to_string(),
            ..ServiceConfig::default()
        };

        assert!(apply_model_preset(
            &mut config,
            model_preset("compact").unwrap()
        ));

        assert_eq!(config.model_family, "multilingual");
        assert_eq!(config.language, "en-US");
        assert_eq!(
            selected_runtime_dir(&config),
            "/models/nemotron-wordpipe-compact-fixed-shape-ort-format"
        );
    }

    #[test]
    fn input_device_map_has_contract_keys() {
        let device = input_device_map(7, "Microphone", true);

        assert_eq!(
            sorted_keys(&device),
            vec!["index", "is_default", "name", "selector"]
        );
        assert_eq!(u32::try_from(device["index"].clone()).unwrap(), 7);
        assert_eq!(
            String::try_from(device["selector"].clone()).unwrap(),
            "Microphone"
        );
        assert_eq!(bool::try_from(device["is_default"].clone()).unwrap(), true);
    }

    #[test]
    fn empty_persisted_model_root_uses_default() {
        let config = apply_persisted_config(
            ServiceConfig::default(),
            PersistedConfig {
                model_root: Some("  ".to_string()),
                ..PersistedConfig::default()
            },
        )
        .unwrap();

        assert_eq!(config.model_root, default_model_root());
    }

    #[test]
    fn empty_persisted_runtime_paths_use_defaults() {
        let config = apply_persisted_config(
            ServiceConfig::default(),
            PersistedConfig {
                worker_path: Some(String::new()),
                model_installer_path: Some("  ".to_string()),
                ..PersistedConfig::default()
            },
        )
        .unwrap();

        assert_eq!(config.worker_path, default_worker_path());
        assert_eq!(config.model_installer_path, default_model_installer_path());
    }

    #[test]
    fn missing_runtime_path_uses_existing_default_path() {
        let root = unique_temp_dir("runtime-path-default");
        fs::create_dir_all(&root).unwrap();
        let fallback = root.join("wordpipe-parakeet-worker");
        fs::write(&fallback, b"worker").unwrap();

        let normalized =
            normalize_executable_path("/tmp/wordpipe-stale-missing-worker".to_string(), || {
                fallback.to_string_lossy().to_string()
            });

        assert_eq!(normalized, fallback.to_string_lossy());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn worker_thread_env_matches_num_threads() {
        let mut command = Command::new("wordpipe-parakeet-worker");
        apply_worker_thread_env(&mut command, 2);
        let envs: HashMap<_, _> = command
            .get_envs()
            .filter_map(|(name, value)| Some((name.to_str()?, value?.to_str()?)))
            .collect();

        assert_eq!(envs.get("OMP_NUM_THREADS"), Some(&"2"));
        assert_eq!(envs.get("MKL_NUM_THREADS"), Some(&"2"));
        assert_eq!(envs.get("OPENBLAS_NUM_THREADS"), Some(&"2"));
    }

    #[test]
    fn find_ort_in_venv_supports_release_lib64_layout() {
        let root = unique_temp_dir("ort-venv-lib64");
        let capi = root
            .join("lib64")
            .join("python3.14")
            .join("site-packages")
            .join("onnxruntime")
            .join("capi");
        fs::create_dir_all(&capi).unwrap();
        let expected = capi.join("libonnxruntime.so.1.27.0");
        fs::write(&expected, b"ort").unwrap();

        assert_eq!(find_ort_in_venv(&root), Some(expected));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_external_onnx_data_is_not_installed() {
        let root = unique_temp_dir("missing-external-data");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("tokenizer.model"), b"tokenizer").unwrap();
        fs::write(root.join("encoder.onnx"), b"external encoder.onnx.data").unwrap();
        fs::write(root.join("decoder_joint.onnx"), b"decoder").unwrap();

        assert!(!profile_installed(
            root.to_string_lossy().as_ref(),
            "compact"
        ));

        fs::write(root.join("encoder.onnx.data"), b"weights").unwrap();
        assert!(profile_installed(
            root.to_string_lossy().as_ref(),
            "compact"
        ));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn profile_marker_must_match_present_file_sizes() {
        let root = unique_temp_dir("profile-marker-size");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("tokenizer.model"), b"tokenizer").unwrap();
        fs::write(root.join("encoder.ort"), b"encoder").unwrap();
        fs::write(root.join("decoder_joint.ort"), b"decoder").unwrap();
        fs::write(
            root.join(PROFILE_COMPLETION_MARKER),
            json!({
                "format": 1,
                "profile": "compact",
                "files": [
                    {"path": "tokenizer.model", "size": 9, "sha256": "unused"},
                    {"path": "encoder.ort", "size": 7, "sha256": "unused"},
                    {"path": "decoder_joint.ort", "size": 7, "sha256": "unused"}
                ]
            })
            .to_string(),
        )
        .unwrap();

        assert!(profile_installed(
            root.to_string_lossy().as_ref(),
            "compact"
        ));

        fs::write(root.join("decoder_joint.ort"), b"changed!").unwrap();
        assert!(!profile_installed(
            root.to_string_lossy().as_ref(),
            "compact"
        ));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn profile_marker_rejects_unsafe_paths() {
        let root = unique_temp_dir("profile-marker-unsafe");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("tokenizer.model"), b"tokenizer").unwrap();
        fs::write(root.join("encoder.ort"), b"encoder").unwrap();
        fs::write(root.join("decoder_joint.ort"), b"decoder").unwrap();
        fs::write(
            root.join(PROFILE_COMPLETION_MARKER),
            json!({
                "format": 1,
                "profile": "compact",
                "files": [
                    {"path": "../tokenizer.model", "size": 9, "sha256": "unused"}
                ]
            })
            .to_string(),
        )
        .unwrap();

        assert!(!profile_installed(
            root.to_string_lossy().as_ref(),
            "compact"
        ));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn profile_metadata_rejects_non_fixed_shape_fast_export() {
        let root = unique_temp_dir("profile-metadata-fast");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("tokenizer.model"), b"tokenizer").unwrap();
        fs::write(root.join("encoder.onnx"), b"encoder").unwrap();
        fs::write(root.join("decoder_joint.onnx"), b"decoder").unwrap();
        fs::write(
            root.join("config.json"),
            json!({
                "projected_cache": true,
                "dynamic_quint8_quantization": false
            })
            .to_string(),
        )
        .unwrap();

        assert!(!profile_installed(root.to_string_lossy().as_ref(), "fast"));

        fs::write(
            root.join("config.json"),
            json!({
                "projected_cache": true,
                "dynamic_quint8_quantization": false,
                "fixed_streaming_shapes": {
                    "input_frames": 65,
                    "output_frames": 7,
                    "num_layers": 24,
                    "cache_len": 56,
                    "hidden_dim": 1024,
                    "conv_context": 8
                }
            })
            .to_string(),
        )
        .unwrap();
        assert!(profile_installed(root.to_string_lossy().as_ref(), "fast"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn persisted_config_rejects_invalid_profile() {
        let err = apply_persisted_config(
            ServiceConfig::default(),
            PersistedConfig {
                model_profile: Some("tiny".to_string()),
                ..PersistedConfig::default()
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("unknown model profile"));
    }

    #[test]
    fn selects_installed_profile_when_default_is_missing() {
        let root = unique_temp_dir("installed-profile");
        let compact_runtime = root.join("nemotron-wordpipe-compact-fixed-shape-ort-format");
        write_test_runtime_profile(&compact_runtime);
        let mut config = ServiceConfig {
            model_root: root.to_string_lossy().to_string(),
            model_profile: "fast".to_string(),
            model_family: "multilingual".to_string(),
            language: "auto".to_string(),
            ..ServiceConfig::default()
        };

        select_installed_model_profile(&mut config);

        assert_eq!(config.model_profile, "compact");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn persisted_uninstalled_profile_falls_back_to_installed_profile() {
        let root = unique_temp_dir("persisted-installed-profile");
        let config_path = root.join("service.json");
        let model_root = root.join("models");
        let compact_runtime = model_root.join("nemotron-wordpipe-compact-fixed-shape-ort-format");
        write_test_runtime_profile(&compact_runtime);
        save_service_config(
            &config_path,
            &ServiceConfig {
                model_root: model_root.to_string_lossy().to_string(),
                model_profile: "fast".to_string(),
                model_family: "multilingual".to_string(),
                language: "auto".to_string(),
                ..ServiceConfig::default()
            },
        )
        .unwrap();

        let config = load_service_config(&config_path).unwrap();

        assert_eq!(config.model_profile, "compact");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn keeps_selected_profile_when_it_is_installed() {
        let root = unique_temp_dir("selected-profile");
        let fast_runtime = root.join("nemotron-wordpipe-fast-fp32-projected");
        let compact_runtime = root.join("nemotron-wordpipe-compact-fixed-shape-ort-format");
        write_test_runtime_profile(&fast_runtime);
        write_test_runtime_profile(&compact_runtime);
        let mut config = ServiceConfig {
            model_root: root.to_string_lossy().to_string(),
            model_profile: "fast".to_string(),
            model_family: "multilingual".to_string(),
            language: "auto".to_string(),
            ..ServiceConfig::default()
        };

        select_installed_model_profile(&mut config);

        assert_eq!(config.model_profile, "fast");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unavailable_model_presets_cannot_be_selected() {
        let root = unique_temp_dir("selectable-model-presets");
        let compact_runtime = root.join("nemotron-wordpipe-compact-fixed-shape-ort-format");
        write_test_runtime_profile(&compact_runtime);
        let config = ServiceConfig {
            model_root: root.to_string_lossy().to_string(),
            ..ServiceConfig::default()
        };

        assert!(model_preset_installed(
            &config,
            model_preset("compact").unwrap()
        ));
        assert!(!model_preset_installed(
            &config,
            model_preset("compact-english").unwrap()
        ));
        assert!(!model_preset_installed(
            &config,
            model_preset("fast").unwrap()
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn state_includes_selected_model_install_status() {
        let root = unique_temp_dir("state-model-installed");
        let compact_runtime = root.join("nemotron-wordpipe-compact-fixed-shape-ort-format");
        write_test_runtime_profile(&compact_runtime);
        let data = ServiceData {
            config: ServiceConfig {
                model_root: root.to_string_lossy().to_string(),
                model_profile: "compact".to_string(),
                model_family: "multilingual".to_string(),
                language: "auto".to_string(),
                ..ServiceConfig::default()
            },
            ..ServiceData::default()
        };

        let state = state_map(&data);

        assert_eq!(
            bool::try_from(state["selected_model_installed"].clone()).unwrap(),
            true
        );
        assert_eq!(
            String::try_from(state["selected_runtime_dir"].clone()).unwrap(),
            compact_runtime.to_string_lossy()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selected_runtime_uses_explicit_english_model_family() {
        let root = unique_temp_dir("english-runtime-selection");
        let config = ServiceConfig {
            model_root: root.to_string_lossy().to_string(),
            model_profile: "fast".to_string(),
            model_family: "english".to_string(),
            language: "en-GB".to_string(),
            ..ServiceConfig::default()
        };

        assert_eq!(
            selected_runtime_dir(&config),
            root.join("nemotron-wordpipe-en-fast-fp32-projected")
                .to_string_lossy()
        );
    }

    #[test]
    fn selected_runtime_uses_explicit_multilingual_model_for_any_language() {
        let root = unique_temp_dir("multilingual-runtime-selection");
        for language in ["auto", "fr-FR"] {
            let config = ServiceConfig {
                model_root: root.to_string_lossy().to_string(),
                model_profile: "fast".to_string(),
                model_family: "multilingual".to_string(),
                language: language.to_string(),
                ..ServiceConfig::default()
            };
            assert_eq!(
                selected_runtime_dir(&config),
                root.join("nemotron-wordpipe-fast-fp32-projected")
                    .to_string_lossy()
            );
        }
    }

    #[test]
    fn state_includes_installing_profile() {
        let data = ServiceData {
            installing: true,
            installing_profile: "compact".to_string(),
            ..ServiceData::default()
        };

        let state = state_map(&data);

        assert_eq!(bool::try_from(state["installing"].clone()).unwrap(), true);
        assert_eq!(
            String::try_from(state["installing_profile"].clone()).unwrap(),
            "compact"
        );
    }

    #[test]
    fn state_includes_last_metrics() {
        let mut metrics = VariantMap::new();
        insert_f64(&mut metrics, "real_time_factor", 0.25);
        insert_u64(&mut metrics, "decode_calls", 3);
        let data = ServiceData {
            last_metrics: metrics,
            ..ServiceData::default()
        };

        let state = state_map(&data);
        let last_metrics =
            <HashMap<String, OwnedValue>>::try_from(state["last_metrics"].clone()).unwrap();

        assert_eq!(
            f64::try_from(last_metrics["real_time_factor"].clone()).unwrap(),
            0.25
        );
        assert_eq!(
            u64::try_from(last_metrics["decode_calls"].clone()).unwrap(),
            3
        );
    }

    #[test]
    fn state_includes_last_install_progress() {
        let mut progress = VariantMap::new();
        insert_str(&mut progress, "profile", "compact");
        insert_str(&mut progress, "phase", "running");
        insert_str(&mut progress, "message", "downloading");
        let data = ServiceData {
            last_install_progress: progress,
            ..ServiceData::default()
        };

        let state = state_map(&data);
        let last_progress =
            <HashMap<String, OwnedValue>>::try_from(state["last_install_progress"].clone())
                .unwrap();

        assert_eq!(
            String::try_from(last_progress["profile"].clone()).unwrap(),
            "compact"
        );
        assert_eq!(
            String::try_from(last_progress["message"].clone()).unwrap(),
            "downloading"
        );
    }

    #[test]
    fn state_includes_transcript_text() {
        let data = ServiceData {
            partial_text: "hello wor".to_string(),
            last_commit_text: "hello world".to_string(),
            ..ServiceData::default()
        };

        let state = state_map(&data);

        assert_eq!(
            String::try_from(state["partial_text"].clone()).unwrap(),
            "hello wor"
        );
        assert_eq!(
            String::try_from(state["last_commit_text"].clone()).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn toggle_starts_when_idle() {
        let data = ServiceData::default();

        assert_eq!(toggle_action(&data), ToggleAction::Start);
    }

    #[test]
    fn toggle_stops_while_listening_or_stopping() {
        let listening = ServiceData {
            listening: true,
            ..ServiceData::default()
        };
        let stopping = ServiceData {
            stopping: true,
            ..ServiceData::default()
        };

        assert_eq!(toggle_action(&listening), ToggleAction::Stop);
        assert_eq!(toggle_action(&stopping), ToggleAction::Stop);
    }

    #[test]
    fn worker_exit_marks_active_session_stopped_with_error() {
        let mut data = ServiceData {
            listening: true,
            session_id: 42,
            model_loaded: true,
            ..ServiceData::default()
        };

        let exit = apply_worker_exit(&mut data);

        assert_eq!(
            exit,
            WorkerExit {
                session_id: 42,
                was_active: true,
                was_expected_shutdown: false,
            }
        );
        assert!(!data.listening);
        assert!(!data.model_loaded);
        assert_eq!(data.last_error, "ASR worker exited unexpectedly");
    }

    #[test]
    fn worker_exit_during_stopping_is_expected() {
        let mut data = ServiceData {
            stopping: true,
            session_id: 43,
            model_loaded: true,
            ..ServiceData::default()
        };

        let exit = apply_worker_exit(&mut data);

        assert_eq!(
            exit,
            WorkerExit {
                session_id: 43,
                was_active: true,
                was_expected_shutdown: true,
            }
        );
        assert!(!data.stopping);
        assert!(data.last_error.is_empty());
    }

    #[test]
    fn fdo_error_message_strips_failed_prefix() {
        let err = zbus::fdo::Error::Failed("model missing".to_string());

        assert_eq!(fdo_error_message(&err), "model missing");
    }

    #[test]
    fn normalizes_spoken_punctuation_commands() {
        assert_eq!(
            normalize_spoken_punctuation(
                "hello comma world period new line second paragraph question mark"
            ),
            "hello, world.\nsecond paragraph?"
        );
        assert_eq!(
            normalize_spoken_punctuation("wait full stop no exclamation point"),
            "wait. no!"
        );
    }

    #[test]
    fn normalizes_spoken_punctuation_without_touching_unknown_words() {
        assert_eq!(
            normalize_spoken_punctuation("new deal comma newline done"),
            "new deal,\ndone"
        );
    }

    #[test]
    fn partial_normalization_holds_incomplete_commands() {
        assert_eq!(normalize_spoken_punctuation_partial("new"), "");
        assert_eq!(
            normalize_spoken_punctuation_partial("hello question"),
            "hello"
        );
        assert_eq!(
            normalize_spoken_punctuation_partial("hello question mark"),
            "hello?"
        );
        assert_eq!(normalize_spoken_punctuation_partial("new deal"), "new deal");
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "wordpipe-service-test-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn write_test_runtime_profile(path: &Path) {
        fs::create_dir_all(path).unwrap();
        fs::write(path.join("tokenizer.model"), b"tokenizer").unwrap();
        fs::write(path.join("encoder.ort"), b"encoder").unwrap();
        fs::write(path.join("decoder_joint.ort"), b"decoder").unwrap();
    }
}
