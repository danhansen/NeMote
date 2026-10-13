// A memory-only service facade. It never connects to D-Bus, launches workers,
// downloads models, or writes the user's extension/service configuration.
import Adw from 'gi://Adw';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Gtk from 'gi://Gtk';
import {createPreferencesPage} from '../extensions/gnome-shell/nemote@dhansen.dev/preferencesPage.js';

export const SCENARIOS = ['ready', 'multilingual', 'missing', 'downloading',
    'converting', 'loading', 'listening', 'stopping', 'error', 'unavailable'];
export const LONG_TEXT = 'Can you meet me there at 3 p.m.? Please bring the quarterly report, '
    + 'the engineering notes, and the revised schedule for Dr. García. '
    + 'The budget is $1,250 & the target is < 5%. '.repeat(5);

export function* descendants(widget) {
    yield widget;
    for (let child = widget.get_first_child(); child; child = child.get_next_sibling())
        yield* descendants(child);
}

export function createPreview(schemaDirectory, scenario = 'ready', width = 680) {
    const source = Gio.SettingsSchemaSource.new_from_directory(schemaDirectory,
        Gio.SettingsSchemaSource.get_default(), false);
    const settings = new Gio.Settings({settings_schema:
        source.lookup('org.gnome.shell.extensions.nemote', false),
    backend: Gio.memory_settings_backend_new()});
    settings.set_string('model-family', 'english');
    settings.set_boolean('itn', true);
    settings.set_boolean('pnc', true);
    settings.set_string('endpoint-mode', 'preview');
    settings.set_string('boost-phrases', 'NeMote\nDr. García\nNemotron\nAVX-VNNI');
    const page = createPreferencesPage(settings, {connectService: false});
    const window = new Adw.PreferencesWindow({
        title: 'NeMote UI Preview — Simulated Service',
        default_width: width, default_height: 850,
    });
    for (const preferencesPage of page.pages)
        window.add(preferencesPage);
    let currentState = {};
    let currentScenario = scenario;
    const scenarioListeners = [];
    let timer = 0;
    const calls = [];
    const config = () => ({model_profile: 'nemo-q8',
        model_family: settings.get_string('model-family'),
        model_preset: settings.get_string('model-family') === 'english' ? 'nemo-q8-english' : 'nemo-q8',
        supported_streaming_latencies_ms: [80, 160, 560, 1120]});
    const devices = [
        {name: 'Core Ultra 200V Series Processors HD Audio Digital Microphone', selector: 'long-microphone'},
        {name: 'MX Brio Analog Stereo', selector: 'usb-microphone', is_default: true},
    ];
    function refresh() {
        page._rebuildProfileOptions();
        page._rebuildProfileRows();
        page._syncFromConfig(config());
        page._refreshInputDevices();
        page._handleState(currentState);
    }
    function setScenario(name) {
        if (!SCENARIOS.includes(name))
            throw new Error(`Unknown scenario: ${name}`);
        currentScenario = name;
        if (timer) { GLib.Source.remove(timer); timer = 0; }
        page._lastInstallProgress = {};
        page._profiles.forEach(profile => { profile.installed = name !== 'missing'; });
        page._pncModels = [
            {id: 'pnc-bert', title: 'English PnC — BERT', installed: name !== 'converting',
                description: '139 MB · English punctuation and capitalization'},
            {id: 'pnc-distilbert', title: 'English PnC — DistilBERT', installed: false,
                description: '94 MB · Smaller, faster punctuation and capitalization model'},
        ];
        settings.set_string('model-family', name === 'multilingual' ? 'multilingual' : 'english');
        settings.set_string('input-device', 'long-microphone');
        currentState = {selected_model_installed: name !== 'missing',
            selected_pnc_model_installed: name !== 'converting',
            partial_text: '', last_commit_text: '', last_error: ''};
        if (name === 'downloading' || name === 'converting') {
            const profile = name === 'downloading' ? 'nemo-q8' : 'pnc-bert';
            page._profiles.find(p => p.id === 'nemo-q8').installed = false;
            Object.assign(currentState, {installing: true, installing_profile: profile,
                last_install_progress: {profile, phase: name === 'downloading' ? 'downloading' : 'converting',
                    message: name === 'downloading'
                        ? 'Downloading nemotron-speech-streaming-multilingual-0.6b.q8_0.gguf: 532 MB / 1.06 GB (50%)'
                        : 'Converting NVIDIA punctuation checkpoint to GGUF. This may take several minutes.',
                    ...(name === 'downloading' ? {fraction: 0.5} : {})}});
        }
        if (name === 'loading')
            Object.assign(currentState, {loading_model: true, loading_message: 'Loading Nemotron model and punctuation model…'});
        if (name === 'listening' || name === 'stopping')
            Object.assign(currentState, {[name]: true, partial_text: LONG_TEXT, last_commit_text: LONG_TEXT,
                last_metrics: {real_audio_real_time_factor: 0.42, audio_seconds: 12.4, decode_seconds: 5.2}});
        if (name === 'error')
            currentState.last_error = 'Model installation failed: not enough disk space in /home/example/.local/share/nemote/models. Free some space and retry. Network response: <error> & unavailable.';
        refresh();
        if (name === 'unavailable') {
            page._statusRow.subtitle = 'Service unavailable';
            page._startButton.sensitive = false;
            page._stopButton.sensitive = false;
        }
        for (const listener of scenarioListeners)
            listener(name);
    }
    page._callRemote = (method, ...args) => {
        calls.push({method, args: args.filter(arg => typeof arg !== 'function')});
        const callback = typeof args.at(-1) === 'function' ? args.pop() : null;
        if (method === 'ListInputDevices') return callback?.(devices);
        else if (method === 'ListModelProfiles') return callback?.(page._profiles);
        else if (method === 'ListPncModels') return callback?.(page._pncModels);
        else if (method === 'GetConfig') return callback?.(config());
        else if (method === 'GetState') return callback?.(currentState);
        else if (method === 'Start') setScenario('listening');
        else if (method === 'Stop') setScenario('ready');
        else if (method === 'InstallModel') {
            const profile = args[0];
            currentState = {...currentState, installing: true, installing_profile: profile};
            page._handleState(currentState);
            let fraction = 0;
            timer = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 200, () => {
                fraction = Math.min(1, fraction + 0.1);
                page._handleInstallProgress(profile, {profile, fraction,
                    phase: fraction === 1 ? 'complete' : 'downloading',
                    message: fraction === 1 ? 'Installed (simulated)' : `Downloading (simulated): ${Math.round(fraction * 100)}%`});
                if (fraction < 1) return GLib.SOURCE_CONTINUE;
                [...page._profiles, ...page._pncModels].find(p => p.id === profile).installed = true;
                currentState = {...currentState, installing: false, installing_profile: '',
                    selected_pnc_model_installed: !settings.get_boolean('pnc') ||
                        page._pncModels.some(p => p.id === `pnc-${settings.get_string('pnc-variant')}` && p.installed),
                    selected_model_installed: page._profiles.some(p => p.id === config().model_preset && p.installed)};
                refresh(); timer = 0; return GLib.SOURCE_REMOVE;
            });
        }
        else if (method === 'SetModelProfile') {
            settings.set_string('model-family', args[0] === 'nemo-q8-english' ? 'english' : 'multilingual');
            page._syncFromConfig(config());
        }
        callback?.();
    };
    setScenario(scenario);
    return {window, page, settings, calls, setScenario,
        get scenario() { return currentScenario; },
        onScenarioChanged(listener) { scenarioListeners.push(listener); },
        close() {
            if (timer) { GLib.Source.remove(timer); timer = 0; }
            scenarioListeners.length = 0;
            page._shortcutButton._editor?.close();
            page._shortcutButton._editor?.run_dispose();
            window.close(); window.run_dispose();
        }};
}

export function addPreviewControls(preview) {
    const page = new Adw.PreferencesPage({title: 'Preview Controls', icon_name: 'applications-engineering-symbolic'});
    const group = new Adw.PreferencesGroup({title: 'Simulation only',
        description: 'No real settings, microphone, downloads or workers. Use this page to exercise service states.'});
    page.add(group);
    const row = new Adw.ComboRow({title: 'Service State', use_subtitle: true,
        model: Gtk.StringList.new(SCENARIOS), selected: SCENARIOS.indexOf(preview.scenario)});
    let syncing = false;
    row.connect('notify::selected', () => {
        if (!syncing)
            preview.setScenario(SCENARIOS[row.selected]);
    });
    preview.onScenarioChanged(name => {
        syncing = true;
        try { row.selected = SCENARIOS.indexOf(name); } finally { syncing = false; }
    });
    group.add(row);
    const theme = new Adw.ComboRow({title: 'Theme', use_subtitle: true,
        model: Gtk.StringList.new(['System', 'Light', 'Dark'])});
    theme.connect('notify::selected', () => {
        Adw.StyleManager.get_default().color_scheme = [Adw.ColorScheme.DEFAULT,
            Adw.ColorScheme.FORCE_LIGHT, Adw.ColorScheme.FORCE_DARK][theme.selected];
    });
    group.add(theme);
    const font = new Adw.SwitchRow({title: 'Larger Text'});
    const originalFont = Gtk.Settings.get_default().gtk_font_name;
    font.connect('notify::active', () => { Gtk.Settings.get_default().gtk_font_name = font.active ? 'Sans 14' : originalFont; });
    group.add(font);
    preview.window.add(page);
}
