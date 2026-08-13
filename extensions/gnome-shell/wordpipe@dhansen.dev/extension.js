import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';
import Meta from 'gi://Meta';
import Shell from 'gi://Shell';
import St from 'gi://St';

import {Extension, gettext as _} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';

const BUS_NAME = 'dev.wordpipe.Service';
const OBJECT_PATH = '/dev/wordpipe/Service';

const SERVICE_XML = `
<node>
  <interface name="dev.wordpipe.Service1">
    <method name="Start"/>
    <method name="Stop"/>
    <method name="Toggle"/>
    <method name="Shutdown"/>
    <method name="GetState"><arg name="state" type="a{sv}" direction="out"/></method>
    <method name="GetConfig"><arg name="config" type="a{sv}" direction="out"/></method>
    <method name="ListBackends"><arg name="backends" type="aa{sv}" direction="out"/></method>
    <method name="ListModelProfiles"><arg name="profiles" type="aa{sv}" direction="out"/></method>
    <method name="ListInputDevices"><arg name="devices" type="aa{sv}" direction="out"/></method>
    <method name="SetBackend"><arg name="backend" type="s" direction="in"/></method>
    <method name="SetModelProfile"><arg name="profile" type="s" direction="in"/></method>
    <method name="SetInputDevice"><arg name="selector" type="s" direction="in"/></method>
    <method name="SetShortcut"><arg name="accelerator" type="s" direction="in"/></method>
    <method name="SetInsertionOptions"><arg name="options" type="a{sv}" direction="in"/></method>
    <method name="SetRuntimeOptions"><arg name="options" type="a{sv}" direction="in"/></method>
    <method name="InstallModel"><arg name="profile" type="s" direction="in"/></method>
    <signal name="StateChanged"><arg name="state" type="a{sv}"/></signal>
    <signal name="ConfigChanged"><arg name="config" type="a{sv}"/></signal>
    <signal name="SessionStarted"><arg name="session_id" type="t"/></signal>
    <signal name="TextDelta"><arg name="session_id" type="t"/><arg name="seq" type="t"/><arg name="text" type="s"/></signal>
    <signal name="Partial"><arg name="session_id" type="t"/><arg name="seq" type="t"/><arg name="full_text" type="s"/></signal>
    <signal name="Commit"><arg name="session_id" type="t"/><arg name="seq" type="t"/><arg name="text" type="s"/></signal>
    <signal name="SessionStopped"><arg name="session_id" type="t"/></signal>
    <signal name="InstallProgress"><arg name="profile" type="s"/><arg name="progress" type="a{sv}"/></signal>
    <signal name="Metrics"><arg name="metrics" type="a{sv}"/></signal>
    <signal name="Error"><arg name="message" type="s"/></signal>
  </interface>
</node>`;

const WordpipeProxy = Gio.DBusProxy.makeProxyWrapper(SERVICE_XML);

const Indicator = GObject.registerClass(
class Indicator extends PanelMenu.Button {
    constructor(extension) {
        super(0.0, _('Wordpipe'));
        this._extension = extension;
        this._listening = false;
        this._lastVoiceLevel = 0.0;
        this._animationPhase = 0;
        this._animationSourceId = 0;

        this._box = new St.BoxLayout({
            style_class: 'wordpipe-panel-status',
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._icon = new St.Icon({
            icon_name: 'audio-input-microphone-symbolic',
            style_class: 'system-status-icon wordpipe-panel-icon',
        });
        this._icon.set_pivot_point(0.5, 0.5);
        this._box.add_child(this._icon);

        this._levelBars = [];
        this._levelBox = new St.BoxLayout({
            style_class: 'wordpipe-level-bars',
            y_align: Clutter.ActorAlign.CENTER,
            opacity: 0,
        });
        for (let i = 0; i < 4; i++) {
            const bar = new St.Widget({
                style_class: 'wordpipe-level-bar',
                width: 2,
                height: 10,
                y_align: Clutter.ActorAlign.CENTER,
            });
            bar.set_pivot_point(0.5, 1.0);
            bar.scale_y = 0.2;
            this._levelBox.add_child(bar);
            this._levelBars.push(bar);
        }
        this._box.add_child(this._levelBox);
        this.add_child(this._box);

        this._toggleItem = new PopupMenu.PopupMenuItem(_('Start Dictation'));
        this._toggleItem.connect('activate', () => this._extension.toggleDictation());
        this.menu.addMenuItem(this._toggleItem);

        this._statusItem = new PopupMenu.PopupMenuItem(_('Service unavailable'), {
            reactive: false,
        });
        this.menu.addMenuItem(this._statusItem);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        this._profileStatusItem = new PopupMenu.PopupMenuItem(_('Model'), {
            reactive: false,
        });
        this.menu.addMenuItem(this._profileStatusItem);

        this._profileItems = [];
        this._installProfileItems = new Map();
        this._installProgressByProfile = new Map();
        this._installing = false;
        this._installingProfile = '';
        this._profiles = [];
        this._selectedProfile = '';
        this._profileSection = new PopupMenu.PopupMenuSection();
        this.menu.addMenuItem(this._profileSection);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        const prefsItem = new PopupMenu.PopupMenuItem(_('Preferences'));
        prefsItem.connect('activate', () => this._extension.openPreferences());
        this.menu.addMenuItem(prefsItem);
    }

    setState(state, available) {
        const previousInstalling = this._installing;
        const previousInstallingProfile = this._installingProfile;
        const listening = available && Boolean(state?.listening);
        const stopping = Boolean(state?.stopping);
        const installing = Boolean(state?.installing);
        const loading = Boolean(state?.loading_model);
        const selectedModelInstalled = state?.selected_model_installed !== false;
        this._installing = installing;
        this._installingProfile = state?.installing_profile ?? '';
        this._icon.icon_name = 'audio-input-microphone-symbolic';
        this._setListening(listening);
        this._toggleItem.label.text = listening || stopping
            ? _('Stop Dictation')
            : _('Start Dictation');
        const canToggleDictation = listening || stopping ||
            (!loading && selectedModelInstalled);
        this._toggleItem.setSensitive(available && canToggleDictation);
        this._statusItem.label.text = available
            ? statusText(state, selectedModelInstalled)
            : _('Service unavailable');
        if (
            previousInstalling !== this._installing ||
            previousInstallingProfile !== this._installingProfile
        )
            this.setProfiles(this._profiles, this._selectedProfile);
        else
            this._syncInstallActions();
    }

    setProfiles(profiles, selectedProfile) {
        this._profiles = profiles;
        this._selectedProfile = selectedProfile;
        for (const item of this._profileItems)
            item.destroy();
        this._profileItems = [];
        this._installProfileItems.clear();

        this._profileStatusItem.label.text = _('Model');

        if (!profiles.length)
            return;

        for (const profile of profiles) {
            const isSelected = profile.id === selectedProfile;
            const installing = this._installingProfile === profile.id;
            const rowReactive = profile.installed && !isSelected;
            const selectItem = new PopupMenu.PopupBaseMenuItem({
                reactive: rowReactive,
                can_focus: rowReactive,
            });
            const titleLabel = new St.Label({
                text: profile.title,
                x_expand: true,
                y_align: Clutter.ActorAlign.CENTER,
            });
            if (!profile.installed)
                titleLabel.style_class = 'wordpipe-model-title-missing';
            selectItem.add_child(titleLabel);
            if (!profile.installed) {
                const progress = this._installProgressByProfile.get(profile.id) ?? {};
                const fraction = numberValue(progress.fraction);
                selectItem.add_child(installing
                    ? createInstallProgress(fraction)
                    : createInstallButton(this._installing,
                        () => this._extension.installModel(profile.id)));
            }
            selectItem.setOrnament(isSelected
                ? PopupMenu.Ornament.CHECK
                : PopupMenu.Ornament.NONE);
            if (profile.installed && !isSelected) {
                selectItem.connect('activate',
                    () => this._extension.selectModelProfile(profile.id));
            }
            if (!profile.installed)
                this._installProfileItems.set(profile.id, selectItem);
            this._profileSection.addMenuItem(selectItem);
            this._profileItems.push(selectItem);
        }
    }

    setMetrics(summary) {
        if (summary)
            this._statusItem.label.text = summary;
    }

    setStatusMessage(message) {
        if (message)
            this._statusItem.label.text = message;
    }

    setInstallProgress(profile, progress) {
        const active = progress.phase !== 'complete' && progress.phase !== 'error';
        if (active)
            this._installProgressByProfile.set(profile, progress);
        else
            this._installProgressByProfile.delete(profile);

        this._installing = active;
        this._installingProfile = active ? profile : '';
        this.setProfiles(this._profiles, this._selectedProfile);
    }

    setVoiceLevel(rms) {
        if (!this._listening)
            return;
        this._lastVoiceLevel = normalizeVoiceLevel(rms);
        this._renderVoiceLevel();
    }

    destroy() {
        this._stopVoiceAnimation();
        super.destroy();
    }

    _syncInstallActions() {
        if (this._installProfileItems.size)
            this.setProfiles(this._profiles, this._selectedProfile);
    }

    _setListening(listening) {
        if (this._listening === listening)
            return;
        this._listening = listening;
        this._levelBox.ease({
            opacity: listening ? 255 : 0,
            duration: 120,
            mode: Clutter.AnimationMode.EASE_OUT_QUAD,
        });
        if (listening)
            this._startVoiceAnimation();
        else {
            this._stopVoiceAnimation();
            this._resetVoiceLevel();
        }
    }

    _startVoiceAnimation() {
        if (this._animationSourceId)
            return;
        this._animationSourceId = GLib.timeout_add(
            GLib.PRIORITY_DEFAULT,
            120,
            () => {
                if (!this._listening) {
                    this._animationSourceId = 0;
                    return GLib.SOURCE_REMOVE;
                }
                this._animationPhase += 1;
                this._renderVoiceLevel();
                return GLib.SOURCE_CONTINUE;
            });
    }

    _stopVoiceAnimation() {
        if (!this._animationSourceId)
            return;
        GLib.Source.remove(this._animationSourceId);
        this._animationSourceId = 0;
    }

    _renderVoiceLevel() {
        if (!this._listening)
            return;
        const level = Math.max(this._lastVoiceLevel, 0.18);
        const iconLevel = level * (0.75 + 0.25 * Math.sin(this._animationPhase * 0.8));
        this._icon.ease({
            scale_x: 1.0 + iconLevel * 0.08,
            scale_y: 1.0 + iconLevel * 0.16,
            duration: 100,
            mode: Clutter.AnimationMode.EASE_OUT_QUAD,
        });

        const multipliers = [0.55, 1.0, 0.75, 0.45];
        this._levelBars.forEach((bar, index) => {
            const wave = 0.5 + 0.5 * Math.sin(this._animationPhase * 0.9 + index * 1.35);
            const scale = 0.25 + level * multipliers[index] * (0.65 + wave);
            bar.ease({
                scale_y: Math.max(0.2, Math.min(1.6, scale)),
                duration: 100,
                mode: Clutter.AnimationMode.EASE_OUT_QUAD,
            });
        });
    }

    _resetVoiceLevel() {
        this._lastVoiceLevel = 0.0;
        this._animationPhase = 0;
        this._icon.ease({
            scale_x: 1.0,
            scale_y: 1.0,
            duration: 120,
            mode: Clutter.AnimationMode.EASE_OUT_QUAD,
        });
        for (const bar of this._levelBars)
            bar.scale_y = 0.2;
    }
});

class TextInjector {
    constructor() {
        this._lastSession = 0;
        this._lastSeq = 0;
        this._insertedText = '';
        this._inputMethod = null;
        this._pendingDeltaIds = new Set();
    }

    reset(sessionId) {
        this._clearPendingDeltas();
        this._lastSession = Number(sessionId);
        this._lastSeq = 0;
        this._insertedText = '';
    }

    insertDelta(sessionId, seq, text, delayMs = 0) {
        if (delayMs > 0) {
            const sourceId = GLib.timeout_add(
                GLib.PRIORITY_DEFAULT,
                delayMs,
                () => {
                    this._pendingDeltaIds.delete(sourceId);
                    this._insertDeltaNow(sessionId, seq, text);
                    return GLib.SOURCE_REMOVE;
                });
            this._pendingDeltaIds.add(sourceId);
            return;
        }
        this._insertDeltaNow(sessionId, seq, text);
    }

    _insertDeltaNow(sessionId, seq, text) {
        const numericSession = Number(sessionId);
        const numericSeq = Number(seq);
        if (numericSession !== this._lastSession)
            this.reset(numericSession);
        if (numericSeq <= this._lastSeq || !text)
            return;
        this._lastSeq = numericSeq;

        const inputMethod = this._getInputMethod();
        if (!inputMethod) {
            log(`Wordpipe text delta without input method: ${text}`);
            return;
        }
        inputMethod.commit(text);
        this._insertedText += text;
    }

    insertCommit(sessionId, seq, text) {
        const numericSession = Number(sessionId);
        const numericSeq = Number(seq);
        if (numericSession !== this._lastSession)
            this.reset(numericSession);
        else
            this._clearPendingDeltas();
        if (numericSeq <= this._lastSeq || !text)
            return;
        this._lastSeq = numericSeq;

        let textToInsert = text;
        if (this._insertedText) {
            if (text === this._insertedText || this._insertedText.startsWith(text))
                return;
            if (text.startsWith(this._insertedText))
                textToInsert = text.slice(this._insertedText.length);
            else {
                log(`Wordpipe commit differs from streamed text; keeping streamed text: ${text}`);
                return;
            }
        }

        const inputMethod = this._getInputMethod();
        if (!inputMethod) {
            log(`Wordpipe commit without input method: ${textToInsert}`);
            return;
        }
        inputMethod.commit(textToInsert);
        this._insertedText += textToInsert;
    }

    _getInputMethod() {
        if (this._inputMethod)
            return this._inputMethod;

        const backend = Clutter.get_default_backend?.();
        this._inputMethod = backend?.get_input_method?.() ?? null;
        return this._inputMethod;
    }

    _clearPendingDeltas() {
        for (const sourceId of this._pendingDeltaIds)
            GLib.Source.remove(sourceId);
        this._pendingDeltaIds.clear();
    }
}

export default class WordpipeExtension extends Extension {
    enable() {
        this._settings = this.getSettings();
        this._state = {};
        this._profiles = [];
        this._selectedProfile = '';
        this._signalIds = [];
        this._syncingSettings = false;
        this._shortcutBound = false;
        this._pushToTalkActive = false;
        this._injector = new TextInjector();

        this._indicator = new Indicator(this);
        Main.panel.addToStatusArea(this.uuid, this._indicator);

        this._settings.set_boolean('shortcut-capture-active', false);
        this._syncShortcutBinding();
        this._connectSettings();
        this._connectProxy();
    }

    disable() {
        this._settings?.disconnectObject(this);
        this._unbindShortcut();

        if (this._proxy) {
            for (const id of this._signalIds)
                this._proxy.disconnectSignal(id);
            this._signalIds = [];
            this._proxy = null;
        }

        this._indicator?.destroy();
        this._indicator = null;
        this._injector = null;
        this._settings = null;
    }

    toggleDictation() {
        this._callRemote('Toggle');
    }

    installModel(profile) {
        this._callRemote('InstallModel', profile);
    }

    selectModelProfile(profile) {
        this._selectedProfile = profile;
        this._syncProfileMenu();
        this._syncingSettings = true;
        try {
            this._settings.set_string('model-profile', profile);
        } finally {
            this._syncingSettings = false;
        }
        this._callRemote('SetModelProfile', profile, () => this._refreshConfigFromService());
    }

    _bindShortcut() {
        if (this._shortcutBound)
            return;
        const pushToTalk = this._settings.get_string('shortcut-mode') === 'push-to-talk';
        const flags = Meta.KeyBindingFlags.IGNORE_AUTOREPEAT |
            (pushToTalk ? Meta.KeyBindingFlags.TRIGGER_RELEASE : 0);
        Main.wm.addKeybinding(
            'toggle-shortcut',
            this._settings,
            flags,
            Shell.ActionMode.NORMAL | Shell.ActionMode.OVERVIEW,
            (_display, _window, event) => this._handleShortcut(event));
        this._shortcutBound = true;
    }

    _unbindShortcut() {
        this._stopPushToTalk();
        if (!this._shortcutBound)
            return;
        Main.wm.removeKeybinding('toggle-shortcut');
        this._shortcutBound = false;
    }

    _handleShortcut(event) {
        if (this._settings.get_string('shortcut-mode') !== 'push-to-talk') {
            this.toggleDictation();
            return;
        }

        if (event?.type() === Clutter.EventType.KEY_RELEASE) {
            this._stopPushToTalk();
            return;
        }

        if (this._pushToTalkActive)
            return;
        this._pushToTalkActive = true;
        this._callRemote('Start');
    }

    _stopPushToTalk() {
        if (!this._pushToTalkActive)
            return;
        this._pushToTalkActive = false;
        this._callRemote('Stop');
    }

    _syncShortcutBinding() {
        if (this._settings.get_boolean('shortcut-capture-active'))
            this._unbindShortcut();
        else
            this._bindShortcut();
    }

    _connectSettings() {
        this._settings.connectObject('changed::shortcut-capture-active', () => {
            this._syncShortcutBinding();
        }, this);
        this._settings.connectObject('changed::shortcut-mode', () => {
            this._unbindShortcut();
            this._syncShortcutBinding();
        }, this);
        this._settings.connectObject('changed', (_settings, key) => {
            if (key === 'shortcut-capture-active')
                return;
            if (this._settings.get_boolean('shortcut-capture-active'))
                return;
            if (!this._syncingSettings)
                this._pushSetting(key);
        }, this);
    }

    _connectProxy() {
        this._proxy = new WordpipeProxy(
            Gio.DBus.session,
            BUS_NAME,
            OBJECT_PATH,
            (proxy, error) => {
                if (error) {
                    this._setAvailable(false);
                    logError(error, 'Wordpipe could not connect to service');
                    return;
                }
                this._setAvailable(true);
                this._subscribeSignals();
                this._refreshState();
                this._refreshProfiles();
                this._refreshConfigFromService();
            });
    }

    _subscribeSignals() {
        this._signalIds.push(this._proxy.connectSignal('StateChanged',
            (_proxy, _sender, [state]) => this._handleState(deepUnpackMap(state))));
        this._signalIds.push(this._proxy.connectSignal('ConfigChanged',
            (_proxy, _sender, [config]) => {
                this._syncSettingsFromConfig(deepUnpackMap(config));
                this._syncProfileMenu();
                this._refreshState();
            }));
        this._signalIds.push(this._proxy.connectSignal('SessionStarted',
            (_proxy, _sender, [sessionId]) => {
                this._injector.reset(sessionId);
            }));
        this._signalIds.push(this._proxy.connectSignal('TextDelta',
            (_proxy, _sender, [sessionId, seq, text]) => {
                if (this._settings.get_boolean('insert-partials')) {
                    this._injector.insertDelta(
                        sessionId,
                        seq,
                        text,
                        this._settings.get_uint('stream-insert-delay-ms'));
                }
            }));
        this._signalIds.push(this._proxy.connectSignal('Commit',
            (_proxy, _sender, [sessionId, seq, text]) => {
                this._injector.insertCommit(sessionId, seq, text);
            }));
        this._signalIds.push(this._proxy.connectSignal('InstallProgress',
            (_proxy, _sender, [profile, progress]) => {
                const values = deepUnpackMap(progress);
                if (typeof values.profile !== 'string')
                    values.profile = profile;
                if (this._state) {
                    this._state.last_install_progress = values;
                    this._state.installing = values.phase !== 'complete' && values.phase !== 'error';
                    this._state.installing_profile = this._state.installing ? values.profile : '';
                }
                const summary = formatInstallProgress(values);
                if (summary)
                    this._indicator?.setStatusMessage(summary);
                if (typeof values.profile === 'string')
                    this._indicator?.setInstallProgress(values.profile, values);
                if (values.phase === 'complete' || values.phase === 'error')
                    this._refreshProfiles();
            }));
        this._signalIds.push(this._proxy.connectSignal('Metrics',
            (_proxy, _sender, [metrics]) => {
                const values = deepUnpackMap(metrics);
                const summary = formatMetrics(values);
                if (summary)
                    this._indicator?.setMetrics(summary);
                this._indicator?.setVoiceLevel(numberValue(values.last_rms) ?? 0.0);
            }));
        this._signalIds.push(this._proxy.connectSignal('Error',
            (_proxy, _sender, [message]) => {
                this._indicator?.setStatusMessage(message);
                log(`Wordpipe service error: ${message}`);
            }));
    }

    _refreshState() {
        this._callRemote('GetState', (state) => this._handleState(deepUnpackMap(state)));
    }

    _refreshConfigFromService(callback = null) {
        this._callRemote('GetConfig', config => {
            this._syncSettingsFromConfig(deepUnpackMap(config));
            this._syncProfileMenu();
            if (callback)
                callback();
        });
    }

    _refreshProfiles() {
        this._callRemote('ListModelProfiles', profiles => {
            this._profiles = profiles.map(profile => {
                const values = deepUnpackMap(profile);
                return {
                    id: values.id ?? '',
                    title: values.title ?? values.id ?? '',
                    installed: Boolean(values.installed),
                };
            }).filter(profile => profile.id);
            this._syncProfileMenu();
        });
    }

    _syncSettingsFromConfig(config) {
        if (typeof config.model_profile === 'string')
            this._selectedProfile = config.model_profile;
        this._syncingSettings = true;
        try {
            if (typeof config.backend === 'string')
                this._settings.set_string('backend', config.backend);
            if (typeof config.model_profile === 'string')
                this._settings.set_string('model-profile', config.model_profile);
            if (typeof config.input_device === 'string')
                this._settings.set_string('input-device', config.input_device);
            if (typeof config.language === 'string')
                this._settings.set_string('language', config.language);
            if (typeof config.model_root === 'string')
                this._settings.set_string('model-root', config.model_root);
            if (typeof config.worker_path === 'string')
                this._settings.set_string('worker-path', config.worker_path);
            if (typeof config.model_installer_path === 'string')
                this._settings.set_string('model-installer-path', config.model_installer_path);
            if (typeof config.shortcut === 'string')
                this._settings.set_strv('toggle-shortcut', config.shortcut ? [config.shortcut] : []);
            if (typeof config.num_threads === 'number')
                this._settings.set_uint('num-threads', config.num_threads);
            if (typeof config.sample_rate === 'number')
                this._settings.set_uint('sample-rate', config.sample_rate);
            if (typeof config.spoken_punctuation === 'boolean')
                this._settings.set_boolean('spoken-punctuation', config.spoken_punctuation);
            if (typeof config.insert_partials === 'boolean')
                this._settings.set_boolean('insert-partials', config.insert_partials);
            if (typeof config.stream_insert_delay_ms === 'number')
                this._settings.set_uint('stream-insert-delay-ms', config.stream_insert_delay_ms);
            if (typeof config.show_overlay === 'boolean')
                this._settings.set_boolean('show-overlay', false);
        } finally {
            this._syncingSettings = false;
        }
    }

    _pushSetting(key) {
        if (!this._proxy)
            return;

        switch (key) {
        case 'backend':
            this._callRemote('SetBackend', this._settings.get_string('backend'));
            break;
        case 'model-profile':
            this._callRemote('SetModelProfile', this._settings.get_string('model-profile'));
            break;
        case 'input-device':
            this._callRemote('SetInputDevice', this._settings.get_string('input-device'));
            break;
        case 'toggle-shortcut': {
            const shortcuts = this._settings.get_strv('toggle-shortcut');
            this._callRemote('SetShortcut', shortcuts.length > 0 ? shortcuts[0] : '');
            break;
        }
        case 'spoken-punctuation':
        case 'insert-partials':
        case 'stream-insert-delay-ms':
        case 'show-overlay':
            this._pushInsertionOptions();
            break;
        case 'model-root':
        case 'language':
        case 'worker-path':
        case 'model-installer-path':
        case 'num-threads':
        case 'sample-rate':
            this._pushRuntimeOptions();
            break;
        default:
            break;
        }
    }

    _pushInsertionOptions() {
        this._callRemote('SetInsertionOptions', {
            spoken_punctuation: new GLib.Variant('b',
                this._settings.get_boolean('spoken-punctuation')),
            insert_partials: new GLib.Variant('b',
                this._settings.get_boolean('insert-partials')),
            stream_insert_delay_ms: new GLib.Variant('u',
                this._settings.get_uint('stream-insert-delay-ms')),
            show_overlay: new GLib.Variant('b', false),
        });
    }

    _pushRuntimeOptions() {
        const language = this._settings.get_string('language');
        const modelRoot = this._settings.get_string('model-root');
        const workerPath = this._settings.get_string('worker-path');
        const modelInstallerPath = this._settings.get_string('model-installer-path');

        this._callRemote('SetRuntimeOptions', {
            model_root: new GLib.Variant('s', modelRoot),
            language: new GLib.Variant('s', language),
            worker_path: new GLib.Variant('s', workerPath),
            model_installer_path: new GLib.Variant('s', modelInstallerPath),
            num_threads: new GLib.Variant('u',
                this._settings.get_uint('num-threads')),
            sample_rate: new GLib.Variant('u',
                this._settings.get_uint('sample-rate')),
        });
    }

    _syncProfileMenu() {
        this._indicator?.setProfiles(
            this._profiles,
            this._selectedProfile || this._settings.get_string('model-profile'));
    }

    _callRemote(method, ...args) {
        const callback = typeof args.at(-1) === 'function' ? args.pop() : null;
        const remote = this._proxy?.[`${method}Remote`];
        if (!remote) {
            this._setAvailable(false);
            return;
        }
        remote.call(this._proxy, ...args, (result, error) => {
            if (error) {
                this._setAvailable(true);
                const message = formatError(error);
                this._indicator?.setStatusMessage(message);
                logError(error, `Wordpipe ${method} failed`);
                if (method === 'SetModelProfile')
                    this._refreshConfigFromService();
                return;
            }
            this._setAvailable(true);
            if (callback)
                callback(...result);
        });
    }

    _handleState(state) {
        this._state = state;
        if (typeof state.model_profile === 'string') {
            this._selectedProfile = state.model_profile;
            this._syncProfileMenu();
        }
        this._indicator?.setState(this._state, true);
        const installSummary = formatInstallProgress(state.last_install_progress ?? {});
        const metricsSummary = formatMetrics(state.last_metrics ?? {});
        if (state.installing && installSummary)
            this._indicator?.setMetrics(installSummary);
        else if (metricsSummary)
            this._indicator?.setMetrics(metricsSummary);
        this._indicator?.setVoiceLevel(numberValue(state.last_metrics?.last_rms) ?? 0.0);
    }

    _setAvailable(available) {
        this._indicator?.setState(this._state, available);
    }
}

function deepUnpackMap(value) {
    if (!value)
        return {};
    const unpacked = deepUnpackValue(value);
    if (!unpacked || typeof unpacked !== 'object' || Array.isArray(unpacked))
        return {};
    return unpacked;
}

function deepUnpackValue(value) {
    const unpacked = value?.deep_unpack ? value.deep_unpack() : value;
    if (!unpacked || typeof unpacked !== 'object' || Array.isArray(unpacked))
        return unpacked;
    const result = {};
    for (const [key, variant] of Object.entries(unpacked))
        result[key] = deepUnpackValue(variant);
    return result;
}

function statusText(state, selectedModelInstalled) {
    if (state?.installing)
        return _('Installing model');
    if (state?.loading_model)
        return _('Loading model');
    if (state?.listening)
        return _('Listening');
    if (state?.stopping)
        return _('Stopping');
    if (!selectedModelInstalled)
        return _('Model missing');
    return _('Ready');
}

function formatMetrics(metrics) {
    const rtf = numberValue(metrics.real_audio_real_time_factor ?? metrics.real_time_factor);
    const audioSeconds = numberValue(metrics.audio_seconds);
    const droppedChunks = numberValue(metrics.dropped_audio_chunks);
    if (rtf === null && audioSeconds === null && droppedChunks === null)
        return '';

    const parts = [];
    if (rtf !== null)
        parts.push(`RTF ${rtf.toFixed(3)}`);
    if (audioSeconds !== null)
        parts.push(`${audioSeconds.toFixed(1)}s`);
    if (droppedChunks)
        parts.push(`${droppedChunks} ${_('dropped')}`);
    return parts.join(' - ');
}

function formatInstallProgress(progress) {
    const profile = typeof progress.profile === 'string' ? progress.profile : '';
    const message = typeof progress.message === 'string'
        ? progress.message
        : typeof progress.phase === 'string'
            ? progress.phase
            : '';
    if (!profile && !message)
        return '';
    if (!profile)
        return markupSafe(message);
    if (!message)
        return markupSafe(profile);
    return markupSafe(`${profile}: ${message}`);
}

function installProgressLabel(fraction) {
    if (fraction === null)
        return _('Installing');
    return `${Math.round(Math.max(0.0, Math.min(1.0, fraction)) * 100)}%`;
}

function createInstallButton(disabled, onClicked) {
    const content = new St.BoxLayout({
        style_class: 'wordpipe-model-download-content',
        y_align: Clutter.ActorAlign.CENTER,
    });
    content.add_child(new St.Icon({
        icon_name: 'folder-download-symbolic',
        style_class: 'wordpipe-model-download-icon',
    }));
    content.add_child(new St.Label({
        text: _('Install'),
        y_align: Clutter.ActorAlign.CENTER,
    }));
    const button = new St.Button({
        style_class: disabled
            ? 'wordpipe-model-download-button wordpipe-model-download-disabled'
            : 'wordpipe-model-download-button',
        child: content,
        reactive: !disabled,
        can_focus: !disabled,
        y_align: Clutter.ActorAlign.CENTER,
    });
    if (!disabled)
        button.connect('clicked', onClicked);
    return button;
}

function createInstallProgress(fraction) {
    const progress = Math.max(0.0, Math.min(1.0, fraction ?? 0.0));
    const box = new St.BoxLayout({
        style_class: 'wordpipe-model-progress',
        y_align: Clutter.ActorAlign.CENTER,
    });
    const track = new St.Bin({
        style_class: 'wordpipe-model-progress-track',
        y_align: Clutter.ActorAlign.CENTER,
    });
    const fill = new St.Bin({
        style_class: 'wordpipe-model-progress-fill',
        style: `width: ${Math.round(progress * 64)}px;`,
        x_align: Clutter.ActorAlign.START,
    });
    track.add_child(fill);
    box.add_child(track);
    box.add_child(new St.Label({
        text: installProgressLabel(fraction),
        style_class: 'wordpipe-model-progress-label',
        y_align: Clutter.ActorAlign.CENTER,
    }));
    return box;
}

function numberValue(value) {
    if (typeof value === 'number' && Number.isFinite(value))
        return value;
    return null;
}

function normalizeVoiceLevel(rms) {
    const value = numberValue(rms) ?? 0.0;
    return Math.max(0.0, Math.min(1.0, (value - 0.004) * 18.0));
}

function formatError(error) {
    const message = error?.message ?? String(error);
    return markupSafe(message.replace(/^GDBus\.Error:[^:]+:\s*/, ''));
}

function markupSafe(value) {
    return GLib.markup_escape_text(String(value), -1);
}
