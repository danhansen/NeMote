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

import {requiredModifiersHeld} from './shortcutState.js';
import {ServiceLifecycle} from './serviceLifecycle.js';

const BUS_NAME = 'dev.wordpipe.Service';
const OBJECT_PATH = '/dev/wordpipe/Service';

const SERVICE_XML = `
<node>
  <interface name="dev.wordpipe.Service1">
    <method name="Start"/>
    <method name="Stop"/>
    <method name="Toggle"/>
    <method name="Shutdown"/>
    <method name="RegisterShellClient"/>
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

        this._profileItems = new Map();
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
    }

    setProfiles(profiles, selectedProfile) {
        const currentIds = [...this._profileItems.keys()];
        const nextIds = profiles.map(profile => profile.id);
        const inventoryChanged = currentIds.length !== nextIds.length ||
            currentIds.some((id, index) => id !== nextIds[index]);
        this._profiles = profiles;
        this._selectedProfile = selectedProfile;

        this._profileStatusItem.label.text = _('Model');

        if (inventoryChanged) {
            for (const row of this._profileItems.values())
                row.item.destroy();
            this._profileItems.clear();
            for (const profile of profiles) {
                const row = this._createProfileRow(profile);
                this._profileItems.set(profile.id, row);
                this._profileSection.addMenuItem(row.item);
            }
        }

        for (const profile of profiles)
            this._updateProfileRow(this._profileItems.get(profile.id), profile);
    }

    setSelectedProfile(selectedProfile) {
        this._selectedProfile = selectedProfile;
        for (const [profileId, row] of this._profileItems.entries()) {
            row.item.setOrnament(profileId === selectedProfile
                ? PopupMenu.Ornament.CHECK
                : PopupMenu.Ornament.NONE);
        }
    }

    _createProfileRow(profile) {
        const item = new PopupMenu.PopupBaseMenuItem();
        const titleLabel = new St.Label({
            x_expand: true,
            y_align: Clutter.ActorAlign.CENTER,
        });
        const installButton = createInstallButton(
            () => this._extension.installModel(profile.id));
        const progress = createInstallProgress();
        item.add_child(titleLabel);
        item.add_child(installButton);
        item.add_child(progress.box);
        item.connect('activate', () => {
            const current = this._profiles.find(candidate => candidate.id === profile.id);
            if (current?.installed)
                this._extension.selectModelProfile(profile.id);
        });
        return {item, titleLabel, installButton, progress};
    }

    _updateProfileRow(row, profile) {
        if (!row)
            return;
        const installing = this._installingProfile === profile.id;
        const progress = this._installProgressByProfile.get(profile.id) ?? {};
        const fraction = numberValue(progress.fraction);
        const installEnabled = !this._installing;

        row.titleLabel.text = profile.title;
        row.titleLabel.style_class = profile.installed
            ? ''
            : 'wordpipe-model-title-missing';
        row.item.reactive = profile.installed;
        row.item.can_focus = profile.installed;
        row.installButton.visible = !profile.installed && !installing;
        row.installButton.reactive = installEnabled;
        row.installButton.can_focus = installEnabled;
        row.installButton.style_class = installEnabled
            ? 'wordpipe-model-download-button'
            : 'wordpipe-model-download-button wordpipe-model-download-disabled';
        row.progress.box.visible = !profile.installed && installing;
        row.progress.fill.style =
            `width: ${Math.round(Math.max(0.0, Math.min(1.0, fraction ?? 0.0)) * 64)}px;`;
        row.progress.label.text = installProgressLabel(fraction);
        row.item.setOrnament(profile.id === this._selectedProfile
            ? PopupMenu.Ornament.CHECK
            : PopupMenu.Ornament.NONE);
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
        this._inputMethod = null;
        this._pendingDeltaIds = new Set();
        this._active = false;
        this._hasPreedit = false;
    }

    reset(sessionId) {
        this.cancel();
        this._lastSession = Number(sessionId);
        this._lastSeq = 0;
        this._active = true;
        this._focusWindow = global.display.focus_window;
        const method = this._getInputMethod();
        this._tracksFocus = method && '_currentFocus' in method;
        this._focus = method?._currentFocus;
    }

    cancel() {
        this._clearPendingDeltas();
        this._clearPreedit();
        this._active = false;
    }

    _targetMatches() {
        const method = this._getInputMethod();
        if (!this._active || !method)
            return false;
        if (global.display.focus_window !== this._focusWindow ||
            (this._tracksFocus && method._currentFocus !== this._focus) ||
            (method._preeditVisible && method._preeditStr)) {
            this.cancel();
            return false;
        }
        return true;
    }

    _clearPreedit() {
        const method = this._inputMethod;
        if (this._hasPreedit && method &&
            (!this._tracksFocus || method._currentFocus === this._focus))
            method.set_preedit_text(null, 0, 0, Clutter.PreeditResetMode.CLEAR);
        this._hasPreedit = false;
    }

    insertPartial(sessionId, seq, text, delayMs = 0) {
        if (Number(sessionId) !== this._lastSession || Number(seq) <= this._lastSeq || !this._targetMatches())
            return;
        this._lastSeq = Number(seq);
        this._clearPendingDeltas();
        const render = () => {
            if (!this._targetMatches())
                return;
            const method = this._getInputMethod();
            // Unsupported clients get committed-only insertion, never hard
            // commits of provisional text that ITN might subsequently rewrite.
            if (!method.set_preedit_text || method.can_show_preedit === false)
                return;
            const cursor = Array.from(text).length;
            method.set_preedit_text(text || null, cursor, cursor, Clutter.PreeditResetMode.CLEAR);
            this._hasPreedit = Boolean(text);
        };
        if (delayMs > 0) {
            const sourceId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, delayMs, () => {
                this._pendingDeltaIds.delete(sourceId);
                render();
                return GLib.SOURCE_REMOVE;
            });
            this._pendingDeltaIds.add(sourceId);
        } else {
            render();
        }
    }

    insertCommit(sessionId, seq, text) {
        const numericSession = Number(sessionId);
        const numericSeq = Number(seq);
        if (numericSession !== this._lastSession || numericSeq <= this._lastSeq || !this._targetMatches())
            return;
        this._clearPendingDeltas();
        this._lastSeq = numericSeq;
        this._clearPreedit();
        const inputMethod = this._getInputMethod();
        if (text)
            inputMethod.commit(text);
        this._active = false;
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
        this._enabled = true;
        this._lifecycle = Symbol('enabled');
        this._proxyReady = false;
        this._nameOwnerSignalId = 0;
        this._proxyCancellable = new Gio.Cancellable();
        this._serviceLifecycle ??= new ServiceLifecycle({
            start: () => this._startService(),
            stop: owner => this._stopService(owner),
            ready: owner => {
                if (this._enabled && this._proxyReady && this._proxy?.g_name_owner === owner)
                    this._refreshConnectedService();
            },
            error: error => {
                if (this._enabled)
                    this._setAvailable(false);
                logError(error, 'Wordpipe service lifecycle failed');
            },
        });
        this._settings = this.getSettings();
        this._state = {};
        this._profiles = [];
        this._selectedProfile = '';
        this._signalIds = [];
        this._syncingSettings = false;
        this._shortcutBound = false;
        this._pushToTalkActive = false;
        this._pushToTalkKeyCode = 0;
        this._pushToTalkModifierMask = 0;
        this._stageCapturedEventId = 0;
        this._injector = new TextInjector();
        this._insertionFocusId = global.display.connect('notify::focus-window', () => this._injector?.cancel());

        this._indicator = new Indicator(this);
        Main.panel.addToStatusArea(this.uuid, this._indicator);

        this._settings.set_boolean('shortcut-capture-active', false);
        this._syncShortcutBinding();
        this._connectSettings();
        this._connectProxy();
        this._serviceLifecycle.enable();
        this._stageCapturedEventId = global.stage.connect(
            'captured-event',
            (_actor, event) => this._handleCapturedEvent(event));
    }

    disable() {
        this._enabled = false;
        this._injector?.cancel();
        if (this._insertionFocusId)
            global.display.disconnect(this._insertionFocusId);
        this._insertionFocusId = 0;
        this._lifecycle = Symbol('disabled');
        this._proxyCancellable?.cancel();
        this._proxyCancellable = null;
        this._serviceLifecycle?.disable();
        this._settings?.disconnectObject(this);
        this._unbindShortcut();
        if (this._stageCapturedEventId) {
            global.stage.disconnect(this._stageCapturedEventId);
            this._stageCapturedEventId = 0;
        }

        if (this._proxy) {
            if (this._nameOwnerSignalId)
                this._proxy.disconnect(this._nameOwnerSignalId);
            this._nameOwnerSignalId = 0;
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
        this._indicator?.setSelectedProfile(profile);
        this._syncProfileMenu();
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
            (_display, _window, event, binding) =>
                this._handleShortcut(event, binding));
        this._shortcutBound = true;
    }

    _unbindShortcut() {
        this._stopPushToTalk();
        if (!this._shortcutBound)
            return;
        Main.wm.removeKeybinding('toggle-shortcut');
        this._shortcutBound = false;
    }

    _handleShortcut(event, binding) {
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
        this._pushToTalkKeyCode = event?.get_key_code() ?? 0;
        // get_mask() is the resolved, physical modifier mask used by the
        // keyboard state events. get_modifiers() returns virtual masks such
        // as SUPER_MASK, which cannot be compared to a physical MOD4 state.
        this._pushToTalkModifierMask = binding?.get_mask() ?? 0;
        this._callRemote('Start');
    }

    _handleCapturedEvent(event) {
        if (!this._pushToTalkActive)
            return Clutter.EVENT_PROPAGATE;

        if (event.type() === Clutter.EventType.KEY_RELEASE) {
            if (event.get_key_code() === this._pushToTalkKeyCode)
                this._stopPushToTalk();
        } else if (event.type() === Clutter.EventType.KEY_STATE) {
            // Mutter emits a complete modifier-state event even when it
            // consumes the original release (notably Super). Checking the
            // resulting state makes release order irrelevant.
            const [pressed, latched, locked] = event.get_key_state();
            if (!requiredModifiersHeld(
                this._pushToTalkModifierMask, pressed, latched, locked))
                this._stopPushToTalk();
        }

        return Clutter.EVENT_PROPAGATE;
    }

    _stopPushToTalk() {
        if (!this._pushToTalkActive)
            return;
        this._pushToTalkActive = false;
        this._pushToTalkKeyCode = 0;
        this._pushToTalkModifierMask = 0;
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
        const lifecycle = this._lifecycle;
        this._proxy = new WordpipeProxy(
            Gio.DBus.session,
            BUS_NAME,
            OBJECT_PATH,
            (proxy, error) => {
                if (!this._enabled || lifecycle !== this._lifecycle)
                    return;
                if (error) {
                    this._setAvailable(false);
                    logError(error, 'Wordpipe could not connect to service');
                    return;
                }
                this._proxyReady = true;
                this._subscribeSignals();
                this._nameOwnerSignalId = proxy.connect('notify::g-name-owner', () => {
                    if (!this._enabled || lifecycle !== this._lifecycle)
                        return;
                    const owner = proxy.g_name_owner;
                    this._serviceLifecycle.observeOwner(owner);
                    this._setAvailable(Boolean(owner));
                    if (owner && this._serviceLifecycle.owner === owner)
                        this._refreshConnectedService();
                });
                this._serviceLifecycle.observeOwner(proxy.g_name_owner);
                if (proxy.g_name_owner && this._serviceLifecycle.owner === proxy.g_name_owner)
                    this._refreshConnectedService();
            }, this._proxyCancellable, Gio.DBusProxyFlags.DO_NOT_AUTO_START);
    }

    _refreshConnectedService() {
        this._setAvailable(true);
        this._refreshState();
        this._refreshProfiles();
        this._refreshConfigFromService();
    }

    _busCall(destination, path, iface, method, parameters, flags = Gio.DBusCallFlags.NONE) {
        return new Promise((resolve, reject) => {
            Gio.DBus.session.call(destination, path, iface, method, parameters, null,
                flags, 10000, null, (connection, result) => {
                    try {
                        resolve(connection.call_finish(result));
                    } catch (error) {
                        reject(error);
                    }
                });
        });
    }

    async _startService() {
        for (let attempt = 0; attempt < 2; attempt++) {
            await this._busCall('org.freedesktop.DBus', '/org/freedesktop/DBus',
                'org.freedesktop.DBus', 'StartServiceByName',
                new GLib.Variant('(su)', [BUS_NAME, 0]));
            let owner;
            try {
                const result = await this._busCall('org.freedesktop.DBus', '/org/freedesktop/DBus',
                    'org.freedesktop.DBus', 'GetNameOwner', new GLib.Variant('(s)', [BUS_NAME]));
                [owner] = result.deep_unpack();
            } catch (error) {
                if (attempt === 0 && /NameHasNoOwner|ServiceUnknown/.test(error.message))
                    continue;
                throw error;
            }
            try {
                await this._busCall(owner, OBJECT_PATH, 'dev.wordpipe.Service1',
                    'RegisterShellClient', null, Gio.DBusCallFlags.NO_AUTO_START);
            } catch (error) {
                if (/UnknownMethod|Unknown method/.test(error.message))
                    return owner; // Explicit Shutdown remains compatible with old services.
                if (attempt === 0 && /UnknownObject|NameHasNoOwner|ServiceUnknown|shutting down/.test(error.message)) {
                    const pending = this._watchServiceExit(owner);
                    try {
                        await pending.promise;
                    } finally {
                        pending.cleanup();
                    }
                    continue;
                }
                throw error;
            }
            return owner;
        }
        throw new Error('Wordpipe service failed to activate');
    }

    _watchServiceExit(owner) {
        let watchId = 0;
        let timeoutId = 0;
        let finish;
        const promise = new Promise((resolve, reject) => {
            finish = resolve;
            watchId = Gio.bus_watch_name_on_connection(Gio.DBus.session, BUS_NAME,
                Gio.BusNameWatcherFlags.NONE,
                (_connection, _name, currentOwner) => {
                    if (currentOwner !== owner)
                        resolve();
                },
                () => resolve());
            timeoutId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 10000, () => {
                timeoutId = 0;
                reject(new Error('Wordpipe service did not exit after shutdown'));
                return GLib.SOURCE_REMOVE;
            });
        });
        promise.catch(() => {});
        return {
            promise,
            finish,
            cleanup: () => {
                if (watchId)
                    Gio.bus_unwatch_name(watchId);
                if (timeoutId)
                    GLib.Source.remove(timeoutId);
            },
        };
    }

    async _stopService(owner) {
        // Unique-name addressing and NO_AUTO_START prevent teardown from
        // activating a daemon after the extension has been disabled.
        const pending = this._watchServiceExit(owner);
        try {
            try {
                await this._busCall(owner, OBJECT_PATH, 'dev.wordpipe.Service1',
                    'Shutdown', null, Gio.DBusCallFlags.NO_AUTO_START);
            } catch (error) {
                if (/NameHasNoOwner|ServiceUnknown/.test(error.message))
                    pending.finish();
                else if (!/UnknownObject|UnknownMethod|shutting down/.test(error.message))
                    throw error;
            }
            await pending.promise;
        } finally {
            pending.cleanup();
        }
    }

    _connectServiceSignal(name, callback) {
        const proxy = this._proxy;
        const lifecycle = this._lifecycle;
        return proxy.connectSignal(name, (senderProxy, sender, args) => {
            if (!this._enabled || lifecycle !== this._lifecycle ||
                proxy !== this._proxy || sender !== proxy.g_name_owner)
                return;
            callback(senderProxy, sender, args);
        });
    }

    _subscribeSignals() {
        this._signalIds.push(this._connectServiceSignal('StateChanged',
            (_proxy, _sender, [state]) => this._handleState(deepUnpackMap(state))));
        this._signalIds.push(this._connectServiceSignal('ConfigChanged',
            (_proxy, _sender, [config]) => {
                this._syncSettingsFromConfig(deepUnpackMap(config));
                this._syncProfileMenu();
                this._refreshProfiles();
                this._refreshState();
            }));
        this._signalIds.push(this._connectServiceSignal('SessionStarted',
            (_proxy, _sender, [sessionId]) => {
                this._injector.reset(sessionId);
            }));
        this._signalIds.push(this._connectServiceSignal('Partial',
            (_proxy, _sender, [sessionId, seq, text]) => {
                if (this._settings.get_boolean('insert-partials')) {
                    this._injector.insertPartial(
                        sessionId,
                        seq,
                        text,
                        this._settings.get_uint('stream-insert-delay-ms'));
                }
            }));
        this._signalIds.push(this._connectServiceSignal('Commit',
            (_proxy, _sender, [sessionId, seq, text]) => {
                this._injector.insertCommit(sessionId, seq, text);
            }));
        this._signalIds.push(this._connectServiceSignal('SessionStopped',
            (_proxy, _sender, [sessionId]) => {
                if (Number(sessionId) === this._injector._lastSession)
                    this._injector.cancel();
            }));
        this._signalIds.push(this._connectServiceSignal('InstallProgress',
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
        this._signalIds.push(this._connectServiceSignal('Metrics',
            (_proxy, _sender, [metrics]) => {
                const values = deepUnpackMap(metrics);
                const summary = formatMetrics(values);
                if (summary)
                    this._indicator?.setMetrics(summary);
                this._indicator?.setVoiceLevel(numberValue(values.last_rms) ?? 0.0);
            }));
        this._signalIds.push(this._connectServiceSignal('Error',
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
                    model_profile: values.model_profile ?? values.id ?? '',
                    model_family: values.model_family ?? 'multilingual',
                    installed: Boolean(values.installed),
                };
            }).filter(profile => profile.id);
            this._syncProfileMenu(true);
        });
    }

    _syncSettingsFromConfig(config) {
        if (typeof config.model_preset === 'string')
            this._selectedProfile = config.model_preset;
        else if (typeof config.model_profile === 'string')
            this._selectedProfile = config.model_family === 'english'
                ? `${config.model_profile}-english`
                : config.model_profile;
        this._syncingSettings = true;
        try {
            if (typeof config.backend === 'string')
                this._settings.set_string('backend', config.backend);
            if (typeof config.model_profile === 'string')
                this._settings.set_string('model-profile', config.model_profile);
            if (typeof config.model_family === 'string')
                this._settings.set_string('model-family', config.model_family);
            if (typeof config.streaming_latency_ms === 'number')
                this._settings.set_uint('streaming-latency-ms', config.streaming_latency_ms);
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
            if (typeof config.itn === 'boolean')
                this._settings.set_boolean('itn', config.itn);
            for (const [key, setting] of [['phrase_boosting', 'phrase-boosting'], ['vad_filtering', 'vad-filtering']]) {
                if (typeof config[key] === 'boolean')
                    this._settings.set_boolean(setting, config[key]);
            }
            for (const [key, setting] of [['boost_phrases', 'boost-phrases'], ['boost_tokenizer_path', 'boost-tokenizer-path'], ['vad_model_path', 'vad-model-path']]) {
                if (typeof config[key] === 'string')
                    this._settings.set_string(setting, config[key]);
            }
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
        case 'model-family':
            // Preset changes are sent atomically by the menu/preferences client.
            // These keys mirror service state and must not be pushed separately.
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
        case 'itn':
        case 'phrase-boosting':
        case 'boost-phrases':
        case 'boost-tokenizer-path':
        case 'vad-filtering':
        case 'vad-model-path':
        case 'language':
        case 'worker-path':
        case 'model-installer-path':
        case 'num-threads':
        case 'sample-rate':
        case 'streaming-latency-ms':
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
            streaming_latency_ms: new GLib.Variant('u', this._settings.get_uint('streaming-latency-ms')),
            itn: new GLib.Variant('b', this._settings.get_boolean('itn')),
            phrase_boosting: new GLib.Variant('b', this._settings.get_boolean('phrase-boosting')),
            boost_phrases: new GLib.Variant('s', this._settings.get_string('boost-phrases')),
            boost_tokenizer_path: new GLib.Variant('s', this._settings.get_string('boost-tokenizer-path')),
            vad_filtering: new GLib.Variant('b', this._settings.get_boolean('vad-filtering')),
            vad_model_path: new GLib.Variant('s', this._settings.get_string('vad-model-path')),
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

    _syncProfileMenu(rebuild = false) {
        if (!this._settings)
            return;
        const profile = this._settings.get_string('model-profile');
        const family = this._settings.get_string('model-family');
        const preset = family === 'english' ? `${profile}-english` : profile;
        const selected = this._profiles.some(item => item.id === this._selectedProfile)
            ? this._selectedProfile
            : preset;
        this._selectedProfile = selected;
        if (rebuild)
            this._indicator?.setProfiles(this._profiles, selected);
        else
            this._indicator?.setSelectedProfile(selected);
    }

    _callRemote(method, ...args) {
        if (!this._enabled)
            return;
        const proxy = this._proxy;
        const lifecycle = this._lifecycle;
        const owner = proxy?.g_name_owner;
        if (!owner) {
            this._serviceLifecycle?.ensureStarted();
            return;
        }
        const callback = typeof args.at(-1) === 'function' ? args.pop() : null;
        const remote = this._proxy?.[`${method}Remote`];
        if (!remote) {
            this._setAvailable(false);
            return;
        }
        remote.call(this._proxy, ...args, (result, error) => {
            if (!this._enabled || lifecycle !== this._lifecycle ||
                proxy !== this._proxy || owner !== proxy.g_name_owner)
                return;
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
        if (typeof state.model_preset === 'string') {
            this._selectedProfile = state.model_preset;
            this._syncProfileMenu();
        } else if (typeof state.model_profile === 'string') {
            this._selectedProfile = state.model_family === 'english'
                ? `${state.model_profile}-english`
                : state.model_profile;
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
        if (!available)
            this._injector?.cancel();
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

function createInstallButton(onClicked) {
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
        style_class: 'wordpipe-model-download-button',
        child: content,
        reactive: true,
        can_focus: true,
        y_align: Clutter.ActorAlign.CENTER,
    });
    button.connect('clicked', onClicked);
    return button;
}

function createInstallProgress() {
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
        style: 'width: 0px;',
        x_align: Clutter.ActorAlign.START,
    });
    track.add_child(fill);
    box.add_child(track);
    const label = new St.Label({
        text: installProgressLabel(null),
        style_class: 'wordpipe-model-progress-label',
        y_align: Clutter.ActorAlign.CENTER,
    });
    box.add_child(label);
    return {box, fill, label};
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
