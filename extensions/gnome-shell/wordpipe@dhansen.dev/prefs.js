import Adw from 'gi://Adw';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';
import Gdk from 'gi://Gdk';
import Gtk from 'gi://Gtk';

import {ExtensionPreferences, gettext as _} from 'resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js';

import {installedModelProfiles, streamingChunkChoices} from './modelState.js';

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

const PROFILES = [
    ['compact', 'compact', 'Dynamic-int8 fixed-shape multilingual model.', 'compact', 'multilingual'],
    ['compact-english', 'compact (English only, lower WER)', 'Dynamic-int8 dedicated English model; selectable chunk size.', 'compact', 'english'],
    ['fast', 'fast', 'FP32 projected-cache multilingual model; largest footprint.', 'fast', 'multilingual'],
    ['fast-english', 'fast (English only, lower WER)', 'FP32 projected-cache dedicated English model; largest footprint.', 'fast', 'english'],
];

const BACKENDS = [
    ['parakeet', 'Parakeet'],
    ['nemo-speech', 'NeMo-Speech.cpp'],
];

const SHORTCUT_MODES = [
    {id: 'toggle'},
    {id: 'push-to-talk'},
];

const LANGUAGES = [
    ['en-US', 'English (US)'],
    ['en-GB', 'English (UK)'],
    ['auto', 'Auto'],
    ['es-US', 'Spanish (US)'],
    ['es-ES', 'Spanish (Spain)'],
    ['fr-FR', 'French (France)'],
    ['fr-CA', 'French (Canada)'],
    ['de-DE', 'German'],
    ['it-IT', 'Italian'],
    ['pt-BR', 'Portuguese (Brazil)'],
    ['pt-PT', 'Portuguese (Portugal)'],
    ['nl-NL', 'Dutch'],
    ['tr-TR', 'Turkish'],
    ['ru-RU', 'Russian'],
    ['ar-AR', 'Arabic'],
    ['hi-IN', 'Hindi'],
    ['ja-JP', 'Japanese'],
    ['ko-KR', 'Korean'],
    ['vi-VN', 'Vietnamese'],
    ['uk-UA', 'Ukrainian'],
    ['pl-PL', 'Polish'],
    ['sv-SE', 'Swedish'],
    ['cs-CZ', 'Czech'],
    ['nb-NO', 'Norwegian Bokmal'],
    ['da-DK', 'Danish'],
    ['bg-BG', 'Bulgarian'],
    ['fi-FI', 'Finnish'],
    ['hr-HR', 'Croatian'],
    ['sk-SK', 'Slovak'],
    ['zh-CN', 'Chinese (Simplified)'],
    ['zh-TW', 'Chinese (Traditional)'],
    ['hu-HU', 'Hungarian'],
    ['ro-RO', 'Romanian'],
    ['et-EE', 'Estonian'],
    ['el-GR', 'Greek'],
    ['lt-LT', 'Lithuanian'],
    ['lv-LV', 'Latvian'],
    ['bn-IN', 'Bengali'],
    ['id-ID', 'Indonesian'],
    ['ms-MY', 'Malay'],
    ['th-TH', 'Thai'],
];

const ShortcutSettingButton = GObject.registerClass({
    Properties: {
        shortcut: GObject.ParamSpec.string(
            'shortcut',
            'shortcut',
            'The shortcut',
            GObject.ParamFlags.READWRITE,
            ''),
    },
    Signals: {
        changed: {param_types: [GObject.TYPE_STRING]},
    },
}, class ShortcutSettingButton extends Gtk.Button {
    constructor(settingsKey, settings) {
        super({
            halign: Gtk.Align.CENTER,
            hexpand: false,
            vexpand: false,
            has_frame: false,
        });
        this._settingsKey = settingsKey;
        this._settings = settings;
        this._editor = null;
        this._previousShortcut = '';
        this._recordingShortcut = false;
        this._shortcut = '';
        this._label = new Gtk.ShortcutLabel({
            disabled_text: _('Not Set'),
            valign: Gtk.Align.CENTER,
            hexpand: false,
            vexpand: false,
        });

        this.connect('clicked', this._onActivated.bind(this));
        this._settings.connect(`changed::${this._settingsKey}`, () => this.syncFromSettings());
        this.set_child(this._label);
        this.syncFromSettings();
    }

    set shortcut(value) {
        this._shortcut = value;
    }

    get shortcut() {
        return this._shortcut;
    }

    syncFromSettings() {
        [this.shortcut] = this._settings.get_strv(this._settingsKey);
        this._label.set_accelerator(this.shortcut ?? '');
    }

    _onActivated(widget) {
        const controller = new Gtk.EventControllerKey();
        const content = new Adw.StatusPage({
            title: _('Set Dictation Shortcut'),
            icon_name: 'preferences-desktop-keyboard-shortcuts-symbolic',
            description: _('Press a key combination. If it does not appear, it may already be used by GNOME.'),
        });
        this._editor = new Adw.Window({
            modal: true,
            hide_on_close: true,
            transient_for: widget.get_root(),
            width_request: 480,
            height_request: 320,
            content,
        });
        this._editor.add_controller(controller);
        controller.connect('key-pressed', this._onKeyPressed.bind(this));
        this._editor.connect('close-request', () => {
            this._finishRecording(false);
            return false;
        });
        this._previousShortcut = this._settings.get_strv(this._settingsKey)[0] ?? '';
        this._recordingShortcut = true;
        this._settings.set_boolean('shortcut-capture-active', true);
        this._settings.set_strv(this._settingsKey, []);
        this._label.set_accelerator(this._previousShortcut);
        GLib.timeout_add(GLib.PRIORITY_DEFAULT, 150, () => {
            this._editor?.present();
            return GLib.SOURCE_REMOVE;
        });
    }

    _onKeyPressed(_controller, keyval, keycode, state) {
        let mask = state & Gtk.accelerator_get_default_mod_mask();
        mask &= ~Gdk.ModifierType.LOCK_MASK;

        if (!mask && keyval === Gdk.KEY_Escape) {
            this._editor?.close();
            return true;
        }

        if (keyval === Gdk.KEY_BackSpace) {
            this._updateShortcut('');
            this._editor?.close();
            return true;
        }

        if (!this._isValidBinding(mask, keycode, keyval) ||
            !this._isValidAccel(mask, keyval))
            return true;

        if (!keyval && !keycode) {
            this._editor?.destroy();
            return true;
        }

        const accelerator = Gtk.accelerator_name_with_keycode(
            null,
            keyval,
            keycode,
            mask);
        this._updateShortcut(accelerator);
        this._editor?.destroy();
        return true;
    }

    _updateShortcut(accelerator) {
        this.shortcut = accelerator;
        this._label.set_accelerator(this.shortcut);
        this._settings.set_strv(this._settingsKey, accelerator ? [accelerator] : []);
        this._finishRecording(true);
        this.emit('changed', this.shortcut);
    }

    _finishRecording(accepted) {
        if (!this._recordingShortcut)
            return;

        this._recordingShortcut = false;
        if (!accepted) {
            this.shortcut = this._previousShortcut;
            this._label.set_accelerator(this.shortcut);
            this._settings.set_strv(this._settingsKey,
                this._previousShortcut ? [this._previousShortcut] : []);
        }
        this._previousShortcut = '';
        this._settings.set_boolean('shortcut-capture-active', false);
    }

    _keyvalIsForbidden(keyval) {
        return [
            Gdk.KEY_Home,
            Gdk.KEY_Left,
            Gdk.KEY_Up,
            Gdk.KEY_Right,
            Gdk.KEY_Down,
            Gdk.KEY_Page_Up,
            Gdk.KEY_Page_Down,
            Gdk.KEY_End,
            Gdk.KEY_Tab,
            Gdk.KEY_KP_Enter,
            Gdk.KEY_Return,
            Gdk.KEY_Mode_switch,
        ].includes(keyval);
    }

    _isValidBinding(mask, keycode, keyval) {
        return !(
            mask === 0 ||
            (mask === Gdk.ModifierType.SHIFT_MASK &&
                keycode !== 0 &&
                ((keyval >= Gdk.KEY_a && keyval <= Gdk.KEY_z) ||
                    (keyval >= Gdk.KEY_A && keyval <= Gdk.KEY_Z) ||
                    (keyval >= Gdk.KEY_0 && keyval <= Gdk.KEY_9) ||
                    (keyval >= Gdk.KEY_kana_fullstop &&
                        keyval <= Gdk.KEY_semivoicedsound) ||
                    (keyval >= Gdk.KEY_Arabic_comma &&
                        keyval <= Gdk.KEY_Arabic_sukun) ||
                    (keyval >= Gdk.KEY_Serbian_dje &&
                        keyval <= Gdk.KEY_Cyrillic_HARDSIGN) ||
                    (keyval >= Gdk.KEY_Greek_ALPHAaccent &&
                        keyval <= Gdk.KEY_Greek_omega) ||
                    (keyval >= Gdk.KEY_hebrew_doublelowline &&
                        keyval <= Gdk.KEY_hebrew_taf) ||
                    (keyval >= Gdk.KEY_Thai_kokai &&
                        keyval <= Gdk.KEY_Thai_lekkao) ||
                    (keyval >= Gdk.KEY_Hangul_Kiyeog &&
                        keyval <= Gdk.KEY_Hangul_J_YeorinHieuh) ||
                    (keyval === Gdk.KEY_space && mask === 0) ||
                    this._keyvalIsForbidden(keyval)))
        );
    }

    _isValidAccel(mask, keyval) {
        return Gtk.accelerator_valid(keyval, mask) ||
            (keyval === Gdk.KEY_Tab && mask !== 0);
    }
});

const WordpipePage = GObject.registerClass(
class WordpipePage extends Adw.PreferencesPage {
    constructor(settings) {
        super({
            title: _('General'),
            icon_name: 'audio-input-microphone-symbolic',
        });
        this._settings = settings;
        this._proxy = null;
        this._signalIds = [];
        this._syncingSettings = false;
        this._selectedPreset = modelPresetId(
            this._settings.get_string('model-profile'),
            this._settings.get_string('model-family'));
        this._backends = BACKENDS.map(([id, title]) => ({id, title, description: ''}));
        this._allLanguages = LANGUAGES.map(([id, title]) => ({id, title}));
        this._languages = [];
        this._profiles = PROFILES.map(([id, title, description, modelProfile, modelFamily]) => ({
            id,
            title,
            description,
            model_profile: modelProfile,
            model_family: modelFamily,
            installed: false,
            runtime_dir: '',
        }));
        this._selectableProfiles = installedModelProfiles(this._profiles);
        this._deviceSelectors = [''];
        this._installRows = new Map();
        this._installButtons = new Map();
        this._installing = false;
        this._installingProfile = '';
        this.advancedPage = new Adw.PreferencesPage({
            title: _('Advanced'),
            icon_name: 'preferences-other-symbolic',
        });
        this._buildGeneralSection();
        this._buildAdvancedSection();
        this._connectProxy();
    }

    vfunc_unroot() {
        if (this._proxy) {
            for (const id of this._signalIds)
                this._proxy.disconnectSignal(id);
            this._signalIds = [];
            this._proxy = null;
        }
        super.vfunc_unroot();
    }

    _buildGeneralSection() {
        this._buildModelGroup(this);
        this._buildInputGroup(this);
        this._buildBehaviorGroup(this);
    }

    _buildAdvancedSection() {
        this._buildAdvancedGroup(this.advancedPage);
        this._buildTranscriptGroup(this.advancedPage);
        this._buildServiceGroup(this.advancedPage);
    }

    _appendSection(section, widget) {
        section.add(widget);
    }

    _buildModelGroup(section) {
        this._modelGroup = new Adw.PreferencesGroup({
            title: _('Model'),
        });
        this._appendSection(section, this._modelGroup);

        this._backendModel = new Gtk.StringList();
        this._backends.forEach(backend => this._backendModel.append(backend.title));
        this._backendRow = new Adw.ComboRow({
            title: _('Backend'),
            model: this._backendModel,
        });
        this._backendRow.selected = this._selectedIndex(
            this._backends, this._settings.get_string('backend'));
        this._backendRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const backend = this._backends[row.selected]?.id;
            if (!backend)
                return;
            this._settings.set_string('backend', backend);
            this._callRemote('SetBackend', backend);
        });
        this._modelGroup.add(this._backendRow);

        this._profileModel = new Gtk.StringList();
        this._selectableProfiles.forEach(profile => this._profileModel.append(profile.title));
        this._profileRow = new Adw.ComboRow({
            title: _('Model Preset'),
            subtitle: _('Install a model below before selecting it.'),
            model: this._profileModel,
            sensitive: this._selectableProfiles.length > 0,
        });
        this._profileRow.selected = this._selectedProfileIndex(this._selectedPresetId());
        this._profileRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const preset = this._selectableProfiles[row.selected];
            if (!preset)
                return;
            this._selectedPreset = preset.id;
            this._withSyncing(() => {
                this._settings.set_string('model-profile', preset.model_profile);
                this._settings.set_string('model-family', preset.model_family);
            });
            this._callRemote('SetModelProfile', preset.id);
        });
        this._modelGroup.add(this._profileRow);

        this._latencyValues = [560, 1120]; // Compatibility fallback for older services.
        this._latencyRow = new Adw.ComboRow({
            title: _('Streaming Chunk Size'),
            subtitle: _('Choose a chunk size supported by the installed model; generic exports share one download.'),
            model: Gtk.StringList.new(this._latencyValues.map(value => `${value} ms`)),
            selected: this._settings.get_uint('streaming-latency-ms') === 1120 ? 1 : 0,
        });
        this._latencyRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const latency = this._latencyValues[row.selected];
            if (!Number.isInteger(latency))
                return;
            this._settings.set_uint('streaming-latency-ms', latency);
            this._pushRuntimeOptions();
        });
        this._modelGroup.add(this._latencyRow);

        this._languageModel = new Gtk.StringList();
        this._languageRow = new Adw.ComboRow({
            title: _('Dictation Language'),
            model: this._languageModel,
        });
        this._updateLanguageOptions();
        this._languageRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const language = this._languages[row.selected]?.id;
            if (!language)
                return;
            this._settings.set_string('language', language);
            this._pushRuntimeOptions();
        });
        this._modelGroup.add(this._languageRow);

        this._rebuildProfileRows();
    }

    _buildInputGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Input'),
        });
        this._appendSection(section, group);

        this._deviceModel = new Gtk.StringList();
        this._deviceModel.append(_('System Default'));
        this._deviceRow = new Adw.ComboRow({
            title: _('Microphone'),
            subtitle: _('The service uses the system default when no device is selected.'),
            model: this._deviceModel,
        });
        this._deviceRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const selector = this._deviceSelectors[row.selected] ?? '';
            this._settings.set_string('input-device', selector);
            this._callRemote('SetInputDevice', selector);
        });
        this._deviceRow.connect('activated', () => this._refreshInputDevices());
        group.add(this._deviceRow);
    }

    _buildBehaviorGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Behavior'),
        });
        this._appendSection(section, group);

        let row = new Adw.SwitchRow({
            title: _('Spoken Punctuation'),
            active: this._settings.get_boolean('spoken-punctuation'),
        });
        row.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('spoken-punctuation', widget.active);
            this._pushInsertionOptions();
        });
        this._spokenPunctuationRow = row;
        group.add(row);

        row = new Adw.SwitchRow({
            title: _('Inverse Text Normalization'),
            subtitle: _('NeMo only: format numbers, dates and units. Grammars: English, Arabic, Chinese, French, German, Hindi and Spanish. Change while stopped.'),
            active: this._settings.get_boolean('itn'),
        });
        row.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('itn', widget.active);
            this._pushRuntimeOptions();
        });
        this._itnRow = row;
        group.add(row);
        this._endpointModes = ['disabled', 'preview', 'commit'];
        this._endpointModeRow = new Adw.ComboRow({
            title: _('Pause Endpointing'),
            subtitle: _('NeMo only. Uses about 800 ms of decoded token silence. ITN switch controls normalization. Preview mode needs partial insertion to show text before Stop.'),
            model: Gtk.StringList.new([_('Disabled'), _('ITN Preview — Do Not Finalize'), _('ITN + Finalize / Commit')]),
            selected: Math.max(0, this._endpointModes.indexOf(this._settings.get_string('endpoint-mode'))),
        });
        this._endpointModeRow.connect('notify::selected', widget => {
            if (this._syncingSettings)
                return;
            const mode = this._endpointModes[widget.selected];
            if (!mode)
                return;
            this._settings.set_string('endpoint-mode', mode);
            this._pushRuntimeOptions();
        });
        group.add(this._endpointModeRow);

        this._phraseBoostingRow = new Adw.SwitchRow({
            title: _('Vocabulary / Phrase Boosting'),
            subtitle: _('NeMo only: favor the names and terms below. May also bias incorrect matches. Change while stopped.'),
            active: this._settings.get_boolean('phrase-boosting'),
        });
        this._phraseBoostingRow.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('phrase-boosting', widget.active);
            this._pushRuntimeOptions();
        });
        group.add(this._phraseBoostingRow);
        this._boostPhrasesRow = new Adw.ExpanderRow({
            title: _('Boosted Names and Phrases'),
            subtitle: _('One phrase per line. Save before enabling. A matching tokenizer is downloaded automatically when needed.'),
        });
        this._boostPhrasesBuffer = new Gtk.TextBuffer({text: this._settings.get_string('boost-phrases')});
        const editor = new Gtk.TextView({
            buffer: this._boostPhrasesBuffer,
            wrap_mode: Gtk.WrapMode.WORD_CHAR,
            top_margin: 8, bottom_margin: 8, left_margin: 12, right_margin: 12,
        });
        const scroll = new Gtk.ScrolledWindow({min_content_height: 120, child: editor});
        this._boostPhrasesRow.add_row(new Adw.PreferencesRow({child: scroll}));
        const savePhrases = new Gtk.Button({label: _('Save Phrases'), valign: Gtk.Align.CENTER});
        savePhrases.connect('clicked', () => {
            const buffer = this._boostPhrasesBuffer;
            this._settings.set_string('boost-phrases', buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), false));
            this._pushRuntimeOptions();
        });
        this._boostPhrasesRow.add_suffix(savePhrases);
        group.add(this._boostPhrasesRow);
        this._boostTokenizerPathRow = new Adw.EntryRow({
            title: _('Boosting Tokenizer Override (Optional)'),
            text: this._settings.get_string('boost-tokenizer-path'),
            show_apply_button: true,
        });
        this._boostTokenizerPathRow.connect('apply', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_string('boost-tokenizer-path', widget.text);
            this._pushRuntimeOptions();
        });
        group.add(this._boostTokenizerPathRow);

        this._vadModelPathRow = new Adw.EntryRow({
            title: _('Silero VAD GGUF Override (Optional)'),
            text: this._settings.get_string('vad-model-path'),
            show_apply_button: true,
        });
        this._vadModelPathRow.connect('apply', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_string('vad-model-path', widget.text);
            this._pushRuntimeOptions();
        });
        group.add(this._vadModelPathRow);
        this._vadFilteringRow = new Adw.SwitchRow({
            title: _('VAD Filtering'),
            subtitle: _('NeMo only: mask non-speech features. Silero is downloaded automatically. Does not enable endpointing. Change while stopped.'),
            active: this._settings.get_boolean('vad-filtering'),
        });
        this._vadFilteringRow.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('vad-filtering', widget.active);
            this._pushRuntimeOptions();
        });
        group.add(this._vadFilteringRow);

        this._insertPartialsRow = new Adw.ComboRow({
            title: _('Text Insertion'),
            subtitle: _('Partials are replaceable previews. Committed-only inserts at finalized endpoints or Stop. Change while stopped.'),
            model: Gtk.StringList.new([_('Partial + Committed'), _('Committed Only')]),
            selected: this._settings.get_boolean('insert-partials') ? 0 : 1,
        });
        this._insertPartialsRow.connect('notify::selected', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('insert-partials', widget.selected === 0);
            this._pushInsertionOptions();
        });
        group.add(this._insertPartialsRow);

        this._delayRow = Adw.SpinRow.new_with_range(0, 1000, 25);
        this._delayRow.title = _('Insertion Delay');
        this._delayRow.subtitle = _('Additional delay in milliseconds before inserting streamed text.');
        this._delayRow.value = this._settings.get_uint('stream-insert-delay-ms');
        this._delayRow.connect('notify::value', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_uint('stream-insert-delay-ms', Math.round(widget.value));
            this._pushInsertionOptions();
        });
        group.add(this._delayRow);

        const shortcutButton = new ShortcutSettingButton('toggle-shortcut', this._settings);
        shortcutButton.connect('changed',
            (_button, accelerator) => this._callRemote('SetShortcut', accelerator));
        shortcutButton.set_vexpand(false);
        shortcutButton.set_valign(Gtk.Align.CENTER);
        const shortcutRow = new Adw.ActionRow({
            title: _('Shortcut'),
            subtitle: _('Click to set the dictation shortcut.'),
        });
        shortcutRow.add_suffix(shortcutButton);
        shortcutRow.set_activatable_widget(shortcutButton);
        this._shortcutButton = shortcutButton;
        this._shortcutRow = shortcutRow;
        group.add(shortcutRow);

        this._shortcutModeModel = new Gtk.StringList();
        SHORTCUT_MODES.forEach(mode =>
            this._shortcutModeModel.append(shortcutModeTitle(mode.id)));
        this._shortcutModeRow = new Adw.ComboRow({
            title: _('Shortcut Behavior'),
            subtitle: _('Toggle on each press, or hold the shortcut while speaking.'),
            model: this._shortcutModeModel,
        });
        this._shortcutModeRow.selected = this._selectedIndex(
            SHORTCUT_MODES,
            this._settings.get_string('shortcut-mode'));
        this._shortcutModeRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const mode = SHORTCUT_MODES[row.selected]?.id;
            if (mode)
                this._settings.set_string('shortcut-mode', mode);
        });
        group.add(this._shortcutModeRow);
    }

    _buildAdvancedGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Advanced'),
        });
        this._appendSection(section, group);

        this._modelRootRow = new Adw.EntryRow({
            title: _('Model Directory'),
            text: this._settings.get_string('model-root'),
        });
        this._modelRootRow.connect('changed', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_string('model-root', row.text.trim());
            this._pushRuntimeOptions();
        });
        group.add(this._modelRootRow);

        this._workerPathRow = new Adw.EntryRow({
            title: _('Worker Binary'),
            text: this._settings.get_string('worker-path'),
        });
        this._workerPathRow.connect('changed', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_string('worker-path', row.text.trim());
            this._pushRuntimeOptions();
        });
        group.add(this._workerPathRow);

        this._modelInstallerPathRow = new Adw.EntryRow({
            title: _('Model Installer'),
            text: this._settings.get_string('model-installer-path'),
        });
        this._modelInstallerPathRow.connect('changed', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_string('model-installer-path', row.text.trim());
            this._pushRuntimeOptions();
        });
        group.add(this._modelInstallerPathRow);

        this._threadsRow = Adw.SpinRow.new_with_range(1, 16, 1);
        this._threadsRow.title = _('Worker Threads');
        this._threadsRow.value = this._settings.get_uint('num-threads');
        this._threadsRow.connect('notify::value', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_uint('num-threads', Math.max(1, Math.round(row.value)));
            this._pushRuntimeOptions();
        });
        group.add(this._threadsRow);

        this._sampleRateRow = Adw.SpinRow.new_with_range(8000, 48000, 1000);
        this._sampleRateRow.title = _('Sample Rate');
        this._sampleRateRow.value = this._settings.get_uint('sample-rate');
        this._sampleRateRow.connect('notify::value', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_uint('sample-rate', Math.max(1, Math.round(row.value)));
            this._pushRuntimeOptions();
        });
        group.add(this._sampleRateRow);
    }

    _buildTranscriptGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Transcript'),
        });
        this._appendSection(section, group);

        this._partialRow = new Adw.ActionRow({
            title: _('Live Transcript'),
            subtitle: _('No speech yet'),
        });
        group.add(this._partialRow);

        this._commitRow = new Adw.ActionRow({
            title: _('Last Committed'),
            subtitle: _('Nothing committed'),
        });
        group.add(this._commitRow);
    }

    _buildServiceGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Service'),
        });
        this._appendSection(section, group);

        this._statusRow = new Adw.ActionRow({
            title: _('Status'),
            subtitle: _('Connecting'),
        });
        group.add(this._statusRow);

        this._progressRow = new Adw.ActionRow({
            title: _('Model Setup'),
            subtitle: _('Idle'),
        });
        this._progressBar = new Gtk.ProgressBar({
            valign: Gtk.Align.CENTER,
            width_request: 140,
            visible: false,
        });
        this._progressRow.add_suffix(this._progressBar);
        group.add(this._progressRow);

        this._metricsRow = new Adw.ActionRow({
            title: _('Runtime Metrics'),
            subtitle: _('No metrics yet'),
        });
        group.add(this._metricsRow);

        const actions = new Adw.ActionRow({
            title: _('Dictation'),
        });
        this._startButton = new Gtk.Button({
            icon_name: 'media-record-symbolic',
            valign: Gtk.Align.CENTER,
            tooltip_text: _('Start dictation'),
        });
        this._startButton.connect('clicked', () => this._callRemote('Start'));
        this._stopButton = new Gtk.Button({
            icon_name: 'media-playback-stop-symbolic',
            valign: Gtk.Align.CENTER,
            tooltip_text: _('Stop dictation'),
        });
        this._stopButton.connect('clicked', () => this._callRemote('Stop'));
        actions.add_suffix(this._startButton);
        actions.add_suffix(this._stopButton);
        group.add(actions);
    }

    _connectProxy() {
        this._proxy = new WordpipeProxy(
            Gio.DBus.session,
            BUS_NAME,
            OBJECT_PATH,
            (_proxy, error) => {
                if (error) {
                    this._statusRow.subtitle = _('Service unavailable');
                    return;
                }
                this._statusRow.subtitle = _('Connected');
                this._subscribeSignals();
                this._refreshBackends();
                this._refreshModelProfiles();
                this._refreshConfig();
                this._refreshState();
                this._refreshInputDevices();
            });
    }

    _subscribeSignals() {
        if (this._signalIds.length > 0)
            return;
        this._signalIds.push(this._proxy.connectSignal('StateChanged',
            (_proxy, _sender, [state]) => this._handleState(deepUnpackMap(state))));
        this._signalIds.push(this._proxy.connectSignal('ConfigChanged',
            (_proxy, _sender, [config]) => {
                this._syncFromConfig(deepUnpackMap(config));
                this._refreshModelProfiles();
                this._refreshInputDevices();
            }));
        this._signalIds.push(this._proxy.connectSignal('InstallProgress',
            (_proxy, _sender, [profile, progress]) => {
                this._handleInstallProgress(profile, deepUnpackMap(progress));
            }));
        this._signalIds.push(this._proxy.connectSignal('Partial',
            (_proxy, _sender, [_sessionId, _seq, fullText]) => {
                this._partialRow.subtitle = fullText || _('No speech yet');
            }));
        this._signalIds.push(this._proxy.connectSignal('Commit',
            (_proxy, _sender, [_sessionId, _seq, text]) => {
                this._commitRow.subtitle = text || _('Nothing committed');
            }));
        this._signalIds.push(this._proxy.connectSignal('Metrics',
            (_proxy, _sender, [metrics]) => {
                const summary = formatMetrics(deepUnpackMap(metrics));
                if (summary)
                    this._metricsRow.subtitle = summary;
            }));
        this._signalIds.push(this._proxy.connectSignal('Error',
            (_proxy, _sender, [message]) => {
                this._statusRow.subtitle = message;
                this._progressRow.subtitle = message;
            }));
    }

    _refreshState() {
        this._callRemote('GetState', state => {
            this._handleState(deepUnpackMap(state));
        });
    }

    _refreshConfig() {
        this._callRemote('GetConfig', config => {
            this._syncFromConfig(deepUnpackMap(config));
        });
    }

    _refreshBackends() {
        this._callRemote('ListBackends', backends => {
            const parsed = backends.map(item => deepUnpackMap(item))
                .filter(item => typeof item.id === 'string');
            if (parsed.length === 0)
                return;
            this._backends = parsed.map(item => ({
                id: item.id,
                title: item.title ?? item.id,
                description: item.description ?? '',
            }));
            this._withSyncing(() => {
                clearStringList(this._backendModel);
                this._backends.forEach(backend => this._backendModel.append(backend.title));
                this._syncComboSelections();
            });
        });
    }

    _refreshModelProfiles() {
        this._callRemote('ListModelProfiles', profiles => {
            const parsed = profiles.map(item => deepUnpackMap(item))
                .filter(item => typeof item.id === 'string');
            if (parsed.length === 0)
                return;
            this._profiles = parsed.map(item => ({
                id: item.id,
                title: item.title ?? item.id,
                description: item.description ?? '',
                model_profile: item.model_profile ?? item.id,
                model_family: item.model_family ?? 'multilingual',
                installed: Boolean(item.installed),
                runtime_dir: item.runtime_dir ?? '',
            }));
            this._rebuildProfileOptions();
            this._rebuildProfileRows();
            this._syncComboSelections();
        });
    }

    _refreshInputDevices() {
        this._callRemote('ListInputDevices', devices => {
            // GTK can emit notify::selected while the list is cleared or
            // repopulated. Guard the whole rebuild, not only the final selection,
            // or SetInputDevice -> ConfigChanged -> refresh loops indefinitely.
            const configured = this._settings.get_string('input-device');
            this._withSyncing(() => {
                clearStringList(this._deviceModel);
                this._deviceSelectors = [''];
                this._deviceModel.append(_('System Default'));

                for (const device of devices) {
                    const values = deepUnpackMap(device);
                    const selector = values.selector ?? values.name ?? '';
                    this._deviceSelectors.push(selector);
                    this._deviceModel.append(values.is_default
                        ? _(`${values.name} (default)`)
                        : values.name);
                }

                this._deviceRow.selected = Math.max(0, this._deviceSelectors.indexOf(configured));
            });
        });
    }

    _syncFromConfig(values) {
        const selectedPreset = modelPresetFromValues(values);
        if (selectedPreset)
            this._selectedPreset = selectedPreset;
        this._syncingSettings = true;
        try {
            const chunks = streamingChunkChoices(values.supported_streaming_latencies_ms);
            if (chunks) {
                this._latencyValues = chunks;
                this._latencyRow.model = Gtk.StringList.new(this._latencyValues.map(value => `${value} ms`));
            }
            if (typeof values.backend === 'string')
                this._settings.set_string('backend', values.backend);
            if (typeof values.model_profile === 'string')
                this._settings.set_string('model-profile', values.model_profile);
            if (typeof values.model_family === 'string')
                this._settings.set_string('model-family', values.model_family);
            if (typeof values.streaming_latency_ms === 'number')
                this._settings.set_uint('streaming-latency-ms', values.streaming_latency_ms);
            if (typeof values.input_device === 'string')
                this._settings.set_string('input-device', values.input_device);
            if (typeof values.language === 'string')
                this._settings.set_string('language', values.language);
            if (typeof values.model_root === 'string')
                this._settings.set_string('model-root', values.model_root);
            if (typeof values.worker_path === 'string')
                this._settings.set_string('worker-path', values.worker_path);
            if (typeof values.model_installer_path === 'string')
                this._settings.set_string('model-installer-path', values.model_installer_path);
            if (typeof values.shortcut === 'string')
                this._settings.set_strv('toggle-shortcut', values.shortcut ? [values.shortcut] : []);
            if (typeof values.num_threads === 'number')
                this._settings.set_uint('num-threads', values.num_threads);
            if (typeof values.sample_rate === 'number')
                this._settings.set_uint('sample-rate', values.sample_rate);
            if (typeof values.spoken_punctuation === 'boolean')
                this._settings.set_boolean('spoken-punctuation', values.spoken_punctuation);
            if (typeof values.insert_partials === 'boolean')
                this._settings.set_boolean('insert-partials', values.insert_partials);
            if (typeof values.itn === 'boolean')
                this._settings.set_boolean('itn', values.itn);
            if (typeof values.endpoint_mode === 'string')
                this._settings.set_string('endpoint-mode', values.endpoint_mode);
            for (const [key, setting] of [['phrase_boosting', 'phrase-boosting'], ['vad_filtering', 'vad-filtering']]) {
                if (typeof values[key] === 'boolean')
                    this._settings.set_boolean(setting, values[key]);
            }
            for (const [key, setting] of [['boost_phrases', 'boost-phrases'], ['boost_tokenizer_path', 'boost-tokenizer-path'], ['vad_model_path', 'vad-model-path']]) {
                if (typeof values[key] === 'string')
                    this._settings.set_string(setting, values[key]);
            }
            if (typeof values.stream_insert_delay_ms === 'number')
                this._settings.set_uint('stream-insert-delay-ms', values.stream_insert_delay_ms);
            if (typeof values.show_overlay === 'boolean')
                this._settings.set_boolean('show-overlay', false);

            this._syncComboSelections();
            this._syncDeviceSelection();
            this._syncControlValues();
        } finally {
            this._syncingSettings = false;
        }
    }

    _syncComboSelections() {
        if (!this._backendRow || !this._profileRow || !this._languageRow)
            return;
        const backend = this._settings.get_string('backend');
        const preset = this._selectedPresetId();
        const family = this._settings.get_string('model-family');
        const language = this._settings.get_string('language');
        this._withSyncing(() => {
            this._updateLanguageOptions(family, language);
            this._backendRow.selected = this._selectedIndex(this._backends, backend);
            this._profileRow.selected = this._selectedProfileIndex(preset);
            this._languageRow.selected = this._selectedIndex(this._languages, language);
        });
    }

    _syncDeviceSelection() {
        if (!this._deviceRow)
            return;
        const configured = this._settings.get_string('input-device');
        const selected = Math.max(0, this._deviceSelectors.indexOf(configured));
        this._withSyncing(() => {
            this._deviceRow.selected = selected;
        });
    }

    _syncControlValues() {
        this._spokenPunctuationRow.active = this._settings.get_boolean('spoken-punctuation');
        this._insertPartialsRow.selected = this._settings.get_boolean('insert-partials') ? 0 : 1;
        this._itnRow.active = this._settings.get_boolean('itn');
        this._itnRow.sensitive = this._settings.get_string('backend') === 'nemo-speech';
        this._endpointModeRow.sensitive = this._itnRow.sensitive;
        this._endpointModeRow.selected = Math.max(0, this._endpointModes.indexOf(this._settings.get_string('endpoint-mode')));
        for (const row of [this._phraseBoostingRow, this._boostPhrasesRow, this._boostTokenizerPathRow, this._vadFilteringRow, this._vadModelPathRow])
            row.sensitive = this._itnRow.sensitive;
        this._vadModelPathRow.text = this._settings.get_string('vad-model-path');
        this._boostTokenizerPathRow.text = this._settings.get_string('boost-tokenizer-path');
        const phrases = this._settings.get_string('boost-phrases');
        const buffer = this._boostPhrasesBuffer;
        if (buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), false) !== phrases)
            buffer.set_text(phrases, -1);
        this._phraseBoostingRow.active = this._settings.get_boolean('phrase-boosting');
        this._vadFilteringRow.active = this._settings.get_boolean('vad-filtering');
        this._delayRow.value = this._settings.get_uint('stream-insert-delay-ms');
        this._modelRootRow.text = this._settings.get_string('model-root');
        this._workerPathRow.text = this._settings.get_string('worker-path');
        this._modelInstallerPathRow.text = this._settings.get_string('model-installer-path');
        this._threadsRow.value = this._settings.get_uint('num-threads');
        this._sampleRateRow.value = this._settings.get_uint('sample-rate');
        this._latencyRow.selected = Math.max(0, this._latencyValues.indexOf(this._settings.get_uint('streaming-latency-ms')));
        this._shortcutModeRow.selected = this._selectedIndex(
            SHORTCUT_MODES,
            this._settings.get_string('shortcut-mode'));
        this._syncShortcutValue();
    }

    _rebuildProfileRows() {
        for (const row of this._installRows.values())
            this._modelGroup.remove(row);
        this._installRows.clear();
        this._installButtons.clear();

        for (const profile of this._profiles) {
            const installing = this._installing && this._installingProfile === profile.id;
            const row = new Adw.ActionRow({
                title: installing
                    ? _(`Installing ${profile.title}`)
                    : profile.installed
                    ? profile.title
                    : _(`Install ${profile.title}`),
                subtitle: this._profileSubtitle(profile),
            });
            const button = new Gtk.Button({
                icon_name: profile.installed
                    ? 'emblem-ok-symbolic'
                    : installing
                        ? 'emblem-synchronizing-symbolic'
                        : 'folder-download-symbolic',
                valign: Gtk.Align.CENTER,
                sensitive: !profile.installed && !this._installing,
                tooltip_text: profile.installed
                    ? _(`${profile.title} is installed`)
                    : installing
                        ? _(`${profile.title} is installing`)
                    : _(`Download and prepare the ${profile.title} model`),
            });
            button.connect('clicked', () => {
                if (profile.installed || this._installing)
                    return;
                this._progressRow.subtitle = _(`Starting ${profile.title}`);
                this._callRemote('InstallModel', profile.id);
            });
            row.add_suffix(button);
            this._installRows.set(profile.id, row);
            this._installButtons.set(profile.id, button);
            this._modelGroup.add(row);
        }
    }

    _profileSubtitle(profile) {
        if (this._installing && this._installingProfile === profile.id)
            return _('Installing');
        const status = profile.installed ? _('Installed') : _('Not installed');
        const detail = profile.description || profile.runtime_dir;
        return detail ? `${status} - ${detail}` : status;
    }

    _handleState(values) {
        const selectedPreset = modelPresetFromValues(values);
        if (selectedPreset && selectedPreset !== this._selectedPreset) {
            this._selectedPreset = selectedPreset;
            if (this._profileRow) {
                this._withSyncing(() => {
                    this._profileRow.selected = this._selectedProfileIndex(selectedPreset);
                });
            }
        }
        const previousInstalling = this._installing;
        const previousInstallingProfile = this._installingProfile;
        this._installing = Boolean(values.installing);
        this._installingProfile = values.installing_profile ?? '';
        this._latencyRow.sensitive = !values.loading_model && !values.listening &&
            !values.stopping && !values.installing;
        const selectedModelInstalled = values.selected_model_installed !== false;
        if (values.loading_model)
            this._statusRow.subtitle = _('Loading model');
        else if (values.listening)
            this._statusRow.subtitle = _('Listening');
        else if (values.stopping)
            this._statusRow.subtitle = _('Stopping');
        else if (values.installing)
            this._statusRow.subtitle = _('Installing model');
        else if (!selectedModelInstalled)
            this._statusRow.subtitle = _('Model missing');
        else if (values.last_error)
            this._statusRow.subtitle = values.last_error;
        else
            this._statusRow.subtitle = _('Ready');
        const metricsSummary = formatMetrics(values.last_metrics ?? {});
        if (metricsSummary)
            this._metricsRow.subtitle = metricsSummary;
        const installSummary = formatInstallProgress(values.last_install_progress ?? {});
        if (installSummary)
            this._setInstallProgress(values.last_install_progress ?? {});
        if (typeof values.partial_text === 'string')
            this._partialRow.subtitle = values.partial_text || _('No speech yet');
        if (typeof values.last_commit_text === 'string')
            this._commitRow.subtitle = values.last_commit_text || _('Nothing committed');
        if (this._startButton) {
            this._startButton.sensitive = !values.loading_model &&
                !values.installing &&
                !values.listening &&
                !values.stopping &&
                selectedModelInstalled;
        }
        if (this._stopButton)
            this._stopButton.sensitive = Boolean(values.listening || values.stopping);
        this._syncInstallButtons();
        if (
            previousInstalling !== this._installing ||
            previousInstallingProfile !== this._installingProfile
        )
            this._rebuildProfileRows();
    }

    _handleInstallProgress(profile, progress) {
        if (typeof progress.profile !== 'string')
            progress.profile = profile;
        this._setInstallProgress(progress);
        const summary = formatInstallProgress(progress);
        if (summary)
            this._statusRow.subtitle = summary;
        if (progress.phase === 'complete' || progress.phase === 'error')
            this._refreshModelProfiles();
    }

    _setInstallProgress(progress) {
        const summary = formatInstallProgress(progress);
        if (summary)
            this._progressRow.subtitle = summary;
        const fraction = numberValue(progress.fraction);
        const active = progress.phase !== 'complete' && progress.phase !== 'error';
        if (fraction !== null && active) {
            this._progressBar.visible = true;
            this._progressBar.fraction = Math.max(0.0, Math.min(1.0, fraction));
        } else if (progress.phase === 'complete') {
            this._progressBar.fraction = 1.0;
            this._progressBar.visible = false;
        } else if (progress.phase === 'error') {
            this._progressBar.visible = false;
        }
    }

    _syncInstallButtons() {
        for (const [profileId, button] of this._installButtons.entries()) {
            const profile = this._profiles.find(item => item.id === profileId);
            button.sensitive = Boolean(profile && !profile.installed && !this._installing);
        }
    }

    _rebuildProfileOptions() {
        this._withSyncing(() => {
            this._selectableProfiles = installedModelProfiles(this._profiles);
            clearStringList(this._profileModel);
            this._selectableProfiles.forEach(profile =>
                this._profileModel.append(profile.title));
            const index = this._selectableProfiles.findIndex(
                profile => profile.id === this._selectedPresetId());
            this._profileRow.selected = index < 0 ? Gtk.INVALID_LIST_POSITION : index;
        });
        if (!this._profileRow)
            return;
        this._profileRow.sensitive = this._selectableProfiles.length > 0;
        this._profileRow.subtitle = this._selectableProfiles.length > 0
            ? _('Choose among installed models.')
            : _('Install a model below before selecting it.');
    }

    _selectedIndex(items, selectedId) {
        return Math.max(0, items.findIndex(item => item.id === selectedId));
    }

    _selectedProfileIndex(selectedId) {
        const index = this._selectableProfiles.findIndex(profile => profile.id === selectedId);
        return index < 0 ? Gtk.INVALID_LIST_POSITION : index;
    }

    _selectedPresetId() {
        if (this._selectedPreset)
            return this._selectedPreset;
        const profile = this._settings.get_string('model-profile');
        const family = this._settings.get_string('model-family');
        return modelPresetId(profile, family);
    }

    _updateLanguageOptions(
        family = this._settings.get_string('model-family'),
        selectedLanguage = this._settings.get_string('language')) {
        this._languages = family === 'english'
            ? this._allLanguages.filter(language => language.id.startsWith('en-'))
            : [...this._allLanguages];
        clearStringList(this._languageModel);
        this._languages.forEach(language => this._languageModel.append(language.title));
        this._languageRow.subtitle = family === 'english'
            ? _('The selected model supports English only.')
            : _('Auto is useful for multilingual dictation, but may be less accurate.');
        this._languageRow.selected = this._selectedIndex(this._languages, selectedLanguage);
    }

    _withSyncing(callback) {
        const wasSyncing = this._syncingSettings;
        this._syncingSettings = true;
        try {
            callback();
        } finally {
            this._syncingSettings = wasSyncing;
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
        this._callRemote('SetRuntimeOptions', {
            itn: new GLib.Variant('b', this._settings.get_boolean('itn')),
            endpoint_mode: new GLib.Variant('s', this._settings.get_string('endpoint-mode')),
            phrase_boosting: new GLib.Variant('b', this._settings.get_boolean('phrase-boosting')),
            boost_phrases: new GLib.Variant('s', this._settings.get_string('boost-phrases')),
            boost_tokenizer_path: new GLib.Variant('s', this._settings.get_string('boost-tokenizer-path')),
            vad_filtering: new GLib.Variant('b', this._settings.get_boolean('vad-filtering')),
            vad_model_path: new GLib.Variant('s', this._settings.get_string('vad-model-path')),
            streaming_latency_ms: new GLib.Variant('u', this._settings.get_uint('streaming-latency-ms')),
            model_root: new GLib.Variant('s', this._settings.get_string('model-root')),
            language: new GLib.Variant('s', this._settings.get_string('language')),
            worker_path: new GLib.Variant('s', this._settings.get_string('worker-path')),
            model_installer_path: new GLib.Variant('s',
                this._settings.get_string('model-installer-path')),
            num_threads: new GLib.Variant('u', this._settings.get_uint('num-threads')),
            sample_rate: new GLib.Variant('u', this._settings.get_uint('sample-rate')),
        });
    }

    _syncShortcutValue() {
        this._shortcutButton?.syncFromSettings();
    }

    _callRemote(method, ...args) {
        const callback = typeof args.at(-1) === 'function' ? args.pop() : null;
        const remote = this._proxy?.[`${method}Remote`];
        if (!remote)
            return;
        remote.call(this._proxy, ...args, (result, error) => {
            if (error) {
                this._statusRow.subtitle = formatError(error);
                logError(error, `Wordpipe ${method} failed`);
                if (method === 'SetModelProfile' || method === 'SetRuntimeOptions' || method === 'SetInsertionOptions')
                    this._refreshConfig();
                return;
            }
            if (callback)
                callback(...result);
        });
    }
});

export default class WordpipePreferences extends ExtensionPreferences {
    fillPreferencesWindow(window) {
        const generalPage = new WordpipePage(this.getSettings());
        window.add(generalPage);
        window.add(generalPage.advancedPage);
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

function clearStringList(model) {
    while (model.get_n_items() > 0)
        model.remove(0);
}

function modelPresetId(profile, family) {
    return family === 'english' ? `${profile}-english` : profile;
}

function modelPresetFromValues(values) {
    if (typeof values.model_preset === 'string')
        return values.model_preset;
    if (typeof values.model_profile === 'string')
        return modelPresetId(values.model_profile, values.model_family);
    return '';
}

function formatMetrics(metrics) {
    const rtf = numberValue(metrics.real_audio_real_time_factor ?? metrics.real_time_factor);
    const audioSeconds = numberValue(metrics.audio_seconds);
    const decodeSeconds = numberValue(metrics.decode_seconds);
    const droppedChunks = numberValue(metrics.dropped_audio_chunks);
    if (
        rtf === null &&
        audioSeconds === null &&
        decodeSeconds === null &&
        droppedChunks === null
    )
        return '';

    const parts = [];
    if (rtf !== null)
        parts.push(`RTF ${rtf.toFixed(3)}`);
    if (audioSeconds !== null)
        parts.push(`${audioSeconds.toFixed(1)}s audio`);
    if (decodeSeconds !== null)
        parts.push(`${decodeSeconds.toFixed(1)}s decode`);
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

function numberValue(value) {
    if (typeof value === 'number' && Number.isFinite(value))
        return value;
    return null;
}

function formatError(error) {
    const message = error?.message ?? String(error);
    return markupSafe(message.replace(/^GDBus\.Error:[^:]+:\s*/, ''));
}

function markupSafe(value) {
    return GLib.markup_escape_text(String(value), -1);
}

function shortcutModeTitle(mode) {
    return mode === 'push-to-talk' ? _('Push to Talk') : _('Toggle');
}
