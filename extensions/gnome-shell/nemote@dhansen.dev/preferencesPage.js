import Adw from 'gi://Adw';
import Gio from 'gi://Gio';
import GObject from 'gi://GObject';
import Gtk from 'gi://Gtk';
import Pango from 'gi://Pango';

// The GNOME host supplies its translator; standalone previews use English.
let _ = text => text;

export function createPreferencesPage(settings, {gettext = text => text, connectService = true} = {}) {
    _ = gettext;
    return new NeMotePage(settings, connectService);
}

import {streamingChunkChoices} from './modelState.js';

import {BUS_NAME, OBJECT_PATH, NeMoteProxy} from './serviceClient.js';
import {deepUnpackMap, formatInstallProgress, numberValue, formatError} from './serviceValues.js';
import {INSERTION_SETTINGS, RUNTIME_SETTINGS, optionsFromSettings, syncSettingsFromConfig} from './serviceSettings.js';

const PROFILES = [
    ['nemo-q8-english', 'English (lower WER)', 'Dedicated English Nemotron model.', 'nemo-q8', 'english'],
    ['nemo-q8', 'Multilingual', 'Multilingual Nemotron model.', 'nemo-q8', 'multilingual'],
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

const NeMotePage = GObject.registerClass(
class NeMotePage extends Adw.PreferencesPage {
    constructor(settings, connectService = true) {
        super({
            title: _('Dictation'),
            icon_name: 'audio-input-microphone-symbolic',
        });
        this._settings = settings;
        this._proxy = null;
        this._signalIds = [];
        this._syncingSettings = false;
        this._selectedPreset = modelPresetId(
            this._settings.get_string('model-profile'),
            this._settings.get_string('model-family'));
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
        this._pncModels = [];
        this._deviceSelectors = [''];
        this._installRows = new Map();
        this._installButtons = new Map();
        this._installChecks = new Map();
        this._installing = false;
        this._installingProfile = '';
        this._lastInstallProgress = {};
        this._runtimeBusy = false;
        this.textPage = new Adw.PreferencesPage({
            title: _('Text'),
            icon_name: 'insert-text-symbolic',
        });
        this.advancedPage = new Adw.PreferencesPage({
            title: _('Advanced'),
            icon_name: 'preferences-other-symbolic',
        });
        this.pages = [this, this.textPage, this.advancedPage];
        this._buildGeneralSection();
        this._buildAdvancedSection();
        if (connectService)
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
        this._buildInputGroup(this);
        this._buildModelGroup(this);
        this._buildBehaviorGroup(this);
        for (const row of [this._latencyRow, this._languageRow,
            this._deviceRow, this._shortcutModeRow])
            row.list_factory = choiceListFactory(row);
        this._syncModelChoices();
    }

    _buildAdvancedSection() {
        this._buildAdvancedGroup(this.advancedPage);
        this._buildTranscriptGroup(this.advancedPage);
        this._buildServiceGroup(this.advancedPage);
        this._runtimeControls = [this._deviceRow, this._latencyRow, this._languageRow,
            this._itnRow, this._pncRow,
            ...this._endpointChoices.map(choice => choice.row),
            this._phraseBoostingRow, this._boostPhrasesRow, this._vadFilteringRow,
            this._modelRootRow, this._workerPathRow, this._modelInstallerPathRow,
            this._threadsRow, this._pncModelPathRow, this._boostTokenizerPathRow,
            this._vadModelPathRow, ...this._insertionChoices.map(choice => choice.row)];
    }

    _appendSection(section, widget) {
        section.add(widget);
    }

    _buildModelGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Speech Recognition'),
            description: _('Smaller chunks show text sooner but use more processing power'),
        });
        this._appendSection(section, group);
        this._modelGroup = new Adw.ExpanderRow({
            title: _('Model'),
            subtitle: _('Choose or install a speech model'),
            use_markup: false,
        });
        group.add(this._modelGroup);

        this._latencyValues = [80, 160, 560, 1120];
        this._latencyRow = new Adw.ComboRow({
            title: _('Streaming Chunk Size'),
            model: Gtk.StringList.new(this._latencyValues.map(value => `${value} ms`)),
            selected: Math.max(0, this._latencyValues.indexOf(this._settings.get_uint('streaming-latency-ms'))),
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
        group.add(this._languageRow);
        group.add(this._latencyRow);

        this._rebuildProfileRows();
    }

    _buildInputGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Audio'),
        });
        this._appendSection(section, group);

        this._deviceModel = new Gtk.StringList();
        this._deviceModel.append(_('System Default'));
        this._deviceRow = new Adw.ComboRow({
            title: _('Microphone'),
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
        this._inputGroup = group;
    }

    _buildBehaviorGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Formatting'),
            description: _('Automatic formatting is applied when dictation stops, or at pauses if enabled below'),
        });
        this.textPage.add(group);

        let row = new Adw.SwitchRow({
            title: _('Spoken Punctuation'),
            subtitle: _('Turn spoken commands such as “comma” into punctuation'),
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
            title: _('Format Numbers and Dates'),
            subtitle: _('Write “twenty dollars” as “$20”, for example'),
            tooltip_text: _('Inverse text normalization supports English, Arabic, Chinese, French, German, Hindi and Spanish'),
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
        this._pncGroup = new Adw.ExpanderRow({
            title: _('Restore Punctuation and Capitalization'),
            subtitle: _('Experimental · English only'),
            subtitle_lines: 0,
            use_markup: false,
        });
        // Keep expansion independent of enablement so installation is available
        // while punctuation restoration is off.
        this._pncRow = new Gtk.Switch({
            active: this._settings.get_boolean('pnc'),
            valign: Gtk.Align.CENTER,
        });
        this._pncRow.update_property([Gtk.AccessibleProperty.LABEL],
            [_('Restore Punctuation and Capitalization')]);
        this._pncRow.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('pnc', widget.active);
            this._syncModelChoices();
            this._pushRuntimeOptions();
        });
        this._pncGroup.add_suffix(this._pncRow);
        group.add(this._pncGroup);
        const endpointGroup = new Adw.PreferencesGroup({
            title: _('Pauses'),
            description: _('Choose what happens after about 0.8 seconds of silence'),
        });
        this.textPage.add(endpointGroup);
        this._endpointChoices = this._addRadioChoices(endpointGroup, [
            ['disabled', _('Do nothing'),
                _('Keep dictating through pauses. Apply automatic formatting and insert final text only when dictation stops.')],
            ['preview', _('Format the preview'),
                _('Apply enabled formatting to all text so far, without finalizing it. Requires live preview to see updates before dictation stops.')],
            ['commit', _('Format and insert final text'),
                _('Apply enabled formatting and finalize the text at each pause. Continue dictating a new segment without stopping the microphone.')],
        ], this._settings.get_string('endpoint-mode'), mode => {
            this._settings.set_string('endpoint-mode', mode);
            this._pushRuntimeOptions();
        });

        const vocabularyGroup = new Adw.PreferencesGroup({title: _('Vocabulary')});

        this._phraseBoostingRow = new Adw.SwitchRow({
            title: _('Favor Custom Vocabulary'),
            subtitle: _('Prefer these names and terms, even when other words sound similar'),
            active: this._settings.get_boolean('phrase-boosting'),
        });
        this._phraseBoostingRow.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('phrase-boosting', widget.active);
            this._pushRuntimeOptions();
        });
        vocabularyGroup.add(this._phraseBoostingRow);
        this._boostPhrasesRow = new Adw.ExpanderRow({
            title: _('Names and Phrases'),
            subtitle: _('One phrase per line. Save changes before dictating.'),
        });
        this._boostPhrasesBuffer = new Gtk.TextBuffer({text: this._settings.get_string('boost-phrases')});
        const editor = new Gtk.TextView({
            buffer: this._boostPhrasesBuffer,
            wrap_mode: Gtk.WrapMode.WORD_CHAR,
            top_margin: 8, bottom_margin: 8, left_margin: 12, right_margin: 12,
        });
        editor.update_property([Gtk.AccessibleProperty.LABEL], [_('Names and Phrases')]);
        const scroll = new Gtk.ScrolledWindow({min_content_height: 120, child: editor});
        this._boostPhrasesRow.add_row(new Adw.PreferencesRow({child: scroll}));
        const savePhrases = new Gtk.Button({label: _('Save Phrases'), valign: Gtk.Align.CENTER});
        savePhrases.connect('clicked', () => {
            const buffer = this._boostPhrasesBuffer;
            this._settings.set_string('boost-phrases', buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), false));
            this._pushRuntimeOptions();
        });
        const saveRow = new Adw.ActionRow({title: _('Vocabulary Changes')});
        saveRow.add_suffix(savePhrases);
        saveRow.activatable_widget = savePhrases;
        this._boostPhrasesRow.add_row(saveRow);
        vocabularyGroup.add(this._boostPhrasesRow);
        this._vadFilteringRow = new Adw.SwitchRow({
            title: _('Filter Non-Speech Audio'),
            subtitle: _('Download a small voice activity model automatically when first used'),
            active: this._settings.get_boolean('vad-filtering'),
        });
        this._vadFilteringRow.connect('notify::active', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_boolean('vad-filtering', widget.active);
            this._pushRuntimeOptions();
        });
        this._inputGroup.add(this._vadFilteringRow);

        const insertionGroup = new Adw.PreferencesGroup({
            title: _('Text Insertion'),
        });
        this.textPage.add(insertionGroup);

        this._insertionChoices = this._addRadioChoices(insertionGroup, [
            [true, _('Live preview and final text'),
                _('Show replaceable text in the focused application while speaking. Insert final text when dictation stops or a pause finalizes it.')],
            [false, _('Final text only'),
                _('Show no text while speaking. Insert text when dictation stops, or at each pause if “Format and insert final text” is selected.')],
        ], this._settings.get_boolean('insert-partials'), value => {
            this._settings.set_boolean('insert-partials', value);
            this._pushInsertionOptions();
        });

        this._delayRow = Adw.SpinRow.new_with_range(0, 1000, 25);
        this._delayRow.title = _('Insertion Delay');
        this._delayRow.subtitle = _('Additional delay before inserting text, in milliseconds');
        this._delayRow.value = this._settings.get_uint('stream-insert-delay-ms');
        this._delayRow.connect('notify::value', widget => {
            if (this._syncingSettings)
                return;
            this._settings.set_uint('stream-insert-delay-ms', Math.round(widget.value));
            this._pushInsertionOptions();
        });
        insertionGroup.add(this._delayRow);

        const shortcutGroup = new Adw.PreferencesGroup({
            title: _('Keyboard Shortcut'),
            description: _('Start and stop dictation from any application'),
        });
        this._appendSection(section, shortcutGroup);

        const shortcutButton = new Gtk.Button({
            label: _('Configure…'),
            valign: Gtk.Align.CENTER,
            sensitive: false,
        });
        shortcutButton.connect('clicked', () => this._callRemote('ConfigureShortcut'));
        const shortcutRow = new Adw.ActionRow({
            title: _('Shortcut'),
            subtitle: _('Managed by GNOME Settings'),
            subtitle_lines: 0,
            use_markup: false,
        });
        shortcutRow.add_suffix(shortcutButton);
        shortcutRow.set_activatable_widget(shortcutButton);
        this._shortcutButton = shortcutButton;
        this._shortcutRow = shortcutRow;
        shortcutGroup.add(shortcutRow);

        this._shortcutModeModel = new Gtk.StringList();
        SHORTCUT_MODES.forEach(mode =>
            this._shortcutModeModel.append(shortcutModeTitle(mode.id)));
        this._shortcutModeRow = new Adw.ComboRow({
            title: _('Shortcut Behavior'),
            model: this._shortcutModeModel,
        });
        this._shortcutModeRow.selected = this._selectedIndex(
            SHORTCUT_MODES,
            this._settings.get_string('shortcut-mode'));
        this._shortcutModeRow.connect('notify::selected', row => {
            if (this._syncingSettings)
                return;
            const mode = SHORTCUT_MODES[row.selected]?.id;
            if (mode) {
                this._settings.set_string('shortcut-mode', mode);
                this._callRemote('SetShortcutMode', mode);
            }
        });
        shortcutGroup.add(this._shortcutModeRow);
        this._appendSection(section, vocabularyGroup);
    }

    _addRadioChoices(group, choices, selected, onSelected) {
        let firstButton = null;
        return choices.map(([value, title, subtitle]) => {
            const button = new Gtk.CheckButton({valign: Gtk.Align.CENTER});
            if (firstButton)
                button.set_group(firstButton);
            else
                firstButton = button;
            const row = new Adw.ActionRow({title, subtitle, subtitle_lines: 0,
                activatable_widget: button, use_markup: false});
            row.add_prefix(button);
            button.update_property([Gtk.AccessibleProperty.LABEL, Gtk.AccessibleProperty.DESCRIPTION],
                [title, subtitle]);
            button.active = value === selected;
            button.connect('toggled', () => {
                if (button.active && !this._syncingSettings)
                    onSelected(value);
            });
            group.add(row);
            return {value, row, button};
        });
    }

    _buildAdvancedGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Runtime'),
            description: _('Custom paths and processing options. Stop dictation before making changes.'),
        });
        this._appendSection(section, group);

        this._modelRootRow = new Adw.EntryRow({
            title: _('Model Directory'),
            show_apply_button: true,
            text: this._settings.get_string('model-root'),
        });
        this._modelRootRow.connect('apply', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_string('model-root', row.text.trim());
            this._pushRuntimeOptions();
        });
        group.add(this._modelRootRow);

        this._workerPathRow = new Adw.EntryRow({
            title: _('Worker Binary'),
            show_apply_button: true,
            text: this._settings.get_string('worker-path'),
        });
        this._workerPathRow.connect('apply', row => {
            if (this._syncingSettings)
                return;
            this._settings.set_string('worker-path', row.text.trim());
            this._pushRuntimeOptions();
        });
        group.add(this._workerPathRow);

        this._modelInstallerPathRow = new Adw.EntryRow({
            title: _('Model Installer'),
            show_apply_button: true,
            text: this._settings.get_string('model-installer-path'),
        });
        this._modelInstallerPathRow.connect('apply', row => {
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

        const overrides = new Adw.PreferencesGroup({
            title: _('Companion Model Overrides'),
            description: _('Optional local files that replace the managed models. Leave empty to use the defaults.'),
        });
        this._appendSection(section, overrides);
        for (const [field, key, title] of [
            ['_pncModelPathRow', 'pnc-model-path', _('Punctuation Model (.gguf)')],
            ['_boostTokenizerPathRow', 'boost-tokenizer-path', _('Boosting Tokenizer')],
            ['_vadModelPathRow', 'vad-model-path', _('Silero VAD Model (.gguf)')],
        ]) {
            const row = new Adw.EntryRow({
                title,
                text: this._settings.get_string(key),
                show_apply_button: true,
            });
            row.connect('apply', widget => {
                if (this._syncingSettings)
                    return;
                this._settings.set_string(key, widget.text.trim());
                this._pushRuntimeOptions();
            });
            this[field] = row;
            overrides.add(row);
        }

    }

    _buildTranscriptGroup(section) {
        const group = new Adw.PreferencesGroup({
            title: _('Transcript'),
        });
        this._appendSection(section, group);

        this._partialRow = new Adw.ActionRow({
            title: _('Live Transcript'),
            subtitle: _('No speech yet'),
            use_markup: false,
            subtitle_selectable: true,
        });
        group.add(this._partialRow);

        this._commitRow = new Adw.ActionRow({
            title: _('Last Committed'),
            subtitle: _('Nothing committed'),
            use_markup: false,
            subtitle_selectable: true,
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
            use_markup: false,
            subtitle_selectable: true,
        });
        group.add(this._statusRow);

        this._progressRow = new Adw.ActionRow({
            title: _('Installation Status'),
            subtitle: _('Idle'),
            use_markup: false,
        });
        this._progressBar = new Gtk.ProgressBar({
            valign: Gtk.Align.CENTER,
            margin_start: 14, margin_end: 14, margin_top: 10, margin_bottom: 10,
            show_text: true,
            visible: false,
        });
        group.add(this._progressRow);
        const progressIndicator = new Adw.PreferencesRow({child: this._progressBar});
        this._progressBar.bind_property('visible', progressIndicator, 'visible', GObject.BindingFlags.SYNC_CREATE);
        group.add(progressIndicator);

        this._metricsRow = new Adw.ActionRow({
            title: _('Runtime Metrics'),
            subtitle: _('No metrics yet'),
            use_markup: false,
        });
        group.add(this._metricsRow);

        const actions = new Adw.ActionRow({
            title: _('Dictation'),
        });
        this._startButton = new Gtk.Button({
            label: _('Start'),
            valign: Gtk.Align.CENTER,
            tooltip_text: _('Start dictation'),
            sensitive: false,
        });
        this._startButton.connect('clicked', () => this._callRemote('Start'));
        this._stopButton = new Gtk.Button({
            label: _('Stop'),
            valign: Gtk.Align.CENTER,
            tooltip_text: _('Stop dictation'),
            sensitive: false,
        });
        this._stopButton.connect('clicked', () => this._callRemote('Stop'));
        actions.add_suffix(this._startButton);
        actions.add_suffix(this._stopButton);
        group.add(actions);
    }

    _connectProxy() {
        this._proxy = new NeMoteProxy(
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

    _refreshModelProfiles() {
        this._refreshPncModels();
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
            this._rebuildProfileRows();
            this._syncComboSelections();
        });
    }

    _refreshPncModels() {
        this._callRemote('ListPncModels', models => {
            this._pncModels = models.map(item => deepUnpackMap(item))
                .filter(item => typeof item.id === 'string');
            this._rebuildProfileRows();
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
            syncSettingsFromConfig(this._settings, values);

            this._syncComboSelections();
            this._syncDeviceSelection();
            this._syncControlValues();
        } finally {
            this._syncingSettings = false;
        }
    }

    _syncComboSelections() {
        if (!this._languageRow)
            return;
        const family = this._settings.get_string('model-family');
        const language = this._settings.get_string('language');
        this._withSyncing(() => {
            this._updateLanguageOptions(family, language);
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
        this._insertionChoices.find(choice => choice.value === this._settings.get_boolean('insert-partials')).button.active = true;
        this._itnRow.active = this._settings.get_boolean('itn');
        this._pncRow.active = this._settings.get_boolean('pnc');
        this._pncModelPathRow.text = this._settings.get_string('pnc-model-path');
        this._syncModelChoices();
        (this._endpointChoices.find(choice => choice.value === this._settings.get_string('endpoint-mode')) ??
            this._endpointChoices[0]).button.active = true;
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
        this._latencyRow.selected = Math.max(0, this._latencyValues.indexOf(this._settings.get_uint('streaming-latency-ms')));
        this._shortcutModeRow.selected = this._selectedIndex(
            SHORTCUT_MODES,
            this._settings.get_string('shortcut-mode'));
    }

    _rebuildProfileRows() {
        for (const [id, row] of this._installRows.entries())
            (id.startsWith('pnc-') ? this._pncGroup : this._modelGroup).remove(row);
        this._installRows.clear();
        this._installButtons.clear();
        this._installChecks.clear();

        for (const profile of [...this._profiles, ...this._pncModels]) {
            const installing = this._installing && this._installingProfile === profile.id;
            const row = new Adw.ActionRow({
                title: profile.title,
                subtitle: this._profileSubtitle(profile),
                use_markup: false,
            });
            const check = new Gtk.Image({icon_name: 'object-select-symbolic',
                accessible_role: Gtk.AccessibleRole.PRESENTATION, opacity: 0});
            row.add_prefix(check);
            row.connect('activated', () => this._selectModel(profile.id));
            const button = new Gtk.Button({
                icon_name: installing ? 'emblem-synchronizing-symbolic' : 'folder-download-symbolic',
                valign: Gtk.Align.CENTER,
                // Installation status is already shown in the subtitle; a
                // disabled confirmation button is almost invisible in light themes.
                visible: !profile.installed,
                sensitive: !profile.installed && !this._runtimeBusy,
                tooltip_text: profile.installed
                    ? _(`${profile.title} is installed`)
                    : installing
                        ? _(`${profile.title} is installing`)
                    : _(`Download and prepare the ${profile.title} model`),
            });
            button.connect('clicked', () => {
                if (profile.installed || this._runtimeBusy)
                    return;
                this._progressRow.subtitle = _(`Starting ${profile.title}`);
                this._callRemote('InstallModel', profile.id);
            });
            row.add_suffix(button);
            this._installRows.set(profile.id, row);
            this._installButtons.set(profile.id, button);
            this._installChecks.set(profile.id, check);
            (profile.id.startsWith('pnc-') ? this._pncGroup : this._modelGroup).add_row(row);
        }
        this._syncModelChoices();
    }

    _selectModel(id) {
        const profile = [...this._profiles, ...this._pncModels].find(item => item.id === id);
        if (!profile?.installed || this._runtimeBusy)
            return;
        if (id.startsWith('pnc-')) {
            if (this._settings.get_string('pnc-model-path'))
                return;
            this._settings.set_string('pnc-variant', id.slice(4));
            this._syncModelChoices();
            this._pushRuntimeOptions();
        } else {
            this._selectedPreset = id;
            this._withSyncing(() => {
                this._settings.set_string('model-profile', profile.model_profile);
                this._settings.set_string('model-family', profile.model_family);
            });
            this._syncModelChoices();
            this._syncComboSelections();
            this._callRemote('SetModelProfile', id);
        }
    }

    _syncModelChoices() {
        const customPnc = this._settings.get_string('pnc-model-path');
        const selectedPnc = `pnc-${this._settings.get_string('pnc-variant')}`;
        const selectedAsr = this._selectedPresetId();
        for (const profile of [...this._profiles, ...this._pncModels]) {
            const row = this._installRows.get(profile.id);
            if (!row)
                continue;
            const pnc = profile.id.startsWith('pnc-');
            const selected = profile.installed && (pnc ? !customPnc && profile.id === selectedPnc : profile.id === selectedAsr);
            row.activatable = profile.installed && !this._runtimeBusy && !(pnc && customPnc);
            // Leave the row sensitive: its independent download action must
            // remain usable even when the model cannot yet be selected.
            this._installChecks.get(profile.id).opacity = selected ? 1 : 0;
            row.update_property([Gtk.AccessibleProperty.DESCRIPTION],
                [selected ? _('Selected model') : profile.installed ? _('Installed model') : _('Not installed')]);
        }
        const asr = this._profiles.find(profile => profile.id === selectedAsr);
        this._modelGroup.subtitle = asr?.installed ? asr.title : _('Select or download a speech model');
        if (!this._pncGroup)
            return;
        const pnc = this._pncModels.find(profile => profile.id === selectedPnc);
        const ready = Boolean(customPnc || pnc?.installed);
        this._pncGroup.subtitle = customPnc
            ? _('Experimental · English only · Custom model from Advanced')
            : pnc?.installed ? `${_('Experimental · English only')} · ${pnc.title}`
            : _('Experimental · English only · Select or download a model');
        this._pncRow.sensitive = !this._runtimeBusy && (ready || this._pncRow.active);
        this._pncRow.tooltip_text = ready ? _('Enable Punctuation Restoration') : _('Select an installed model before enabling');
    }

    _profileSubtitle(profile) {
        if (this._lastInstallProgress?.profile === profile.id &&
            ((this._installing && this._installingProfile === profile.id) || this._lastInstallProgress.phase === 'error'))
            return formatInstallProgress(this._lastInstallProgress) || _('Installing');
        if (this._installing && this._installingProfile === profile.id)
            return _('Installing');
        const status = profile.installed ? _('Installed') : _('Not installed');
        const detail = profile.description || profile.runtime_dir;
        return detail ? `${status} - ${detail}` : status;
    }

    _handleState(values) {
        if (this._shortcutRow) {
            this._shortcutRow.subtitle = values.shortcut_error ||
                (values.shortcut_status === 'connecting' ? _('Connecting to desktop shortcuts…') :
                    values.shortcut_status === 'configuring' ? _('Finish shortcut setup in GNOME Settings') :
                    values.shortcut_description || _('Not configured — choose a shortcut in GNOME Settings'));
            this._shortcutButton.sensitive =
                !['connecting', 'configuring'].includes(values.shortcut_status);
        }
        const selectedPreset = modelPresetFromValues(values);
        if (selectedPreset && selectedPreset !== this._selectedPreset) {
            this._selectedPreset = selectedPreset;
        }
        const previousInstalling = this._installing;
        const previousInstallingProfile = this._installingProfile;
        this._installing = Boolean(values.installing);
        this._installingProfile = values.installing_profile ?? '';
        this._runtimeBusy = Boolean(values.loading_model || values.listening || values.stopping || values.installing);
        for (const control of this._runtimeControls ?? [])
            control.sensitive = !this._runtimeBusy;
        this._syncModelChoices();
        this._latencyRow.sensitive = !values.loading_model && !values.listening &&
            !values.stopping && !values.installing;
        const selectedModelInstalled = values.selected_model_installed !== false;
        const selectedPncInstalled = values.selected_pnc_model_installed !== false;
        if (values.loading_model)
            this._statusRow.subtitle = values.loading_message || _('Loading model');
        else if (values.listening)
            this._statusRow.subtitle = _('Listening');
        else if (values.stopping)
            this._statusRow.subtitle = _('Stopping');
        else if (values.installing)
            this._statusRow.subtitle = _('Installing model');
        else if (!selectedModelInstalled)
            this._statusRow.subtitle = _('Model missing');
        else if (!selectedPncInstalled)
            this._statusRow.subtitle = _('Install the selected punctuation model under Text → Restore Punctuation and Capitalization, or disable restoration');
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
                selectedModelInstalled && selectedPncInstalled;
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
        this._lastInstallProgress = progress;
        const summary = formatInstallProgress(progress);
        if (summary)
            this._progressRow.subtitle = summary;
        const installRow = this._installRows.get(progress.profile);
        if (summary && installRow)
            installRow.subtitle = summary;
        const fraction = numberValue(progress.fraction);
        const active = progress.phase !== 'complete' && progress.phase !== 'error';
        if (fraction !== null && active) {
            this._progressBar.visible = true;
            this._progressBar.fraction = Math.max(0.0, Math.min(1.0, fraction));
        } else if (progress.phase === 'complete') {
            this._progressBar.fraction = 1.0;
            this._progressBar.visible = false;
        } else {
            this._progressBar.visible = false;
        }
    }

    _syncInstallButtons() {
        for (const [profileId, button] of this._installButtons.entries()) {
            const profile = [...this._profiles, ...this._pncModels].find(item => item.id === profileId);
            button.sensitive = Boolean(profile && !profile.installed && !this._runtimeBusy);
        }
    }

    _selectedIndex(items, selectedId) {
        return Math.max(0, items.findIndex(item => item.id === selectedId));
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
        this._languageRow.tooltip_text = family === 'english'
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
        this._callRemote('SetInsertionOptions', optionsFromSettings(this._settings, INSERTION_SETTINGS));
    }

    _pushRuntimeOptions() {
        this._callRemote('SetRuntimeOptions', optionsFromSettings(this._settings, RUNTIME_SETTINGS));
    }

    _callRemote(method, ...args) {
        const callback = typeof args.at(-1) === 'function' ? args.pop() : null;
        const remote = this._proxy?.[`${method}Remote`];
        if (!remote)
            return;
        remote.call(this._proxy, ...args, (result, error) => {
            if (error) {
                this._statusRow.subtitle = formatError(error);
                logError(error, `NeMote ${method} failed`);
                if (method === 'SetModelProfile' || method === 'SetRuntimeOptions' || method === 'SetInsertionOptions')
                    this._refreshConfig();
                return;
            }
            if (callback)
                callback(...result);
        });
    }
});

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

// Follow GNOME Settings' cc-device-combo-row.c: let the popup request the
// full label width, and reserve the checkmark's space independently of hover.
// https://gitlab.gnome.org/GNOME/gnome-control-center/-/blob/main/panels/sound/cc-device-combo-row.c
function choiceListFactory(row) {
    const factory = new Gtk.SignalListItemFactory();
    factory.connect('setup', (_factory, item) => {
        const box = new Gtk.Box();
        item._choiceLabel = new Gtk.Label({xalign: 0, width_chars: 1,
            ellipsize: Pango.EllipsizeMode.END, valign: Gtk.Align.CENTER,
            use_markup: false});
        item._choiceCheck = new Gtk.Image({icon_name: 'object-select-symbolic',
            accessible_role: Gtk.AccessibleRole.PRESENTATION});
        box.append(item._choiceLabel);
        box.append(item._choiceCheck);
        item.set_child(box);
    });
    factory.connect('bind', (_factory, item) => {
        item._choiceLabel.label = item.item.get_string();
        const update = () => {
            item._choiceCheck.opacity = row.selected_item === item.item ? 1 : 0;
        };
        item._choiceSelectedId = row.connect('notify::selected-item', update);
        update();
    });
    factory.connect('unbind', (_factory, item) => {
        if (item._choiceSelectedId) {
            row.disconnect(item._choiceSelectedId);
            item._choiceSelectedId = 0;
        }
    });
    return factory;
}

function shortcutModeTitle(mode) {
    return mode === 'push-to-talk' ? _('Push to Talk') : _('Toggle');
}
