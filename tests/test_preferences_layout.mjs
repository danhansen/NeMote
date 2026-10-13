import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL(
    '../extensions/gnome-shell/nemote@dhansen.dev/preferencesPage.js', import.meta.url), 'utf8');

function harness() {
    class Widget {
        constructor(options = {}) { Object.assign(this, options); this.children = []; this.handlers = {}; }
        add(child) { this.children.push(child); }
        add_row(child) { this.add(child); }
        add_suffix(child) { this.add(child); }
        connect(signal, callback) { this.handlers[signal] = callback; }
        set_vexpand() {}
        set_valign() {}
        update_property() {}
        set_activatable_widget(widget) { this.activatable_widget = widget; }
        append() {}
        static new(items) { return new Widget({items}); }
        static new_with_range() { return new Widget(); }
    }
    const Page = vm.runInNewContext(`(class {
        ${source.slice(source.indexOf('    _buildBehaviorGroup('), source.indexOf('    _buildTranscriptGroup('))}
    })`, {
        _: text => text,
        Adw: {PreferencesGroup: Widget, SwitchRow: Widget, EntryRow: Widget,
            ComboRow: Widget, ExpanderRow: Widget, PreferencesRow: Widget,
            ActionRow: Widget, SpinRow: Widget},
        Gtk: {StringList: Widget, TextBuffer: Widget, TextView: Widget,
            ScrolledWindow: Widget, Button: Widget, WrapMode: {WORD_CHAR: 0}, Align: {CENTER: 0},
            AccessibleProperty: {LABEL: 0}},
        ShortcutSettingButton: Widget, SHORTCUT_MODES: [], shortcutModeTitle: () => '',
    });
    const page = new Page();
    const writes = [];
    Object.assign(page, {
        _settings: {get_string: () => '', get_boolean: () => false, get_uint: () => 2,
            set_string: (...args) => writes.push(args)},
        _appendSection: (section, group) => section.add(group),
        _selectedIndex: () => 0, _pushRuntimeOptions() {},
        textPage: new Widget(), modelsPage: new Widget(), _inputGroup: new Widget(),
    });
    const general = new Widget();
    const advanced = new Widget();
    page._buildBehaviorGroup(general);
    page._buildAdvancedGroup(advanced);
    return {page, general, advanced, writes};
}

test('all choice rows display their selected value beneath the title, with wrapping', () => {
    const choices = [...source.matchAll(/new Adw\.ComboRow\(\{([\s\S]*?)\n        \}\)/g)];
    assert.equal(choices.length, 8);
    for (const [, options] of choices) {
        assert.match(options, /use_subtitle: true/);
        assert.match(options, /subtitle_lines: 0/);
        assert.doesNotMatch(options, /\n\s+subtitle:/, 'help must not replace selected text');
    }
});

test('untrusted transcript, error and progress text is displayed literally', () => {
    for (const field of ['_partialRow', '_commitRow', '_statusRow', '_progressRow', '_metricsRow']) {
        const start = source.indexOf(`this.${field} = new Adw.ActionRow({`);
        assert.ok(start >= 0);
        assert.match(source.slice(start, source.indexOf('\n        });', start)), /use_markup: false/);
    }
    const functions = vm.runInNewContext(`(() => {
        ${source.slice(source.indexOf('function formatInstallProgress('), source.indexOf('function choiceListFactory('))}
        return {formatInstallProgress, formatError};
    })()`);
    const text = '<error> & unavailable';
    assert.equal(functions.formatInstallProgress({profile: 'model', message: text}), `model: ${text}`);
    assert.equal(functions.formatError(new Error(text)), text);
});

test('text formatting and model installation are separate from dictation controls', () => {
    const {page, general} = harness();
    assert.deepEqual(Array.from(general.children, group => group.title), [
        'Keyboard Shortcut', 'Vocabulary',
    ]);
    assert.ok(page.textPage.children.some(group => group.children.includes(page._pncRow)));
    assert.ok(page.modelsPage.children.includes(page._pncGroup));
    assert.equal(page._boostPhrasesRow.children.length, 2, 'editor plus save row, no crowded suffix button');
    assert.equal(page._boostPhrasesRow.children[1].children[0].label, 'Save Phrases');
});

test('companion overrides are in Advanced and still apply the corresponding settings', () => {
    const {page, general, advanced, writes} = harness();
    const overrides = advanced.children.find(group => group.title === 'Companion Model Overrides');
    for (const [field, key] of [
        ['_pncModelPathRow', 'pnc-model-path'],
        ['_boostTokenizerPathRow', 'boost-tokenizer-path'],
        ['_vadModelPathRow', 'vad-model-path'],
    ]) {
        const row = page[field];
        assert.ok(overrides.children.includes(row));
        assert.ok(general.children.every(group => !group.children.includes(row)));
        row.text = ' /tmp/model-file ';
        row.handlers.apply(row);
        assert.deepEqual(writes.at(-1), [key, '/tmp/model-file']);
    }
});
