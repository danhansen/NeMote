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
        add_prefix(child) { this.add(child); }
        set_group() {}
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
            ScrolledWindow: Widget, Button: Widget, Switch: Widget, CheckButton: Widget, WrapMode: {WORD_CHAR: 0}, Align: {CENTER: 0},
            AccessibleProperty: {LABEL: 0, DESCRIPTION: 1}},
        ShortcutSettingButton: Widget, SHORTCUT_MODES: [], shortcutModeTitle: () => '',
    });
    const page = new Page();
    const writes = [];
    Object.assign(page, {
        _settings: {get_string: () => '', get_boolean: () => false, get_uint: () => 2,
            set_string: (...args) => writes.push(args), set_boolean: (...args) => writes.push(args)},
        _appendSection: (section, group) => section.add(group),
        _selectedIndex: () => 0, _pushRuntimeOptions() {}, _pushInsertionOptions() {},
        textPage: new Widget(), _inputGroup: new Widget(),
    });
    const general = new Widget();
    const advanced = new Widget();
    page._buildBehaviorGroup(general);
    page._buildAdvancedGroup(advanced);
    return {page, general, advanced, writes};
}

test('choice rows use the native inline selected value', () => {
    const choices = [...source.matchAll(/new Adw\.ComboRow\(\{([\s\S]*?)\n        \}\)/g)];
    assert.equal(choices.length, 4);
    for (const [, options] of choices) {
        assert.doesNotMatch(options, /use_subtitle: true/);
        assert.doesNotMatch(options, /\n\s+subtitle:/, 'help must not replace selected text');
    }
});

test('untrusted transcript, error and progress text is displayed literally', () => {
    for (const field of ['_partialRow', '_commitRow', '_statusRow', '_progressRow', '_metricsRow']) {
        const start = source.indexOf(`this.${field} = new Adw.ActionRow({`);
        assert.ok(start >= 0);
        assert.match(source.slice(start, source.indexOf('\n        });', start)), /use_markup: false/);
    }
});

test('punctuation enablement and models share the expandable text setting', () => {
    const {page, general} = harness();
    assert.deepEqual(Array.from(general.children, group => group.title), [
        'Keyboard Shortcut', 'Vocabulary',
    ]);
    assert.ok(page.textPage.children.some(group => group.children.includes(page._pncGroup)));
    assert.ok(page._pncGroup.children.includes(page._pncRow));
    assert.equal(page._boostPhrasesRow.children.length, 2, 'editor plus save row, no crowded suffix button');
    assert.equal(page._boostPhrasesRow.children[1].children[0].label, 'Save Phrases');
});

test('radio choices write the existing settings only when activated by the user', () => {
    const {page, writes} = harness();
    for (const [choices, key] of [
        [page._endpointChoices, 'endpoint-mode'],
        [page._insertionChoices, 'insert-partials'],
    ]) {
        for (const {value, row, button} of choices) {
            assert.equal(row.activatable_widget, button, 'the whole row activates its radio button');
            button.active = true;
            button.handlers.toggled();
            assert.deepEqual(writes.at(-1), [key, value]);
            const count = writes.length;
            button.active = false;
            button.handlers.toggled();
            assert.equal(writes.length, count, 'group deselection does not overwrite the setting');
            page._syncingSettings = true;
            button.active = true;
            button.handlers.toggled();
            assert.equal(writes.length, count, 'service synchronization does not send settings back');
            page._syncingSettings = false;
        }
    }
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

test('popup hover does not move the checkmark; committed selection does', () => {
    class Widget {
        constructor(options = {}) { Object.assign(this, {visible: true}, options); this.handlers = new Map(); }
        connect(signal, callback) { this.handlers.set(signal, callback); return signal; }
        disconnect(id) { this.handlers.delete(id); }
        append() {}
        set_child() {}
    }
    const factoryFor = vm.runInNewContext(`(${source.slice(
        source.indexOf('function choiceListFactory('), source.indexOf('function shortcutModeTitle('))})`, {
        Gtk: {SignalListItemFactory: Widget, Box: Widget, Label: Widget, Image: Widget,
            Align: {CENTER: 0}, AccessibleRole: {PRESENTATION: 0}},
        Pango: {EllipsizeMode: {END: 0}},
    });
    const first = {get_string: () => 'First'};
    const second = {get_string: () => 'Second'};
    const row = new Widget({selected_item: first});
    const factory = factoryFor(row);
    const item = new Widget({item: second, selected: false});
    factory.handlers.get('setup')(factory, item);
    factory.handlers.get('bind')(factory, item);
    assert.equal(item._choiceCheck.opacity, 0);
    item.selected = true;
    item.handlers.get('notify::selected')?.(item);
    assert.equal(item._choiceCheck.opacity, 0);
    assert.equal(item._choiceCheck.visible, true, 'icon keeps its layout space');
    row.selected_item = second;
    row.handlers.get('notify::selected-item')();
    assert.equal(item._choiceCheck.opacity, 1);
    factory.handlers.get('unbind')(factory, item);
    assert.equal(row.handlers.size, 0, 'recycled items disconnect from the row');
    item.item = first;
    factory.handlers.get('bind')(factory, item);
    assert.equal(item._choiceCheck.opacity, 0);
    factory.handlers.get('unbind')(factory, item);
});
