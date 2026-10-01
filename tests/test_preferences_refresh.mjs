import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL(
    '../extensions/gnome-shell/wordpipe@dhansen.dev/prefs.js', import.meta.url), 'utf8');

// Exercise the production callbacks without requiring a GNOME display. The
// StringList/ComboRow double emits selection changes synchronously on removal
// and on insertion into an empty list, just as GTK can during a list rebuild.
function method(name, next) {
    return source.slice(source.indexOf(`    ${name}(`), source.indexOf(`    ${next}(`));
}

function harness(kind, configured) {
    const values = {[kind === 'device' ? 'input-device' : 'backend']: configured};
    const calls = [];
    const context = vm.createContext({
        _: text => text,
        deepUnpackMap: value => value,
        clearStringList: model => {
            while (model.get_n_items())
                model.remove(0);
        },
    });
    const Page = vm.runInContext(`(class {
        ${method('_refreshBackends', '_refreshModelProfiles')}
        ${method('_refreshInputDevices', '_syncFromConfig')}
        ${method('_withSyncing', '_pushInsertionOptions')}
    })`, context);
    const page = new Page();
    page._syncingSettings = false;
    page._settings = {
        get_string: key => values[key],
        set_string: (key, value) => { values[key] = value; },
    };
    page._deviceSelectors = [''];
    page._backends = [{id: 'parakeet', title: 'Parakeet'}, {id: 'nemo-speech', title: 'NeMo'}];
    let reply = [];
    page._callRemote = (name, arg) => {
        if (name.startsWith('List'))
            arg(reply);
        else
            calls.push([name, arg]);
    };
    const prefix = kind === 'device' ? '_device' : '_backend';
    const callback = source.match(new RegExp(
        `this\\.${prefix}Row\\.connect\\('notify::selected', row => \\{([\\s\\S]*?)\\n        \\}\\);`))[1];
    const onSelected = vm.runInContext(`(function(row) {${callback}})`, context);
    let selected = kind === 'device' ? 0 : 1;
    const row = {
        get selected() { return selected; },
        set selected(value) {
            if (selected === value)
                return;
            selected = value;
            onSelected.call(page, row);
        },
    };
    const model = {
        items: kind === 'device' ? ['System Default'] : ['Parakeet', 'NeMo'],
        get_n_items() { return this.items.length; },
        remove(index) {
            this.items.splice(index, 1);
            if (!this.items.length)
                row.selected = 0xffffffff;
            else if (row.selected >= this.items.length)
                row.selected = this.items.length - 1;
        },
        append(text) {
            this.items.push(text);
            if (row.selected === 0xffffffff)
                row.selected = 0;
        },
    };
    page[`${prefix}Row`] = row;
    page[`${prefix}Model`] = model;
    page._syncComboSelections = () => page._withSyncing(() => {
        row.selected = page._backends.findIndex(item => item.id === values.backend);
    });
    return {
        page, row, calls, values,
        refresh(items) {
            reply = items;
            page[kind === 'device' ? '_refreshInputDevices' : '_refreshBackends']();
        },
    };
}

const devices = [
    {name: 'Digital Microphone', selector: '5', is_default: false},
    {name: 'MX Brio Analog Stereo', selector: '9', is_default: true},
];

for (const configured of ['', '9', 'missing-device']) {
    test(`device refresh preserves ${JSON.stringify(configured)} without configuration feedback`, () => {
        const h = harness('device', configured);
        for (let i = 0; i < 20; i++)
            h.refresh(devices);
        assert.deepEqual(h.calls, []);
        assert.equal(h.values['input-device'], configured);
        assert.equal(h.row.selected, configured === '9' ? 2 : 0);
        assert.equal(h.page._syncingSettings, false);
        // A real user change still sends exactly one request.
        h.row.selected = 1;
        assert.deepEqual(h.calls, [['SetInputDevice', '5']]);
    });
}

test('backend refresh does not switch NeMo back to Parakeet', () => {
    const h = harness('backend', 'nemo-speech');
    for (let i = 0; i < 20; i++)
        h.refresh(h.page._backends);
    assert.deepEqual(h.calls, []);
    assert.equal(h.values.backend, 'nemo-speech');
    assert.equal(h.row.selected, 1);
    h.row.selected = 0;
    assert.deepEqual(h.calls, [['SetBackend', 'parakeet']]);
});

test('device rebuild restores an enclosing sync guard', () => {
    const h = harness('device', '9');
    h.page._syncingSettings = true;
    h.refresh(devices);
    assert.equal(h.page._syncingSettings, true);
    assert.deepEqual(h.calls, []);
});
