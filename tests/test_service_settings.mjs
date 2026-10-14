import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const extension = new URL('../extensions/gnome-shell/nemote@dhansen.dev/', import.meta.url);
const source = readFileSync(new URL('serviceSettings.js', extension), 'utf8');
const {INSERTION_SETTINGS, RUNTIME_SETTINGS, optionsFromSettings, syncSettingsFromConfig} =
    await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const schema = readFileSync(new URL('schemas/org.gnome.shell.extensions.nemote.gschema.xml', extension), 'utf8');
const types = Object.fromEntries([...schema.matchAll(/<key name="([^"]+)" type="([^"]+)"/g)]
    .map(([, name, type]) => [name, type]));

function fixture() {
    const writes = new Map();
    const variants = Object.fromEntries(Object.entries(types).map(([key, type]) =>
        [key, {get_type_string: () => type}]));
    const settings = {get_value: key => variants[key]};
    for (const [setter, type] of Object.entries({string: 's', boolean: 'b', uint: 'u', strv: 'as'})) {
        settings[`set_${setter}`] = (key, value) => {
            assert.equal(types[key], type, `schema type for ${key}`);
            writes.set(key, value);
        };
    }
    return {settings, variants, writes};
}

test('option groups are disjoint and reuse schema-typed variants', () => {
    const {settings, variants} = fixture();
    const keys = [...INSERTION_SETTINGS, ...RUNTIME_SETTINGS];
    assert.equal(new Set(keys).size, keys.length);
    for (const group of [INSERTION_SETTINGS, RUNTIME_SETTINGS]) {
        const options = optionsFromSettings(settings, group);
        for (const key of group) {
            assert.ok(variants[key], `known schema key ${key}`);
            assert.equal(options[key.replaceAll('-', '_')], variants[key]);
        }
    }
    assert.ok(!keys.includes('show-overlay'));
    assert.ok(!keys.includes('backend'));
});

test('all service options synchronize with schema types, including false and zero', () => {
    const {settings, writes} = fixture();
    const keys = ['backend', 'model-profile', 'model-family', 'input-device',
        ...INSERTION_SETTINGS, ...RUNTIME_SETTINGS];
    const values = {s: 'example', b: false, u: 0};
    const config = Object.fromEntries(keys.map(key => [key.replaceAll('-', '_'), values[types[key]]]));
    syncSettingsFromConfig(settings, {...config, shortcut: '<Control>space'});
    for (const key of keys)
        assert.equal(writes.get(key), values[types[key]], key);
    assert.equal(writes.has('toggle-shortcut'), false);
});

test('missing, invalid and retired fields do not overwrite local settings', () => {
    const {settings, writes} = fixture();
    syncSettingsFromConfig(settings, {itn: 'true', num_threads: '2', language: false,
        show_overlay: true, unknown: 'ignored'});
    assert.equal(writes.size, 0);
    syncSettingsFromConfig(settings, {shortcut: ''});
    assert.equal(writes.size, 0);
});

test('Shell change routing uses the same option groups as serialization', () => {
    const shell = readFileSync(new URL('extension.js', extension), 'utf8');
    const method = shell.slice(shell.indexOf('    _pushSetting('), shell.indexOf('    _pushInsertionOptions('));
    const Client = vm.runInNewContext(`(class { ${method} })`, {INSERTION_SETTINGS, RUNTIME_SETTINGS});
    const client = new Client();
    const calls = [];
    Object.assign(client, {_proxy: {},
        _pushInsertionOptions: () => calls.push('insertion'),
        _pushRuntimeOptions: () => calls.push('runtime'),
    });
    for (const [group, expected] of [[INSERTION_SETTINGS, 'insertion'], [RUNTIME_SETTINGS, 'runtime']]) {
        for (const key of group) {
            client._pushSetting(key);
            assert.equal(calls.pop(), expected, key);
        }
    }
    for (const key of ['backend', 'model-profile', 'model-family', 'show-overlay', 'unknown'])
        client._pushSetting(key);
    assert.equal(calls.length, 0);
});
