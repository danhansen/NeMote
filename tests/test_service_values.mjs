import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

const source = readFileSync(new URL(
    '../extensions/gnome-shell/nemote@dhansen.dev/serviceValues.js', import.meta.url), 'utf8');
const {deepUnpackMap, formatInstallProgress, numberValue, formatError} =
    await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('service snapshots unpack nested variants and reject non-dictionaries', () => {
    const variant = value => ({deep_unpack: () => value});
    assert.deepEqual(deepUnpackMap(variant({itn: variant(false),
        metrics: variant({rtf: variant(0.5)}), chunks: variant([80, 160])})),
    {itn: false, metrics: {rtf: 0.5}, chunks: [80, 160]});
    for (const value of [null, undefined, [], 'text'])
        assert.deepEqual(deepUnpackMap(value), {});
});

test('progress and errors preserve literal text, not markup', () => {
    const text = '<error> & unavailable';
    assert.equal(formatInstallProgress({profile: 'model', message: text}), `model: ${text}`);
    assert.equal(formatError(new Error(text)), text);
    assert.equal(formatError(new Error(`GDBus.Error:org.example.Failed: ${text}`)), text);
    assert.equal(formatInstallProgress({phase: 'converting'}), 'converting');
    assert.equal(formatInstallProgress({profile: 'model'}), 'model');
    assert.equal(formatInstallProgress({}), '');
});

test('metrics accept finite numbers only, including zero', () => {
    assert.equal(numberValue(0), 0);
    assert.equal(numberValue(0.5), 0.5);
    for (const value of [NaN, Infinity, undefined, null, '0.5'])
        assert.equal(numberValue(value), null);
});
