import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

const helperUrl = new URL(
    '../extensions/gnome-shell/nemote@dhansen.dev/shortcutState.js',
    import.meta.url);
const helperSource = readFileSync(helperUrl, 'utf8');
const helperModule = await import(
    `data:text/javascript;base64,${Buffer.from(helperSource).toString('base64')}`);
const {requiredModifiersHeld} = helperModule;

const SHIFT_MASK = 1 << 0;
const CONTROL_MASK = 1 << 2;
const MOD4_MASK = 1 << 6;

test('Super-first release ends a Super chord', () => {
    assert.equal(requiredModifiersHeld(MOD4_MASK, MOD4_MASK, 0, 0), true);
    assert.equal(requiredModifiersHeld(MOD4_MASK, 0, 0, 0), false);
});

test('unrelated modifier changes do not end the chord', () => {
    assert.equal(
        requiredModifiersHeld(MOD4_MASK, MOD4_MASK | SHIFT_MASK, 0, 0),
        true);
    assert.equal(requiredModifiersHeld(MOD4_MASK, MOD4_MASK, 0, 0), true);
});

test('releasing either required modifier ends a multi-modifier chord', () => {
    const required = CONTROL_MASK | MOD4_MASK;
    assert.equal(requiredModifiersHeld(required, required, 0, 0), true);
    assert.equal(requiredModifiersHeld(required, MOD4_MASK, 0, 0), false);
    assert.equal(requiredModifiersHeld(required, CONTROL_MASK, 0, 0), false);
});

test('latched required modifiers count as active', () => {
    assert.equal(requiredModifiersHeld(MOD4_MASK, 0, MOD4_MASK, 0), true);
});
