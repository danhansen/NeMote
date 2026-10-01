import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../extensions/gnome-shell/wordpipe@dhansen.dev/extension.js', import.meta.url), 'utf8');
const injectorSource = source.slice(source.indexOf('class TextInjector {'), source.indexOf('export default class WordpipeExtension'));

function fixture({preedit = true} = {}) {
    const calls = [];
    const timers = new Map();
    let nextTimer = 0;
    const method = {
        _currentFocus: {},
        can_show_preedit: preedit,
        set_preedit_text: (...args) => calls.push(['preedit', ...args]),
        commit: text => calls.push(['commit', text]),
    };
    const display = {focus_window: {}};
    const context = vm.createContext({
        log: () => {},
        global: {display},
        Clutter: {PreeditResetMode: {CLEAR: 0}, get_default_backend: () => ({get_input_method: () => method})},
        GLib: {PRIORITY_DEFAULT: 0, SOURCE_REMOVE: false,
            timeout_add: (_priority, _delay, callback) => { timers.set(++nextTimer, callback); return nextTimer; },
            Source: {remove: id => timers.delete(id)}},
    });
    const Injector = vm.runInContext(`(${injectorSource})`, context);
    const injector = new Injector();
    injector.reset(1);
    return {injector, calls, timers, method, display,
        flush: () => { for (const [id, callback] of [...timers]) { timers.delete(id); callback(); } }};
}

test('partials are replaceable preedit; ITN final is committed once', () => {
    const {injector, calls} = fixture();
    injector.insertPartial(1, 1, 'twenty');
    injector.insertPartial(1, 2, 'twenty five');
    injector.insertCommit(1, 3, '25');
    injector.insertCommit(1, 4, '25');
    assert.deepEqual(calls, [
        ['preedit', 'twenty', 6, 6, 0], ['preedit', 'twenty five', 11, 11, 0],
        ['preedit', null, 0, 0, 0], ['commit', '25'],
    ]);
});

test('committed-only has no preedit and accepts complete final', () => {
    const {injector, calls} = fixture();
    injector.insertCommit(1, 2, 'Final text.');
    assert.deepEqual(calls, [['commit', 'Final text.']]);
});

test('final cancels queued partials; stale events cannot restart a session', () => {
    const {injector, calls, flush, timers} = fixture();
    injector.insertPartial(1, 1, 'provisional', 100);
    injector.insertCommit(1, 2, 'final');
    injector.cancel(); // SessionStopped, not an intermediate endpoint commit.
    flush();
    injector.insertPartial(1, 3, 'late');
    injector.insertPartial(99, 4, 'wrong session');
    assert.equal(timers.size, 0);
    assert.deepEqual(calls, [['commit', 'final']]);
});

test('newer delayed snapshots replace older ones', () => {
    const {injector, calls, flush} = fixture();
    injector.insertPartial(1, 1, 'old', 100);
    injector.insertPartial(1, 2, 'new', 100);
    injector.insertPartial(1, 1, 'out of order');
    flush();
    assert.deepEqual(calls, [['preedit', 'new', 3, 3, 0]]);
});

test('cursor counts Unicode characters rather than UTF-16 units', () => {
    const {injector, calls} = fixture();
    injector.insertPartial(1, 1, 'a🙂');
    assert.deepEqual(calls, [['preedit', 'a🙂', 2, 2, 0]]);
});

test('unsupported preedit safely falls back to committed-only', () => {
    const {injector, calls} = fixture({preedit: false});
    injector.insertPartial(1, 1, 'one dollar');
    injector.insertCommit(1, 2, '$1');
    assert.deepEqual(calls, [['commit', '$1']]);
});

for (const target of ['window', 'field']) {
    test(`focus change to another ${target} cancels insertion`, () => {
        const {injector, calls, display, method, flush} = fixture();
        injector.insertPartial(1, 1, 'pending', 100);
        if (target === 'window') display.focus_window = {};
        else method._currentFocus = {};
        flush();
        injector.insertCommit(1, 2, 'must not reach another field');
        assert.deepEqual(calls, []);
    });
}

test('cancellation clears owned preedit and timers', () => {
    const {injector, calls, flush} = fixture();
    injector.insertPartial(1, 1, 'pending');
    injector.insertPartial(1, 2, 'queued', 100);
    injector.cancel();
    flush();
    injector.insertCommit(1, 3, 'late');
    assert.deepEqual(calls, [['preedit', 'pending', 7, 7, 0], ['preedit', null, 0, 0, 0]]);
});

test('do not overwrite an existing IBus composition', () => {
    const {injector, calls, method} = fixture();
    method._preeditVisible = true;
    method._preeditStr = 'owned by another input method';
    injector.insertPartial(1, 1, 'dictation');
    injector.insertCommit(1, 2, 'dictation');
    assert.deepEqual(calls, []);
});

test('endpoint commits insert each suffix once and preserve later previews', () => {
    const {injector, calls} = fixture();
    injector.insertPartial(1, 1, 'twenty five');
    injector.insertCommit(1, 2, '25');
    injector.insertPartial(1, 3, '25 next utterance');
    injector.insertCommit(1, 4, '25 next utterance.');
    injector.insertCommit(1, 5, '25 next utterance.');
    assert.deepEqual(calls, [
        ['preedit', 'twenty five', 11, 11, 0], ['preedit', null, 0, 0, 0], ['commit', '25'],
        ['preedit', ' next utterance', 15, 15, 0], ['preedit', null, 0, 0, 0], ['commit', ' next utterance.'],
    ]);
});

test('committed-only inserts successive finalized utterances without duplication', () => {
    const {injector, calls} = fixture();
    injector.insertCommit(1, 1, '$25');
    injector.insertCommit(1, 2, '$25. Tomorrow.');
    injector.insertCommit(1, 3, '$25. Tomorrow.');
    assert.deepEqual(calls, [['commit', '$25'], ['commit', '. Tomorrow.']]);
});

test('pause ITN preview remains replaceable until a real final', () => {
    const {injector, calls} = fixture();
    injector.insertPartial(1, 1, 'twenty five');
    injector.insertPartial(1, 2, '25');
    injector.insertPartial(1, 3, 'twenty five dollars');
    injector.insertCommit(1, 4, '$25');
    assert.deepEqual(calls.filter(call => call[0] === 'commit'), [['commit', '$25']]);
});
