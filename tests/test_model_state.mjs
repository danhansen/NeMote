import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

const helperUrl = new URL(
    '../extensions/gnome-shell/wordpipe@dhansen.dev/modelState.js',
    import.meta.url);
const helperSource = readFileSync(helperUrl, 'utf8');
const helperModule = await import(
    `data:text/javascript;base64,${Buffer.from(helperSource).toString('base64')}`);
const {installedModelProfiles, streamingChunkChoices} = helperModule;

test('chunk selector exposes every metadata-supported size without a fixed list', () => {
    assert.deepEqual(streamingChunkChoices([1120, 80, 560, 160]), [80, 160, 560, 1120]);
    assert.deepEqual(streamingChunkChoices([240, 80, 320, 240]), [80, 240, 320]);
});

test('invalid chunk metadata preserves the compatibility fallback', () => {
    for (const values of [undefined, [], [80, 0], [80, '160'], [1.5]])
        assert.equal(streamingChunkChoices(values), null);
});

test('model selector exposes installed profiles only', () => {
    const profiles = [
        {id: 'compact', installed: true},
        {id: 'compact-english', installed: false},
        {id: 'fast', installed: true},
        {id: 'fast-english', installed: false},
    ];

    assert.deepEqual(
        installedModelProfiles(profiles).map(profile => profile.id),
        ['compact', 'fast']);
});

test('model selector is empty when no profile is installed', () => {
    assert.deepEqual(installedModelProfiles([
        {id: 'compact', installed: false},
        {id: 'fast', installed: false},
    ]), []);
});
