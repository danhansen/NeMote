import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

const source = readFileSync(new URL(
    '../extensions/gnome-shell/wordpipe@dhansen.dev/serviceLifecycle.js',
    import.meta.url), 'utf8');
const {ServiceLifecycle} = await import(
    `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const deferred = () => {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return {promise, resolve};
};

test('disable while activation is pending shuts down the late instance', async () => {
    const activation = deferred();
    const stops = [];
    const controller = new ServiceLifecycle({
        start: () => activation.promise,
        stop: async owner => stops.push(owner),
        ready: () => assert.fail('disabled extension became ready'),
        error: error => assert.fail(error.message),
    });
    const pending = controller.enable();
    await tick();
    controller.disable();
    activation.resolve(':1.100');
    await pending;
    assert.deepEqual(stops, [':1.100']);
    assert.equal(controller.owner, null);
});

test('rapid re-enable waits for old instance shutdown before activation', async () => {
    const stopping = deferred();
    const starts = [];
    const stops = [];
    const controller = new ServiceLifecycle({
        start: async () => {
            const owner = `:1.${starts.length + 1}`;
            starts.push(owner);
            return owner;
        },
        stop: owner => { stops.push(owner); return stopping.promise; },
        ready: () => {},
        error: error => assert.fail(error.message),
    });
    await controller.enable();
    controller.disable();
    const pending = controller.enable();
    await tick();
    assert.equal(starts.length, 1);
    stopping.resolve();
    await pending;
    assert.equal(controller.owner, ':1.2');
    assert.deepEqual(stops, [':1.1']);
});

test('re-enable during pending activation keeps the needed service alive', async () => {
    const activation = deferred();
    const stops = [];
    const controller = new ServiceLifecycle({
        start: () => activation.promise,
        stop: async owner => stops.push(owner),
        ready: () => {},
        error: error => assert.fail(error.message),
    });
    controller.enable();
    await tick();
    controller.disable();
    const pending = controller.enable();
    activation.resolve(':1.101');
    await pending;
    assert.equal(controller.owner, ':1.101');
    assert.deepEqual(stops, []);
});

test('owner loss restarts only while extension is enabled', async () => {
    let starts = 0;
    const controller = new ServiceLifecycle({
        start: async () => `:1.${++starts}`,
        stop: async () => {},
        ready: () => {},
        error: error => assert.fail(error.message),
    });
    await controller.enable();
    controller.observeOwner(':1.1');
    controller.observeOwner(null);
    await tick();
    assert.equal(starts, 2);
    controller.disable();
    controller.observeOwner(':1.2');
    controller.observeOwner(null);
    await tick();
    assert.equal(starts, 2);
});

test('owner loss during activation cannot adopt a stale service instance', async () => {
    const activation = deferred();
    let starts = 0;
    const ready = [];
    const controller = new ServiceLifecycle({
        start: () => ++starts === 1 ? activation.promise : Promise.resolve(':1.2'),
        stop: async () => {},
        ready: owner => ready.push(owner),
        error: error => assert.fail(error.message),
    });
    const pending = controller.enable();
    await tick();
    controller.observeOwner(':1.1');
    controller.observeOwner(null);
    activation.resolve(':1.1');
    await pending;
    await controller.ensureStarted();
    assert.equal(controller.owner, ':1.2');
    assert.deepEqual(ready, [':1.2']);
});
