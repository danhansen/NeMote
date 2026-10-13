import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL(
    '../extensions/gnome-shell/nemote@dhansen.dev/preferencesPage.js', import.meta.url), 'utf8');
const method = (name, next) => source.slice(source.indexOf(`    ${name}(`), source.indexOf(`    ${next}(`));

function harness() {
    class Row {
        constructor(options) { Object.assign(this, options); }
        add_suffix(button) { this.button = button; }
        connect(_signal, callback) { this.click = callback; }
    }
    const Page = vm.runInNewContext(`(class {
        ${method('_rebuildProfileRows', '_handleState')}
        ${method('_handleState', '_handleInstallProgress')}
        ${method('_setInstallProgress', '_syncInstallButtons')}
        ${method('_syncInstallButtons', '_rebuildProfileOptions')}
    })`, {
        _: text => text, Adw: {ActionRow: Row}, Gtk: {Button: Row, Align: {CENTER: 0}},
        modelPresetFromValues: () => null, formatMetrics: () => '', formatInstallProgress: progress => progress.message ?? '',
        numberValue: value => typeof value === 'number' ? value : null,
    });
    const page = new Page();
    const group = () => ({rows: [], add(row) { this.rows.push(row); },
        remove(row) { this.rows.splice(this.rows.indexOf(row), 1); }});
    Object.assign(page, {
        _profiles: [{id: 'nemo-q8-english', title: 'English', installed: true}],
        _pncModels: [{id: 'pnc-bert', title: 'BERT', installed: false},
            {id: 'pnc-distilbert', title: 'DistilBERT', installed: true}],
        _installRows: new Map(), _installButtons: new Map(),
        _modelGroup: group(), _pncGroup: group(),
        _installing: false, _installingProfile: '', _runtimeBusy: false,
        _latencyRow: {}, _statusRow: {}, _metricsRow: {}, _startButton: {}, _stopButton: {},
        _progressRow: {}, _progressBar: {}, calls: [],
        _callRemote(name, id) { this.calls.push([name, id]); },
    });
    return page;
}

test('PnC installation rows share the explicit ASR install action, without changing settings', () => {
    const page = harness();
    page._rebuildProfileRows();
    assert.equal(page._modelGroup.rows.length, 1);
    assert.equal(page._pncGroup.rows.length, 2);
    const bert = page._installButtons.get('pnc-bert');
    assert.equal(bert.sensitive, true);
    assert.equal(page._installButtons.get('pnc-distilbert').sensitive, false);
    assert.equal(page._installButtons.get('pnc-distilbert').visible, false);
    bert.click();
    assert.deepEqual(page.calls, [['InstallModel', 'pnc-bert']]);
    page._rebuildProfileRows();
    assert.equal(page._pncGroup.rows.length, 2, 'refresh must not duplicate rows');
});

test('missing selected PnC disables Start but leaves its install button available', () => {
    const page = harness();
    page._rebuildProfileRows();
    const ready = {selected_model_installed: true, selected_pnc_model_installed: false};
    page._handleState(ready);
    assert.equal(page._startButton.sensitive, false);
    assert.match(page._statusRow.subtitle, /Install the selected punctuation model/);
    assert.equal(page._installButtons.get('pnc-bert').sensitive, true);
    page._handleState({...ready, installing: true, installing_profile: 'pnc-bert'});
    assert.equal(page._installButtons.get('pnc-bert').sensitive, false);
    page._setInstallProgress({profile: 'pnc-bert', phase: 'downloading', message: 'Downloading: 50%', fraction: 0.5});
    assert.equal(page._installRows.get('pnc-bert').subtitle, 'Downloading: 50%');
    assert.equal(page._progressBar.fraction, 0.5);
    page._rebuildProfileRows();
    assert.equal(page._installRows.get('pnc-bert').subtitle, 'Downloading: 50%');
    page._setInstallProgress({profile: 'pnc-bert', phase: 'converting', message: 'Converting model'});
    assert.equal(page._progressBar.visible, false);
    assert.equal(page._installRows.get('pnc-bert').subtitle, 'Converting model');
    page._handleState({...ready, selected_pnc_model_installed: true});
    assert.equal(page._startButton.sensitive, true);
    page._handleState({...ready, listening: true});
    assert.equal(page._installButtons.get('pnc-bert').sensitive, false);
});

test('runtime controls and model selection stay disabled across a busy inventory refresh', () => {
    const page = harness();
    page._profileRow = {};
    page._selectableProfiles = [{}];
    page._runtimeControls = [{sensitive: true}, {sensitive: true}];
    page._rebuildProfileRows();
    for (const busy of [{loading_model: true}, {listening: true}, {stopping: true}, {installing: true}]) {
        page._handleState(busy);
        page._rebuildProfileRows();
        assert.equal(page._profileRow.sensitive, false);
        assert.ok(page._runtimeControls.every(control => !control.sensitive));
        assert.equal(page._installButtons.get('pnc-bert').sensitive, false);
    }
    page._handleState({});
    assert.equal(page._profileRow.sensitive, true);
    assert.ok(page._runtimeControls.every(control => control.sensitive));
});
