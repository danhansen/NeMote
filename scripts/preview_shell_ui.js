// Executed only by an isolated GNOME Shell through --automation-script.
// Imports the production indicator without enabling the extension or its service.
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Shell from 'gi://Shell';
import St from 'gi://St';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as Scripting from 'resource:///org/gnome/shell/ui/scripting.js';
import {createIndicator} from '../extensions/gnome-shell/nemote@dhansen.dev/extension.js';

export const METRICS = {};
Gio._promisify(Shell.Screenshot.prototype, 'screenshot_area', 'screenshot_area_finish');
const output = GLib.getenv('NEMOTE_UI_AUDIT_OUTPUT');
const root = Gio.File.new_for_uri(import.meta.url).get_parent().get_parent();
const escape = text => GLib.markup_escape_text(String(text), -1);
const write = (name, value) => Gio.File.new_for_path(`${output}/${name}`)
    .replace_contents(value, null, false, Gio.FileCreateFlags.REPLACE_DESTINATION, null);

export async function run() {
    if (!output)
        throw new Error('This script requires the isolated scripts/preview-ui launcher.');
    new Gio.Settings({schema_id: 'org.gnome.desktop.interface'}).set_boolean('enable-animations', false);
    Main.overview.hide();
    const theme = St.ThemeContext.get_for_stage(global.stage).get_theme();
    const stylesheet = root.get_child('extensions/gnome-shell/nemote@dhansen.dev/stylesheet.css');
    theme.load_stylesheet(stylesheet);
    const indicator = createIndicator({toggleDictation() {}, installModel() {},
        selectModelProfile() {}, openPreferences() {}}, text => text);
    Main.panel.addToStatusArea('nemote-ui-preview', indicator);
    const manifest = {scope: 'Production GNOME Shell indicator in an isolated compositor, simulated service only.',
        captures: [], issues: []};
    try {
        for (const scale of [1, 1.35]) {
        new Gio.Settings({schema_id: 'org.gnome.desktop.interface'}).set_double('text-scaling-factor', scale);
        for (const [scenario, state, available] of [
            ['ready', {}, true],
            ['missing', {selected_model_installed: false}, true],
            ['downloading', {installing: true, installing_profile: 'nemo-q8'}, true],
            ['converting', {installing: true, installing_profile: 'nemo-q8'}, true],
            ['loading', {loading_model: true}, true],
            ['listening', {listening: true}, true],
            ['stopping', {stopping: true}, true],
            ['error', {last_error: 'Model installation failed: not enough disk space in /home/example/.local/share/nemote/models. Free some space and retry. Response: <error> & unavailable.'}, true],
            ['pnc-missing', {selected_pnc_model_installed: false}, true],
            ['unavailable', {}, false],
        ]) {
            const profiles = [
                {id: 'nemo-q8-english', title: 'English (lower WER)', installed: scenario !== 'missing'},
                {id: 'nemo-q8', title: 'Multilingual', installed: false},
            ];
            indicator.setState(state, available);
            indicator.setProfiles(profiles, 'nemo-q8-english');
            const protectedStatus = indicator._statusItem.label.text;
            if (indicator._statusProtected) {
                indicator.setMetrics('RTF 0.420');
                if (indicator._statusItem.label.text !== protectedStatus)
                    manifest.issues.push(`${scenario}: metrics overwrite an important status`);
            }
            if (state.installing)
                indicator.setInstallProgress('nemo-q8', {phase: scenario === 'converting' ? 'converting' : 'downloading',
                    ...(scenario === 'downloading' ? {fraction: 0.5} : {})});
            if (state.listening)
                indicator.setVoiceLevel(0.04);
            indicator.menu.open();
            await Scripting.sleep(250);
            const actor = indicator.menu.actor;
            const [x, y] = actor.get_transformed_position();
            const [width, height] = actor.get_transformed_size();
            if (x < 0 || y < 0 || x + width > global.stage.width + 1 || y + height > global.stage.height + 1)
                manifest.issues.push(`${scenario}: menu extends offscreen`);
            const status = indicator._statusItem.label.clutter_text;
            if (status.get_layout().is_ellipsized())
                manifest.issues.push(`${scenario}: status is ellipsized`);
            const busy = !available || state.installing || state.loading_model || state.listening || state.stopping;
            for (const row of indicator._profileItems.values()) {
                if (busy && row.installButton.reactive)
                    manifest.issues.push(`${scenario}: install button remains active while busy or unavailable`);
            }
            const filename = `shell-panel-${scale === 1 ? 'normal' : 'large-text'}-${scenario}.png`;
            const file = Gio.File.new_for_path(`${output}/${filename}`);
            const stream = file.replace(null, false, Gio.FileCreateFlags.REPLACE_DESTINATION, null);
            try {
                await new Shell.Screenshot().screenshot_area(Math.max(0, Math.floor(x)), 0,
                    Math.min(Math.ceil(width), global.stage.width - Math.max(0, Math.floor(x))),
                    Math.min(Math.ceil(y + height + 8), global.stage.height), stream);
            } finally {
                stream.close(null);
            }
            manifest.captures.push({scenario, scale, filename});
            indicator.menu.close();
            print(`Captured Shell panel/${scenario}`);
        }
        }
    } finally {
        indicator.destroy();
        theme.unload_stylesheet(stylesheet);
    }
    write('shell-manifest.json', JSON.stringify(manifest, null, 2));
    const gallery = manifest.captures.map(item => `<section><h2>${escape(item.scenario)} (text scale ${item.scale})</h2>`
        + `<img src="${item.filename}" alt="${escape(item.scenario)}"></section>`).join('\n');
    write('shell.html', `<!doctype html><meta charset="utf-8"><title>NeMote Shell UI audit</title>
        <style>body{font:16px system-ui;background:#eee;margin:24px}section{display:inline-block;vertical-align:top;padding:16px;margin:12px;background:white;border-radius:12px}img{max-width:100%;height:auto}</style>
        <h1>NeMote GNOME panel audit</h1><a href="index.html">Preferences gallery</a>
        <p>${escape(manifest.scope)}</p><p>${manifest.issues.length} layout alerts.</p>${gallery}`);
    if (manifest.issues.length)
        throw new Error(manifest.issues.join('\n'));
}
