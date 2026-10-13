// Run with `gjs -m tests/check_preferences_layout.js`
// under a GTK display (including GTK's Broadway backend). No service is started
// and settings use a memory backend. Exercises the real production GTK widgets.
import Adw from 'gi://Adw';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Gtk from 'gi://Gtk';
import {createPreferencesPage} from '../extensions/gnome-shell/nemote@dhansen.dev/preferencesPage.js';

GLib.setenv('GSETTINGS_BACKEND', 'memory', true);
Adw.init();
Gtk.Settings.get_default().gtk_enable_animations = false;
const directory = Gio.File.new_for_uri(import.meta.url).get_parent().get_parent();
const extension = directory.get_child('extensions/gnome-shell/nemote@dhansen.dev');
const schemaDirectory = GLib.dir_make_tmp('nemote-layout-schemas-XXXXXX');
const [spawned, , stderr, status] = GLib.spawn_sync(null, [
    'glib-compile-schemas', '--strict', '--targetdir', schemaDirectory,
    extension.get_child('schemas').get_path(),
], null, GLib.SpawnFlags.SEARCH_PATH, null);
if (!spawned || status !== 0)
    throw new Error(new TextDecoder().decode(stderr));
const schemaSource = Gio.SettingsSchemaSource.new_from_directory(schemaDirectory,
    Gio.SettingsSchemaSource.get_default(), false);
const settings = new Gio.Settings({settings_schema:
    schemaSource.lookup('org.gnome.shell.extensions.nemote', false)});
settings.set_string('model-family', 'english');
settings.set_string('endpoint-mode', 'preview');

const page = createPreferencesPage(settings, {connectService: false});
page._profiles[0].installed = true;
page._rebuildProfileOptions();
page._pncModels = [
    {id: 'pnc-bert', title: 'BERT', installed: false},
    {id: 'pnc-distilbert', title: 'DistilBERT', installed: true},
];
page._rebuildProfileRows();
page._deviceModel.append('Core Ultra 200V Series Processors HD Audio Digital Microphone');
page._deviceRow.selected = 1;
page._pncVariantRow.selected = 1;
page._boostPhrasesRow.expanded = true;
const window = new Adw.PreferencesWindow({default_width: 680, default_height: 900});
for (const preferencesPage of page.pages)
    window.add(preferencesPage);
window.present();
const wait = () => new Promise(resolve => GLib.timeout_add(GLib.PRIORITY_DEFAULT,
    500, () => { resolve(); return GLib.SOURCE_REMOVE; }));
function* descendants(widget) {
    yield widget;
    for (let child = widget.get_first_child(); child; child = child.get_next_sibling())
        yield* descendants(child);
}
function assert(condition, message) {
    if (!condition)
        throw new Error(message);
}
try {
    for (const width of [680, 480]) {
        // Also cover a larger system font at the narrow size.
        if (width === 480)
            Gtk.Settings.get_default().gtk_font_name = 'Sans 14';
        window.set_default_size(width, 900);
        await wait();
        // Allocate explicitly for deterministic sizing on headless displays.
        window.allocate(width, 900, -1, null);
        assert(window.get_width() === width, 'Test window did not resize');
        const scroll = [...descendants(page)].find(widget => widget instanceof Gtk.ScrolledWindow);
        const adjustment = scroll.get_vadjustment();
        adjustment.set_value(0);
        await wait();
        for (const [field, expected] of [
            ['_profileRow', 'English (lower WER)'],
            ['_deviceRow', 'Core Ultra 200V Series Processors HD Audio Digital Microphone'],
            ['_pncVariantRow', 'DistilBERT — Faster'],
            ['_endpointModeRow', 'Format Preview'],
            ['_insertPartialsRow', 'Live Preview and Final Text'],
        ]) {
            const row = page[field];
            const owner = row.get_ancestor(Adw.PreferencesPage);
            window.set_visible_page(owner);
            await wait();
            const ownerScroll = [...descendants(owner)].find(widget => widget instanceof Gtk.ScrolledWindow);
            const ownerAdjustment = ownerScroll.get_vadjustment();
            assert(row.use_subtitle && row.subtitle === expected,
                `${field}: selected value was replaced or hidden: ${row.subtitle}`);
            const [, bounds] = row.compute_bounds(owner);
            ownerAdjustment.set_value(ownerAdjustment.get_value() + bounds.origin.y - 180);
            await wait();
            window.allocate(width, 900, -1, null);
            assert(row.get_width() <= window.get_width(), `${field} exceeds window width`);
            const labels = [...descendants(row)].filter(widget =>
                widget instanceof Gtk.Label && widget.get_label() === expected && widget.get_mapped());
            assert(labels.length > 0, `${field}: selected label is not visible`);
            for (const label of labels)
                assert(!label.get_layout().is_ellipsized(), `${field}: selected value is ellipsized at ${width}px`);
            for (const label of labels)
                assert(label.get_width() <= row.get_width(), `${field}: label exceeds its row`);
        }
        window.set_visible_page(page.advancedPage);
        await wait();
        window.allocate(width, 900, -1, null);
        for (const field of ['_pncModelPathRow', '_boostTokenizerPathRow', '_vadModelPathRow']) {
            assert([...descendants(page.advancedPage)].includes(page[field]), `${field} missing from Advanced`);
            assert(![...descendants(page)].includes(page[field]), `${field} still in General`);
        }
        window.set_visible_page(page);
        print(`PASS: real GTK layout at ${width}px`);
    }
} finally {
    window.close();
    window.run_dispose();
    // This test creates exactly these generated schema files, no recursive delete.
    Gio.File.new_for_path(`${schemaDirectory}/gschemas.compiled`).delete(null);
    Gio.File.new_for_path(schemaDirectory).delete(null);
}
