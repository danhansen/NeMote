import Adw from 'gi://Adw';
import Gdk from 'gi://Gdk';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Graphene from 'gi://Graphene';
import Gsk from 'gi://Gsk';
import Gtk from 'gi://Gtk';
import {addPreviewControls, createPreview, descendants, SCENARIOS} from './ui_preview_fixture.js';

Adw.init();
const schemaDirectory = GLib.getenv('NEMOTE_UI_SCHEMA_DIR');
if (!schemaDirectory)
    throw new Error('Launch this through scripts/preview-ui.');
const option = name => ARGV.includes(name) ? ARGV[ARGV.indexOf(name) + 1] : null;
const output = option('--audit');
const wait = (ms = 180) => new Promise(resolve => GLib.timeout_add(GLib.PRIORITY_DEFAULT,
    ms, () => { resolve(); return GLib.SOURCE_REMOVE; }));
const escape = value => GLib.markup_escape_text(String(value), -1);
const write = (name, value) => Gio.File.new_for_path(`${output}/${name}`)
    .replace_contents(value, null, false, Gio.FileCreateFlags.REPLACE_DESTINATION, null);
const slug = value => value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

async function reveal(page, widget) {
    const scroll = [...descendants(page)].find(child => child instanceof Gtk.ScrolledWindow);
    if (!scroll) throw new Error('Preferences page has no scroller');
    const adjustment = scroll.get_vadjustment();
    const [success, bounds] = widget.compute_bounds(page);
    if (!success) throw new Error('Cannot locate preference widget');
    adjustment.set_value(adjustment.get_value() + bounds.origin.y - 100);
    await wait();
}

async function capture(widget, window, filename) {
    // A searchable GtkPopover has its own native surface. Snapshot its real
    // content widget, not the native wrapper (which can yield an empty node).
    if (widget instanceof Gtk.Popover)
        widget = widget.get_first_child();
    for (let attempt = 0; attempt < 5; attempt++) {
        await wait();
        const width = widget.get_width();
        const height = widget.get_height();
        if (width <= 0 || height <= 0) continue;
        const content = new Gtk.Snapshot();
        new Gtk.WidgetPaintable({widget}).snapshot(content, width, height);
        const contentNode = content.to_node();
        // Never mistake the synthetic background for a successful capture.
        // Popovers can need an extra frame before their render node is ready.
        if (!contentNode) continue;
        const snapshot = new Gtk.Snapshot();
        const [, color] = window.get_style_context().lookup_color('window_bg_color');
        const bounds = new Graphene.Rect(); bounds.init(0, 0, width, height);
        snapshot.append_color(color, bounds);
        snapshot.append_node(contentNode);
        const node = snapshot.to_node();
        if (!node) continue;
        const renderer = new Gsk.CairoRenderer();
        try {
            renderer.realize(window.get_surface());
            const texture = renderer.render_texture(node, bounds);
            if (!texture.save_to_png(`${output}/${filename}`))
                throw new Error(`Failed saving ${filename}`);
        } finally {
            renderer.unrealize();
        }
        return {width, height};
    }
    throw new Error(`GTK did not render ${filename}`);
}

function checkLayout(widget, context, issues) {
    const width = widget.get_width();
    for (const child of descendants(widget)) {
        if (!child.get_mapped()) continue;
        if (child instanceof Adw.PreferencesRow && child.get_width() > width + 1)
            issues.push(`${context}: row exceeds section width: ${child.title}`);
        if (!(child instanceof Gtk.Label) || !child.get_label()) continue;
        if (!child.get_text())
            issues.push(`${context}: nonempty label rendered as empty: ${child.get_label()}`);
        if (child.get_layout().is_ellipsized())
            issues.push(`${context}: ellipsized label: ${child.get_label()}`);
        const [success, bounds] = child.compute_bounds(widget);
        if (success && (bounds.origin.x < -1 || bounds.origin.x + bounds.size.width > width + 1))
            issues.push(`${context}: label extends outside section: ${child.get_label()}`);
    }
}

async function audit() {
    Gtk.Settings.get_default().gtk_enable_animations = false;
    const originalFont = Gtk.Settings.get_default().gtk_font_name;
    const manifest = {scope: 'Production GTK preferences, dropdowns, shortcut dialog and simulated service states. '
        + 'The headless launcher also captures the production GNOME panel in shell.html.', captures: [], issues: []};
    const cases = [
        {name: 'normal-light', width: 680, dark: false},
        {name: 'narrow-dark-large-text', width: 480, dark: true, font: 'Sans 14'},
    ];
    for (const style of cases) {
        Adw.StyleManager.get_default().color_scheme = style.dark ? Adw.ColorScheme.FORCE_DARK : Adw.ColorScheme.FORCE_LIGHT;
        Gtk.Settings.get_default().gtk_font_name = style.font ?? originalFont;
        for (const scenario of SCENARIOS) {
            const preview = createPreview(schemaDirectory, scenario, style.width);
            try {
                preview.page._boostPhrasesRow.expanded = true;
                preview.window.present();
                await wait(350);
                if (Math.abs(preview.window.get_width() - style.width) > 1)
                    manifest.issues.push(`${style.name}/${scenario}: window requested ${style.width}px but allocated ${preview.window.get_width()}px`);
                const pages = preview.page.pages;
                for (const page of pages) {
                    preview.window.set_visible_page(page);
                    await wait();
                    if (scenario === 'ready') {
                        const name = `${style.name}-page-${slug(page.title)}`;
                        const dimensions = await capture(preview.window, preview.window, `${name}.png`);
                        manifest.captures.push({name, scenario, style: style.name, page: page.title, section: 'Full window',
                            filename: `${name}.png`, ...dimensions});
                    }
                    const groups = [...descendants(page)].filter(widget => widget instanceof Adw.PreferencesGroup);
                    for (const group of groups) {
                        // All sections in the default state; otherwise capture
                        // the sections whose content changes with service state.
                        if (scenario !== 'ready' && !['Speech Recognition', 'Punctuation and Capitalization', 'Transcript', 'Service', 'Model Setup'].includes(group.title))
                            continue;
                        await reveal(page, group);
                        const name = `${style.name}-${scenario}-${slug(page.title)}-${slug(group.title)}`;
                        checkLayout(group, name, manifest.issues);
                        const dimensions = await capture(group, preview.window, `${name}.png`);
                        manifest.captures.push({name, scenario, style: style.name, page: page.title, section: group.title,
                            filename: `${name}.png`, ...dimensions});
                    }
                }
                if (scenario === 'ready') {
                    preview.window.set_visible_page(preview.page);
                    await wait();
                    for (const field of ['_profileRow', '_latencyRow', '_languageRow', '_deviceRow',
                        '_pncVariantRow', '_endpointModeRow', '_insertPartialsRow', '_shortcutModeRow']) {
                        const row = preview.page[field];
                        const owner = row.get_ancestor(Adw.PreferencesPage);
                        preview.window.set_visible_page(owner);
                        await wait();
                        await reveal(owner, row);
                        row.activate();
                        await wait();
                        const popup = [...descendants(row)].find(widget => widget instanceof Gtk.Popover && widget.get_mapped());
                        if (!popup) throw new Error(`No dropdown popup found for ${field}`);
                        const name = `${style.name}-dropdown-${field.slice(1)}`;
                        checkLayout(popup, name, manifest.issues);
                        const dimensions = await capture(popup, preview.window, `${name}.png`);
                        manifest.captures.push({name, scenario, style: style.name, page: owner.title, section: row.title,
                            filename: `${name}.png`, ...dimensions});
                        popup.popdown();
                        await wait();
                    }
                    preview.window.set_visible_page(preview.page);
                    await wait();
                    await reveal(preview.page, preview.page._shortcutRow);
                    preview.page._shortcutButton.emit('clicked');
                    await wait(350);
                    const editor = preview.page._shortcutButton._editor;
                    const name = `${style.name}-shortcut-dialog`;
                    const dimensions = await capture(editor, editor, `${name}.png`);
                    manifest.captures.push({name, scenario, style: style.name, page: preview.page.title, section: 'Shortcut dialog',
                        filename: `${name}.png`, ...dimensions});
                    editor.close();
                    if (preview.settings.get_boolean('shortcut-capture-active'))
                        manifest.issues.push(`${style.name}: shortcut capture remains active after closing`);
                }
            } finally {
                preview.close();
            }
            print(`Captured ${style.name}/${scenario}`);
        }
    }
    manifest.issues = [...new Set(manifest.issues)];
    write('manifest.json', JSON.stringify(manifest, null, 2));
    const gallery = manifest.captures.map(item => `<section><h2>${escape(item.name)}</h2><p>${item.width} × ${item.height}</p>`
        + `<a href="${item.filename}"><img loading="lazy" src="${item.filename}" alt="${escape(item.name)}"></a></section>`).join('\n');
    write('index.html', `<!doctype html><meta charset="utf-8"><title>NeMote UI audit</title>
        <style>body{font:16px system-ui;background:#e9e9ed;color:#222;margin:24px}section{display:inline-block;vertical-align:top;margin:12px;padding:16px;background:white;border-radius:12px;max-width:680px}img{max-width:100%;height:auto}h2{font-size:16px;overflow-wrap:anywhere}</style>
        <h1>NeMote preferences audit</h1><a href="shell.html">GNOME panel gallery (headless audit)</a><p>${escape(manifest.scope)}</p><p>${manifest.captures.length} captures; ${manifest.issues.length} layout alerts.</p>
        <ul>${manifest.issues.map(issue => `<li>${escape(issue)}</li>`).join('')}</ul>${gallery}`);
    print(`Audit: ${manifest.captures.length} captures, ${manifest.issues.length} layout alerts. ${output}/index.html`);
    if (manifest.issues.length) throw new Error('UI layout audit found issues; see manifest.json');
}

if (output) {
    await audit();
} else {
    const preview = createPreview(schemaDirectory, option('--scenario') ?? 'ready', Number(option('--width') ?? 680));
    addPreviewControls(preview);
    const loop = new GLib.MainLoop(null, false);
    preview.window.connect('close-request', () => { loop.quit(); return false; });
    preview.window.present();
    loop.run();
    preview.close();
}
