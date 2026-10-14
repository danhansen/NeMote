// GSettings uses hyphens; service JSON/D-Bus dictionaries use underscores.
// Keep the grouping here identical for the Shell and preferences clients.
export const INSERTION_SETTINGS = [
    'spoken-punctuation', 'insert-partials', 'stream-insert-delay-ms',
];

export const RUNTIME_SETTINGS = [
    'streaming-latency-ms', 'language', 'model-root', 'worker-path',
    'model-installer-path', 'num-threads', 'sample-rate',
    'itn', 'pnc', 'pnc-model-path', 'pnc-variant', 'endpoint-mode',
    'phrase-boosting', 'boost-phrases', 'boost-tokenizer-path',
    'vad-filtering', 'vad-model-path',
];

const MIRRORED_SETTINGS = [
    'backend', 'model-profile', 'model-family', 'input-device',
    ...INSERTION_SETTINGS, ...RUNTIME_SETTINGS,
];

const SETTING_TYPES = {
    s: {jsType: 'string', setter: 'set_string'},
    b: {jsType: 'boolean', setter: 'set_boolean'},
    u: {jsType: 'number', setter: 'set_uint'},
};

export function optionsFromSettings(settings, keys) {
    // get_value already returns a correctly typed GLib.Variant. Reuse the
    // schema's type rather than maintaining a second list of D-Bus signatures.
    return Object.fromEntries(keys.map(key => [
        key.replaceAll('-', '_'), settings.get_value(key),
    ]));
}

export function syncSettingsFromConfig(settings, config) {
    // The caller suppresses its change handlers while applying this snapshot.
    // Missing fields are left alone for compatibility with older services.
    for (const key of MIRRORED_SETTINGS) {
        const value = config[key.replaceAll('-', '_')];
        const type = SETTING_TYPES[settings.get_value(key).get_type_string()];
        if (type && typeof value === type.jsType)
            settings[type.setter](key, value);
    }
    // The legacy toggle-shortcut key is no longer mirrored: assignments now
    // belong to the desktop portal, not to our service configuration.
}
