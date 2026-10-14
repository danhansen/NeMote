// Toolkit-independent decoding and display helpers for service snapshots.
export function deepUnpackMap(value) {
    if (!value)
        return {};
    const unpacked = deepUnpackValue(value);
    if (!unpacked || typeof unpacked !== 'object' || Array.isArray(unpacked))
        return {};
    return unpacked;
}

export function deepUnpackValue(value) {
    const unpacked = value?.deep_unpack ? value.deep_unpack() : value;
    if (!unpacked || typeof unpacked !== 'object' || Array.isArray(unpacked))
        return unpacked;
    const result = {};
    for (const [key, variant] of Object.entries(unpacked))
        result[key] = deepUnpackValue(variant);
    return result;
}

export function formatInstallProgress(progress) {
    const profile = typeof progress.profile === 'string' ? progress.profile : '';
    const message = typeof progress.message === 'string'
        ? progress.message
        : typeof progress.phase === 'string'
            ? progress.phase
            : '';
    if (!profile && !message)
        return '';
    if (!profile)
        return message;
    if (!message)
        return profile;
    return `${profile}: ${message}`;
}

export function numberValue(value) {
    if (typeof value === 'number' && Number.isFinite(value))
        return value;
    return null;
}

export function formatError(error) {
    const message = error?.message ?? String(error);
    return message.replace(/^GDBus\.Error:[^:]+:\s*/, '');
}
