export function installedModelProfiles(profiles) {
    return profiles.filter(profile => profile.installed === true);
}

export function streamingChunkChoices(values) {
    if (!Array.isArray(values) || !values.length ||
        !values.every(value => Number.isSafeInteger(value) && value > 0))
        return null;
    return [...new Set(values)].sort((a, b) => a - b);
}
