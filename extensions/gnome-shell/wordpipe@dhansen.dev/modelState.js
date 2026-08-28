export function installedModelProfiles(profiles) {
    return profiles.filter(profile => profile.installed === true);
}
