import {ExtensionPreferences, gettext as _} from 'resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js';
import {createPreferencesPage} from './preferencesPage.js';

export default class NeMotePreferences extends ExtensionPreferences {
    fillPreferencesWindow(window) {
        const generalPage = createPreferencesPage(this.getSettings(), {gettext: _});
        for (const page of generalPage.pages)
            window.add(page);
    }
}
