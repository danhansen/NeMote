// Shared by the Shell and GTK preferences processes. No toolkit imports here.
import Gio from 'gi://Gio';

export const BUS_NAME = 'dev.nemote.Service';
export const OBJECT_PATH = '/dev/nemote/Service';

// CMake embeds this same XML in the native service. Keeping it next to this
// module also makes it available in source-only extension installations.
const protocol = Gio.File.new_for_uri(import.meta.url).get_parent().get_child('service.xml');
const [, contents] = protocol.load_contents(null);
export const NeMoteProxy = Gio.DBusProxy.makeProxyWrapper(new TextDecoder().decode(contents));
