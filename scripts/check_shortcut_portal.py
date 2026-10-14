#!/usr/bin/env python3
"""Exercise the production service against a fake portal on a private D-Bus."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from gi.repository import Gio, GLib

PORTAL = "org.freedesktop.portal.Desktop"
PATH = "/org/freedesktop/portal/desktop"
SHORTCUTS = "org.freedesktop.portal.GlobalShortcuts"
XML = """<node>
<interface name="org.freedesktop.host.portal.Registry">
  <method name="Register"><arg type="s" direction="in"/><arg type="a{sv}" direction="in"/></method>
</interface>
<interface name="org.freedesktop.portal.GlobalShortcuts">
  <property name="version" type="u" access="read"/>
  <method name="CreateSession"><arg type="a{sv}" direction="in"/><arg type="o" direction="out"/></method>
  <method name="ListShortcuts"><arg type="o" direction="in"/><arg type="a{sv}" direction="in"/><arg type="o" direction="out"/></method>
  <method name="BindShortcuts"><arg type="o" direction="in"/><arg type="a(sa{sv})" direction="in"/><arg type="s" direction="in"/><arg type="a{sv}" direction="in"/><arg type="o" direction="out"/></method>
  <method name="ConfigureShortcuts"><arg type="o" direction="in"/><arg type="s" direction="in"/><arg type="a{sv}" direction="in"/></method>
</interface>
</node>"""


class Portal:
    def __init__(self, bus):
        self.bus = bus
        self.description = ""
        self.session = ""
        self.cancel = False
        self.calls = []
        self.version = 2
        self.registrations = []
        for interface in Gio.DBusNodeInfo.new_for_xml(XML).interfaces:
            self.registrations.append(bus.register_object(PATH, interface, self.method,
                lambda *args: GLib.Variant("u", self.version), None))
        self.own()

    def own(self):
        self.bus.call_sync("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus",
            "RequestName", GLib.Variant("(su)", (PORTAL, 0)), None, Gio.DBusCallFlags.NONE, 2000, None)

    def release(self):
        self.bus.call_sync("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus",
            "ReleaseName", GLib.Variant("(s)", (PORTAL,)), None, Gio.DBusCallFlags.NONE, 2000, None)

    def shortcuts(self):
        return [("dictation", {"description": GLib.Variant("s", "Dictation"),
            "trigger_description": GLib.Variant("s", self.description)})] if self.description else []

    def method(self, bus, sender, path, interface, name, parameters, invocation):
        args = parameters.unpack()
        self.calls.append(name)
        if name == "Register":
            assert args[0] == "dev.nemote.Service"
            invocation.return_value(None)
            return
        if name == "ConfigureShortcuts":
            assert args[0] == self.session
            self.description = "Super+D"
            bus.emit_signal(None, PATH, SHORTCUTS, "ShortcutsChanged",
                GLib.Variant("(oa(sa{sv}))", (self.session, self.shortcuts())))
            invocation.return_value(None)
            return
        options = args[-1]
        peer = sender[1:].replace(".", "_")
        handle = f"{PATH}/request/{peer}/{options['handle_token']}"
        results = {}
        if name == "CreateSession":
            self.session = f"{PATH}/session/{peer}/{options['session_handle_token']}"
            close_xml = '<node><interface name="org.freedesktop.portal.Session"><method name="Close"/></interface></node>'
            info = Gio.DBusNodeInfo.new_for_xml(close_xml).interfaces[0]
            self.registrations.append(bus.register_object(self.session, info,
                lambda bus, sender, path, interface, name, params, call: call.return_value(None), None, None))
            results["session_handle"] = GLib.Variant("s", self.session)
        else:
            assert args[0] == self.session
            if name == "BindShortcuts" and not self.cancel:
                assert args[1][0][0] == "dictation"
                self.description = self.description or "Ctrl+Alt+Space"
            results["shortcuts"] = GLib.Variant("a(sa{sv})", self.shortcuts())
        code = 1 if name == "BindShortcuts" and self.cancel else 0
        self.cancel = False
        # Deliberately emit before replying: clients must subscribe beforehand.
        bus.emit_signal(None, handle, "org.freedesktop.portal.Request", "Response",
            GLib.Variant("(ua{sv})", (code, results)))
        invocation.return_value(GLib.Variant("(o)", (handle,)))

    def key(self, active, session=None, shortcut="dictation"):
        self.bus.emit_signal(None, PATH, SHORTCUTS, "Activated" if active else "Deactivated",
            GLib.Variant("(osta{sv})", (session or self.session, shortcut, 0, {})))


def main():
    if os.environ.get("NEMOTE_TEST_BUS") != "1":
        raise SystemExit("Use dbus-run-session -- env NEMOTE_TEST_BUS=1; never use the desktop bus")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", type=Path, required=True)
    args = parser.parse_args()
    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    portal = Portal(bus)
    with tempfile.TemporaryDirectory(prefix="nemote-shortcut-test-") as temporary:
        root = Path(temporary)
        model = root / "models/nemotron-nemo-en-q8/model.gguf"
        model.parent.mkdir(parents=True)
        with model.open("wb") as output:
            output.write(b"GGUF")
            output.truncate(699872960)
        worker = root / "worker"
        shutil.copy2(Path(__file__).resolve().parents[1] / "tests/fixtures/mock_runtime_worker.py", worker)
        worker.chmod(0o755)
        config = root / "service.json"
        config.write_text(json.dumps({"model_root": str(root / "models"), "worker_path": str(worker)}))
        process = subprocess.Popen([str(args.service), "--config", str(config)], stderr=subprocess.PIPE, text=True)

        def call(method, params=None):
            return bus.call_sync("dev.nemote.Service", "/dev/nemote/Service", "dev.nemote.Service1",
                method, params, None, Gio.DBusCallFlags.NO_AUTO_START, 2000, None).unpack()

        def wait(predicate):
            deadline = time.monotonic() + 5
            state = {}
            while time.monotonic() < deadline:
                while GLib.MainContext.default().pending():
                    GLib.MainContext.default().iteration(False)
                if process.poll() is not None:
                    raise AssertionError(process.stderr.read())
                try:
                    state = call("GetState")[0]
                    if predicate(state):
                        return state
                except GLib.Error:
                    pass
                time.sleep(0.01)
            raise AssertionError(f"State timed out: {state}; portal calls: {portal.calls}")

        try:
            wait(lambda state: True)
            call("RegisterShellClient")
            wait(lambda state: state["shortcut_status"] == "not-configured")
            assert "BindShortcuts" not in portal.calls, "First launch must not prompt"
            portal.cancel = True
            call("ConfigureShortcut")
            wait(lambda state: state["shortcut_status"] == "error")
            call("ConfigureShortcut")
            wait(lambda state: state["shortcut_status"] == "ready")
            assert portal.calls.count("Register") == 1, "Retry must reuse registered identity"
            call("ConfigureShortcut")
            wait(lambda state: state["shortcut_description"] == "Super+D")

            call("SetShortcutMode", GLib.Variant("(s)", ("push-to-talk",)))
            portal.key(True)
            started = wait(lambda state: state["listening"] and state["model_loaded"])
            portal.key(True)  # Autorepeat must not toggle or start a second session.
            assert wait(lambda state: state["listening"])["session_id"] == started["session_id"]
            portal.key(False)
            wait(lambda state: not state["listening"] and not state["stopping"])

            call("SetShortcutMode", GLib.Variant("(s)", ("toggle",)))
            portal.key(True)
            wait(lambda state: state["listening"])
            portal.key(False)
            wait(lambda state: state["listening"])
            portal.key(True)
            wait(lambda state: not state["listening"] and not state["stopping"])
            portal.key(False)

            # A release must not stop dictation started through the panel.
            call("SetShortcutMode", GLib.Variant("(s)", ("push-to-talk",)))
            call("Start")
            wait(lambda state: state["listening"])
            portal.key(True)
            wait(lambda state: state["listening"])
            portal.key(False)
            wait(lambda state: state["listening"])
            call("Stop")
            wait(lambda state: not state["stopping"])

            portal.key(True)
            wait(lambda state: state["listening"])
            old_session = portal.session
            portal.release()
            wait(lambda state: state["shortcut_status"] == "unavailable" and not state["stopping"] and not state["listening"])
            portal.own()
            wait(lambda state: state["shortcut_status"] == "ready")
            assert portal.calls.count("Register") == 2
            portal.key(True, session=old_session)
            portal.key(True, shortcut="unrelated")
            wait(lambda state: state["shortcut_status"] == "ready")
            assert not call("GetState")[0]["listening"]

            bus.emit_signal(None, portal.session, "org.freedesktop.portal.Session", "Closed", GLib.Variant("(a{sv})", ({},)))
            wait(lambda state: state["shortcut_status"] == "error")
            call("ConfigureShortcut")
            wait(lambda state: state["shortcut_status"] == "ready")
            portal.release()
            wait(lambda state: state["shortcut_status"] == "unavailable")
            portal.version = 1
            portal.own()
            wait(lambda state: state["shortcut_status"] == "error" and "version 2" in state["shortcut_error"])
            call("Shutdown")
            assert process.wait(timeout=5) == 0
            print("Portal setup, cancellation, repeat, press/release, reconfiguration, restart and version checks passed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stderr.close()


if __name__ == "__main__":
    main()
