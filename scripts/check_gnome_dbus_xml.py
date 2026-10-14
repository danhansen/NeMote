#!/usr/bin/env python3
"""Check that GNOME clients use the shared protocol and known settings."""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extensions/gnome-shell/nemote@dhansen.dev"


def prefs_import_errors(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    module_constants, separator, _rest = text.partition("const NeMotePage")
    if not separator:
        return [f"{path.name} is missing NeMotePage"]
    if re.search(r"\b_\s*\(", module_constants):
        return [f"{path.name} calls gettext while importing; translate labels after initialization"]
    return []


def main() -> int:
    # XML and GSettings schema are the specifications, rather than duplicate
    # declarations embedded in each client or another hand-written type table.
    protocol = ET.parse(EXTENSION / "service.xml").getroot()
    if protocol.find("interface") is None:
        raise RuntimeError("D-Bus XML does not contain an interface")
    schema = ET.parse(EXTENSION / "schemas/org.gnome.shell.extensions.nemote.gschema.xml")
    keys = {key.attrib["name"] for key in schema.findall("./schema/key")}
    shared = (EXTENSION / "serviceSettings.js").read_text(encoding="utf-8")
    errors = []
    for name in ("extension.js", "preferencesPage.js"):
        path = EXTENSION / name
        text = path.read_text(encoding="utf-8")
        for module in ("serviceClient", "serviceSettings"):
            if f"from './{module}.js'" not in text:
                errors.append(f"{name} must use shared {module}")
        # The retired overlay key stays in the schema for older Shell processes
        # during an upgrade, but must not drive any current UI behavior.
        missing = sorted(key for key in keys - {"show-overlay", "shortcut-capture-active", "toggle-shortcut"}
                         if f"'{key}'" not in text + shared)
        if missing:
            errors.append(f"{name} does not reference settings: {', '.join(missing)}")
    errors.extend(prefs_import_errors(EXTENSION / "preferencesPage.js"))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
