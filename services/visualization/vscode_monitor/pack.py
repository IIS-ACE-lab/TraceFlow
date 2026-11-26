# services/visualization/vscode_monitor/pack.py
"""
pack.py
=======
Builds the TraceFlow Monitor's .vsix with the standard library. A .vsix is a
zip holding a content-types file, a manifest and the extension under
extension/. No npm, no vsce. The Python helpers (layout.py, yamledit.py)
are not packed: the extension runs them from the repository it watches.

    python -m services.visualization.vscode_monitor.pack OUT_DIR    prints the .vsix path

Author: Shraddha Mishra
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
FILES = ("package.json", "extension.js", "state.js", "gpu.js", "history.js", "commands.js", "README.md",
         "webview/panel.html", "webview/panel.css", "webview/panel.js", "webview/menus.js",
         "webview/planner.html", "webview/planner.css", "webview/planner.js", "webview/planner_core.js",
         "syntaxes/traceflow-log-arrows.tmLanguage.json")   # the epoch blocks' arrows coloured in .log files
# The icons come from brand/, the one source of the TraceFlow mark (make_brand.py), packed at the
# paths package.json names: the Activity Bar icon and the Extensions view's icon.
BRAND = HERE / "brand"
ICONS = {"media/traceflow.svg": BRAND / "traceflow.svg", "media/icon.png": BRAND / "traceflow-icon.png",
         # the status page's button on the TraceFlow Monitor title line (package.json traceflow.share)
         "media/share-dark.png": BRAND / "traceflow-share-dark.png", "media/share-light.png": BRAND / "traceflow-share-light.png"}

CONTENT_TYPES = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension=".json" ContentType="application/json"/>
<Default Extension=".js" ContentType="application/javascript"/>
<Default Extension=".html" ContentType="text/html"/>
<Default Extension=".css" ContentType="text/css"/>
<Default Extension=".md" ContentType="text/markdown"/>
<Default Extension=".svg" ContentType="image/svg+xml"/>
<Default Extension=".png" ContentType="image/png"/>
<Default Extension=".vsixmanifest" ContentType="text/xml"/>
</Types>
"""


def manifest(pkg: dict) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011" xmlns:d="http://schemas.microsoft.com/developer/vsx-schema-design/2011">
  <Metadata>
    <Identity Language="en-US" Id="{escape(pkg['name'])}" Version="{escape(pkg['version'])}" Publisher="{escape(pkg['publisher'])}"/>
    <DisplayName>{escape(pkg['displayName'])}</DisplayName>
    <Description xml:space="preserve">{escape(pkg['description'])}</Description>
    <Icon>extension/{escape(pkg['icon'])}</Icon>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{escape(pkg['engines']['vscode'])}"/>
    </Properties>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code"/>
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true"/>
    <Asset Type="Microsoft.VisualStudio.Services.Icons.Default" Path="extension/{escape(pkg['icon'])}" Addressable="true"/>
  </Assets>
</PackageManifest>
"""


def pack(out_dir: Path) -> Path:
    pkg = json.loads((HERE / "package.json").read_text())
    out = Path(out_dir) / f"{pkg['name']}-{pkg['version']}.vsix"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("extension.vsixmanifest", manifest(pkg))
        for f in FILES:
            z.write(HERE / f, f"extension/{f}")
        for packed, source in ICONS.items():
            z.write(source, f"extension/{packed}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m services.visualization.vscode_monitor.pack OUT_DIR")
    print(pack(Path(sys.argv[1])))
