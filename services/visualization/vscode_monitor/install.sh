#!/usr/bin/env bash
# services/visualization/vscode_monitor/install.sh
# Installs the TraceFlow Monitor into this machine's VS Code. Never published.
# Run it from a VS Code terminal (that is where `code` reaches the editor),
# re-run after changing the extension, then reload the window.
#
# Usage: bash services/visualization/vscode_monitor/install.sh     (or: make panel)
# Author: Shraddha Mishra
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
CODE="${CODE:-$(command -v code || true)}"
if [[ -z "$CODE" ]]; then
    CODE="$(ls -dt "$HOME"/.vscode-server/cli/servers/Stable-*/server/bin/remote-cli/code 2>/dev/null | head -1 || true)"
fi
[[ -n "$CODE" ]] || { echo "[TRACEFLOW] VS Code CLI not found; set CODE=/path/to/code" >&2; exit 1; }
# The TraceFlow Monitor is built for the VS Code editor only; other editors' `code` CLIs are refused.
case "$(readlink -f "$CODE" 2>/dev/null || echo "$CODE")" in
    *cursor*|*Cursor*|*codium*|*VSCodium*|*windsurf*|*Windsurf*|*trae*|*Trae*|*positron*|*Positron*)
        echo "[TRACEFLOW] $CODE is not VS Code; the TraceFlow Monitor panel is for VS Code only" >&2; exit 1 ;;
esac

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
VSIX="$(cd "$ROOT_DIR" && PYTHONPATH="$ROOT_DIR" "$PYTHON" -m services.visualization.vscode_monitor.pack "$OUT")"
"$CODE" --install-extension "$VSIX" --force
echo "[TRACEFLOW] installed $(basename "$VSIX"); now run 'Developer: Reload Window'"
