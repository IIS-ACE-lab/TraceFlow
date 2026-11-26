#!/usr/bin/env bash
# services/side_channel_core/scripts/setup.sh
#
# One-command host setup for TraceFlow. Creates the `sca` conda environment
# that runs the Makefile, the orchestration scripts, the graph manager and
# the model judge on the host (training itself runs in containers).
#
# Usage (from anywhere):
#   bash services/side_channel_core/scripts/setup.sh              # conda env only
#   bash services/side_channel_core/scripts/setup.sh --sys-deps   # + apt packages (sudo)
#
# The environment (Python 3.12, PyTorch cu128, infra/environment/requirements.txt)
# is installed by infra/environment/setup_sca_env.sh; this wrapper makes the
# repository's shell scripts executable first and forwards all arguments.
#
# Author: Shraddha Mishra

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCC_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "[SETUP] conda not found. Install Miniconda or Anaconda first:" >&2
    echo "        https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi

echo "[SETUP] Making TraceFlow shell scripts executable..."
chmod +x "$SCC_DIR"/scripts/*.sh "$SCC_DIR"/infra/environment/*.sh
if compgen -G "$SCC_DIR/ci_cd/*.sh" >/dev/null; then
    chmod +x "$SCC_DIR"/ci_cd/*.sh
fi

echo "[SETUP] Creating the 'sca' conda environment (infra/environment/setup_sca_env.sh)..."
exec bash "$SCC_DIR/infra/environment/setup_sca_env.sh" "$@"
