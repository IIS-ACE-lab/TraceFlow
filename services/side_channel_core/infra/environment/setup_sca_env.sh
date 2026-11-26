#!/bin/bash
# services/side_channel_core/infra/environment/setup_sca_env.sh
#
# Creates the host `sca` conda environment: Python 3.12 (the training image
# runs Ubuntu 24.04's Python 3.12), PyTorch 2.9.1 for CUDA 12.8 (the image
# uses the same cu128 wheels) and the pins of requirements.txt next to this
# file. Entry point: services/side_channel_core/scripts/setup.sh.
#
# Usage: bash services/side_channel_core/infra/environment/setup_sca_env.sh [--sys-deps]
#   --sys-deps   also apt-installs system packages (sudo; Ubuntu 24.04 names)

set -e

# Resolve paths from this file, so the script works from any directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

ENV_NAME="sca"
PYTHON_VERSION="3.12"
CUDA_TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PINS="torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
INSTALL_SYS_DEPS=false

# Check if --sys-deps flag is passed
if [[ "$1" == "--sys-deps" ]]; then
    INSTALL_SYS_DEPS=true
fi

# Step 0: Optional system packages installation
if [ "$INSTALL_SYS_DEPS" = true ]; then
    echo "Installing system dependencies..."
    sudo apt-get update || { sleep 30; sudo apt-get update; }

    sudo apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        build-essential libcurl4-openssl-dev libcares2 hdf5-tools gettext jq \
        bzip2 ca-certificates curl expat git libxml2 openssl \
        sqlite3 libsqlite3-dev tk zstd \
    || (
        echo "Retrying system dependency install with --fix-missing..."
        sleep 10
        sudo apt-get update && \
        sudo apt-get install -y --fix-missing --no-install-recommends \
            python3 python3-pip python3-venv python3-dev \
            build-essential libcurl4-openssl-dev libcares2 hdf5-tools gettext jq \
            bzip2 ca-certificates curl expat git libxml2 openssl \
            sqlite3 libsqlite3-dev tk zstd
    )
    echo "System dependencies installed."
fi

# Step 1: Accept Anaconda Terms of Service if required
echo "Accepting Anaconda Terms of Service (if required)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Step 2: Create Conda environment
echo "Creating Conda environment: $ENV_NAME (Python $PYTHON_VERSION)..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" pip -y

# Keras 3 runs on PyTorch, as in the training image (ENV KERAS_BACKEND=torch)
conda env config vars set -n "$ENV_NAME" KERAS_BACKEND=torch

# Activate Conda env properly
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Step 3: PyTorch for CUDA 12.8 (same wheels as the training image) and the pins of
# requirements.txt, in one pip resolution (as the Dockerfile and requirements.yaml): each pin is
# installed once at its version. The +cu128 versions exist only on the PyTorch index.
echo "Installing PyTorch ($TORCH_PINS) from $CUDA_TORCH_INDEX and $REQUIREMENTS..."
python -m pip install --upgrade pip
# shellcheck disable=SC2086
python -m pip install -r "$REQUIREMENTS" $TORCH_PINS --extra-index-url "$CUDA_TORCH_INDEX"

# Step 5: Save environment metadata next to this script.
# requirements.txt is the hand-kept pin list; the pip freeze goes to its own file.
echo "Saving environment files to $SCRIPT_DIR..."
ENV_FILE="$SCRIPT_DIR/environment_dev.yaml"
FREEZE_FILE="$SCRIPT_DIR/pip_freeze_dev.txt"

if [[ -f "$ENV_FILE" || -f "$FREEZE_FILE" ]]; then
    echo "Environment files already exist:"
    echo " - $ENV_FILE"
    echo " - $FREEZE_FILE"
    read -p "Do you want to overwrite them? [y/N]: " choice
    case "$choice" in
        y|Y )
            conda env export > "$ENV_FILE"
            python -m pip freeze > "$FREEZE_FILE"
            echo "Overwritten environment files."
            ;;
        * )
            echo "Skipped overwriting environment files."
            ;;
    esac
else
    conda env export > "$ENV_FILE"
    python -m pip freeze > "$FREEZE_FILE"
    echo "Environment files saved."
fi

# Step 6: Verify GPU access
echo "Verifying GPU availability..."
python - <<EOF
import torch
print("Python:", __import__("sys").version.split()[0])
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("PyTorch CUDA Available:", torch.cuda.is_available())
EOF

sudo chown -R "$(id -un):$(id -gn)" "$SCRIPT_DIR" && \
sudo chmod -R u+rwX "$SCRIPT_DIR" && \
ls -ld "$SCRIPT_DIR" && ls -l "$SCRIPT_DIR/"

echo "Environment '$ENV_NAME' setup complete and GPU-ready!"

# Step 7: TraceFlow Monitor, a side panel for the VS Code editor only.
# Installed only when the editor is VS Code (its `code` CLI is reachable);
# never for Cursor, VSCodium, Windsurf or other VS Code forks. Never stops the setup.
# Later, from a VS Code terminal: make panel ENV=dev
PANEL_LATER="Install the TraceFlow Monitor (VS Code only) later from a VS Code terminal: make panel ENV=dev"
CODE_CLI="$(command -v code || true)"
if [[ -z "$CODE_CLI" ]]; then
    CODE_CLI="$(ls -dt "$HOME"/.vscode-server/cli/servers/Stable-*/server/bin/remote-cli/code 2>/dev/null | head -1 || true)"
fi
CODE_REAL="$(readlink -f "$CODE_CLI" 2>/dev/null || echo "$CODE_CLI")"
case "${CODE_REAL,,}" in
    *cursor*|*codium*|*windsurf*|*trae*|*positron*) CODE_CLI="" ;;
esac
if [[ -n "$CODE_CLI" ]] && "$CODE_CLI" --version >/dev/null 2>&1; then
    echo "Installing the TraceFlow Monitor into VS Code ($CODE_CLI)..."
    CODE="$CODE_CLI" PYTHON="$(command -v python)" bash "$REPO_DIR/services/visualization/vscode_monitor/install.sh" \
        || echo "TraceFlow Monitor not installed. $PANEL_LATER"
else
    echo "VS Code CLI not found (the TraceFlow Monitor is for VS Code only; other editors are skipped)."
    echo "$PANEL_LATER"
fi
