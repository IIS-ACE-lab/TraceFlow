# 01 · Setting up the `sca` environment

The `sca` conda environment runs everything TraceFlow does on the host:
the Makefile, the orchestration scripts, `make judge` and `make graph`.
Training runs in containers (see [02](02_container_runtime_and_image_build.md)).

## The normal path

```bash
cd "$(git rev-parse --show-toplevel)"
bash services/side_channel_core/scripts/setup.sh              # conda env only
bash services/side_channel_core/scripts/setup.sh --sys-deps   # also apt packages (asks for sudo)
conda activate sca
```

What happens, in order:

| Step | Done by | What |
|---|---|---|
| 1 | `scripts/setup.sh` | Checks `conda` is on `PATH`; makes `scripts/*.sh`, `infra/environment/*.sh`, `ci_cd/*.sh` executable |
| 2 | `infra/environment/setup_sca_env.sh` | With `--sys-deps`: `apt-get install` of build tools, HDF5 tools, `jq`, `gettext`, … (Ubuntu 24.04 package names) |
| 3 | 〃 | Accepts the Anaconda Terms of Service for the `main` and `r` channels |
| 4 | 〃 | `conda create -n sca python=3.12 pip`; sets `KERAS_BACKEND=torch` in the env (as in the image) |
| 5 | 〃 | PyTorch 2.9.1, torchvision 0.24.1, torchaudio 2.9.1 from the CUDA 12.8 index (`whl/cu128`, as in the training image) |
| 6 | 〃 | `pip install -r infra/environment/requirements.txt` (pinned: numpy, keras 3.12.4, optuna, clearml, pytest, pylint, …) |
| 7 | 〃 | Saves `infra/environment/environment_dev.yaml` and `pip_freeze_dev.txt` |
| 8 | 〃 | Prints the Python and PyTorch versions and whether PyTorch sees the GPU |
| 9 | 〃 | `sudo chown`/`chmod` of `infra/environment/` back to your user |
| 10 | 〃 | Installs the TraceFlow Monitor, only when the editor is VS Code (see 1.15) |

The host env and the training image (`ubuntu:24.04` with PyTorch's cu128 wheels, which bring their own CUDA libraries)
both run Python 3.12. `infra/environment/requirements.yaml` describes the same
environment for `conda env create -f`.

Expect 15–40 minutes, mostly downloads.

The files, in `services/side_channel_core/infra/environment/`:

| File | What |
|---|---|
| `setup_sca_env.sh` | what `setup.sh` runs: Python 3.12, PyTorch 2.9.1 for CUDA 12.8, `requirements.txt`; saves `environment_dev.yaml` and `pip_freeze_dev.txt`, reports whether PyTorch sees the GPU, installs the panel |
| `requirements.txt` | the pinned packages (plus pytest, pylint, pyflakes); the training image installs the same file |
| `requirements.yaml` | the same environment as a conda file |

## Check it worked

```bash
conda activate sca
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import yaml, h5py, rich, optuna, clearml; print('ok')"
make -n graph ENV=dev GRAPH_ID=1        # dry run: Makefile parses
```

`torch.cuda.is_available()` must print `True` for GPU work on the host.
Training itself uses the container's own PyTorch.

---

## Problems

### 1.1 `[SETUP] conda not found`

`conda` is not on `PATH`. Install Miniconda, then open a new shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
```

If conda is installed but not found in a non-login shell, run
`eval "$(~/miniconda3/bin/conda shell.bash hook)"` first.

### 1.2 `CondaToSNonInteractiveError` / Terms of Service not accepted

The script accepts the ToS for `repo.anaconda.com/pkgs/main` and `pkgs/r`.
Older conda versions have no `conda tos` command; the step then prints an
error and continues (`|| true`). If `conda create` still refuses:

```bash
conda update -n base conda
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 1.3 Running the script again deletes my `sca` environment

`conda create -n sca -y` replaces an existing `sca` environment without
asking. To keep an environment, rename it before re-running:

```bash
conda create -n sca_backup --clone sca
```

To only add a missing package, do not re-run the script:
`conda activate sca && pip install <package>`.

### 1.4 The script stops at `Do you want to overwrite them? [y/N]`

Step 7 found `infra/environment/environment_dev.yaml` or
`pip_freeze_dev.txt` from an earlier run. Answer `y` to refresh them, or
Enter to keep them. The prompt needs a terminal; for an unattended run,
delete the two files first or pipe the answer:

```bash
yes n | bash services/side_channel_core/scripts/setup.sh
```

### 1.5 `torch.cuda.is_available()` is `False`

| Check | Command | Expected |
|---|---|---|
| Driver works | `nvidia-smi` | a GPU table |
| Driver is new enough for CUDA 12.8 | `nvidia-smi` header | `CUDA Version: 12.8` or higher |
| CUDA wheel installed | `python -c "import torch; print(torch.version.cuda)"` | `12.8`, not `None` |

If `torch.version.cuda` is `None`, a CPU wheel was installed (usually by a
later `pip install` pulling `torch` from PyPI). Reinstall the three CUDA wheels only;
`--no-deps` keeps the pins of `requirements.txt` (without it, `--force-reinstall` also replaces
numpy, pillow and Jinja2 with their newest versions):

```bash
pip install --force-reinstall --no-deps \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

If the driver is older than CUDA 12.8, update the driver (the training image
needs it too), or use a matching wheel index on the host only (for example
`.../whl/cu126`): edit `CUDA_TORCH_INDEX` at the top of
`infra/environment/setup_sca_env.sh`.

### 1.6 `No module named 'tensorflow'` or `'onnxruntime'`

The environment no longer installs TensorFlow or ONNX Runtime: TraceFlow
imports neither. Only the upstream ANSSI reference scripts
`data/ASCAD/ASCAD_train_models.py` and `ASCAD_test_models.py` import
TensorFlow; install it yourself (in a separate env) to run them.

### 1.7 pip dependency conflicts or `ResolutionImpossible`

`infra/environment/requirements.txt` pins every package TraceFlow imports
with `==` (each has a CPython 3.12 wheel); pip resolves their own
dependencies. A conflict usually comes from a package installed by hand
afterwards. Find the package in the error, change its pin in
`requirements.txt` (the training image installs the same file), and re-run
only that install:

```bash
conda activate sca
pip install -r services/side_channel_core/infra/environment/requirements.txt
```

### 1.8 Files named `=2.19.0` or `=1.17.3` appear in the repository

An old version of `setup_sca_env.sh` ran `pip install tensorflow>=2.19.0`
unquoted, which the shell reads as a redirection to a file called
`=2.19.0`. The current script quotes these. Delete the stray files:

```bash
find . -maxdepth 4 -name '=*' -type f -print -delete
```

### 1.9 `pip` installs into `~/.local` instead of the environment

pip found the environment's `site-packages` not writable. See
[03 · File permissions](03_file_permissions.md#32-stop-pip-from-trying---user-inside-conda).

### 1.10 `apt-get` fails with `--sys-deps`

`Could not get lock /var/lib/dpkg/lock-frontend`: another apt process
(often unattended-upgrades) is running. Wait for it, or:

```bash
sudo systemctl stop unattended-upgrades
bash services/side_channel_core/scripts/setup.sh --sys-deps
```

Missing packages (`E: Unable to locate package`) mean the apt sources are
out of date or not Ubuntu; run `sudo apt-get update` and install the
failing package by hand.

### 1.11 `sudo` prompts at the end of the script

Step 9 runs `sudo chown -R` on `services/side_channel_core/infra/environment`
so the environment files are owned by you, not root. Enter your password, or
press Ctrl-C — the environment is already complete at that point.

### 1.12 `make` fails with `No module named yaml` or `python: command not found`

The Makefile calls `python` from the active shell. Activate the
environment in every new terminal:

```bash
conda activate sca
```

### 1.13 Recreating the environment from the saved files

```bash
conda env create -n sca -f services/side_channel_core/infra/environment/requirements.yaml
# or the exact export of an earlier run:
conda env create -n sca -f services/side_channel_core/infra/environment/environment_dev.yaml
# or, into an existing Python 3.12 env (one resolution: every pin installed once):
pip install -r services/side_channel_core/infra/environment/requirements.txt \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

### 1.14 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2` / `module 'numpy' has no attribute 'trapezoid'`

TraceFlow needs NumPy >= 2.2: the `auc_ge` and `auc_sr` metrics call
`np.trapezoid`, which NumPy 1 does not have.
`services/side_channel_core/infra/environment/requirements.txt` pins
`numpy==2.2.6` with NumPy-2 builds of h5py, scikit-learn, pandas and
matplotlib. An environment built with NumPy 1 keeps compiled packages that no
longer import once NumPy is upgraded; reinstall the pinned set:

```bash
conda activate sca
pip install -r services/side_channel_core/infra/environment/requirements.txt
python -c "import numpy, h5py, sklearn, matplotlib; print(numpy.__version__)"
```

Old NumPy-1 builds of TensorFlow, `ml-dtypes`, `shap` or `tables` left in an
old environment also fail to import under NumPy 2; TraceFlow imports none of
them. Recreate the environment (1.13) to get rid of them.

### 1.15 The TraceFlow Monitor was not installed

The TraceFlow Monitor is a side panel for the VS Code editor only. The setup
installs it only when VS Code's `code` CLI is reachable (a VS Code terminal,
or `~/.vscode-server` on a remote host); it skips Cursor, VSCodium, Windsurf
and other editors, and prints how to install it later. From a VS Code
terminal:

```bash
conda activate sca
make panel ENV=dev        # then: Developer: Reload Window
```
