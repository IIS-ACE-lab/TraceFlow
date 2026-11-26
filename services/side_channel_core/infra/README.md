# services/side_channel_core/infra
Host environment installer, Python dependency lists and machine-local secrets. `environment/setup_sca_env.sh` creates the host `sca` conda environment (Python 3.12, PyTorch 2.9.1 from the CUDA 12.8 index, the pins of `environment/requirements.txt`) used by the Makefile, orchestration helpers, graph manager and model judge; training itself runs in containers, whose image installs the same `requirements.txt` on Ubuntu 24.04's Python 3.12. `scripts/setup.sh` is the entry point that calls it.

## Contents
```text
infra/
├── environment/
│   ├── setup_sca_env.sh      # create `sca` conda env; --sys-deps also apt-installs system packages (sudo)
│   ├── requirements.txt      # pinned pip packages (Python 3.12); host env and deployment/docker/Dockerfile
│   ├── requirements.yaml     # conda env spec: name sca, python=3.12, pip -r requirements.txt + torch cu128, KERAS_BACKEND=torch
│   ├── system_conda.txt      # conda package names of the old Python 3.10 env; not used by scripts
│   ├── environment_dev.yaml  # output of setup_sca_env.sh (conda env export), when it has run
│   └── pip_freeze_dev.txt    # output of setup_sca_env.sh (pip freeze), when it has run
├── secrets_this_machine/     # WireGuard key pair of this host (names only, see its README)
├── secrets_other_machine/    # peer-host secrets (currently empty)
├── .empty/                   # empty placeholder directory
└── README.md                 # this file
```

## Used by
- `scripts/setup.sh` (`exec bash infra/environment/setup_sca_env.sh "$@"`; also `chmod +x infra/environment/*.sh`).
- `environment/requirements.txt`: `deployment/docker/Dockerfile`, `environment/setup_sca_env.sh`, `environment/requirements.yaml`.
- `secrets_*`: no code references them; WireGuard setup is described in `deployment/docker/README.md`.

## TODOs
None.

## Usage

```bash
# from the repository root
bash services/side_channel_core/scripts/setup.sh [--sys-deps]
# or the installer itself
bash services/side_channel_core/infra/environment/setup_sca_env.sh [--sys-deps]
# or the conda file
conda env create -f services/side_channel_core/infra/environment/requirements.yaml
```

PyTorch is not in `requirements.txt`: its CUDA 12.8 wheels
(`torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128`) come from the cu128
index in the same pip call as `requirements.txt` (`--extra-index-url`), so one resolution
installs every pin once. Same pins and call in `setup_sca_env.sh`, `requirements.yaml` and the
Dockerfile.

The TraceFlow Monitor (`services/visualization/vscode_monitor/`) is for the
VS Code editor only. `setup_sca_env.sh` installs it only when VS Code's
`code` CLI is reachable and skips other editors; later, from a VS Code
terminal: `make panel ENV=dev`.

Troubleshooting: `services/doc/01_setting_up_the_sca_environment.md`.
