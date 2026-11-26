# services/doc

Troubleshooting guides and the command-line reference for TraceFlow. TraceFlow
is used from the TraceFlow panel in VS Code (root [README](../../README.md),
§1); these guides are for what the panel cannot fix, numbered in the order you
usually meet the problems: set up the host, the container runtime,
permissions, then runs, judging and figures. Guides 08 and 09 are the panel's
steps as `make` commands.

| # | Guide | Covers |
|---|---|---|
| 01 | [Setting up the `sca` environment](01_setting_up_the_sca_environment.md) | `scripts/setup.sh` and `infra/environment/setup_sca_env.sh`: conda, Python 3.12, Terms of Service, cu128 wheels, pinned requirements, re-runs, prompts, apt, GPU checks, NumPy 2 upgrades, VS Code-only panel |
| 02 | [Container runtime and image build](02_container_runtime_and_image_build.md) | containerd inactive, nerdctl/BuildKit, DNS, GPU in the container, zombie containers, disk space, rebuilds |
| 03 | [File permissions](03_file_permissions.md) | root-owned artifacts, pip `--user`, bind mounts, script permissions, stale lock files |
| 04 | [Running experiments](04_running_experiments.md) | `ENV`/`HY`/`PROJECT`, DONE trials and `FORCE`, `DATASET`/`MODEL_ID`, sweep size, key enumeration vs cross-validation, data errors, OOM, logs, `make test` and snapshot checks, `weight_init`, rejected cross-validation settings, `ConfigError` for a wrong key in any configuration file |
| 05 | [Model judge and graphs](05_model_judge_and_graphs.md) | `make judge` and `make graph` errors, missing metrics and models, job validation, models by name or hash, cross-validation fold checks |
| 06 | [Python imports and tests](06_python_imports_and_tests.md) | `PYTHONPATH`, unit tests, lint, stale code in containers |
| 07 | [Git and GitHub](07_git_and_github.md) | SSH keys, remotes, syncing the self-hosted and GitHub remotes |
| 08 | [TraceFlow from the command line](08_command_line.md) | every panel button as its `make` command, tmux by hand, a first run, every `make` target and variable |
| 09 | [DLSCA_ASCAD from the command line](09_dlsca_ascad_command_line.md) | the ASCAD reproduction's commands: clean slate, detached runs, the commands of every figure and of Experiments 1–27, ship and reproduce |
| 10 | [To the developers](10_to_the_developers.md) | Changing the system safely: configuration keys and `utils/config_schema.py`, new config files, the run hash, `ConfigError`, dataset types, the blueprint, dependencies and the image, the panel, the checks before a merge |
| 11 | [The status page](11_phone_status_share.md) | the runs on a phone and links to share them: start, add a person, home screen, withdraw, who can open the links (same network, VPN, public, SSH only), a link that does not open, what a link can see |
| — | [Tutorial: training every dataset type](TUTORIAL.md) | ASCADf, ASCADr, CHES_CTF_2018, AES_RD, DPAv4, AES_HD: fetching, data entries, `sca.attack_byte`, cross-validation, models that recover the key, reading the runs, the 2026-10-18 verification |
| — | [Repository tree](../../Traceflow_tree.md) | Every directory and file with a one-line purpose |

## Other files

```text
services/doc/
├── README.md               # this index
├── 01_…11_*.md             # the guides above
├── TUTORIAL.md             # training every dataset type
└── TraceFlow_tree.md       # older copy of the repository tree; current one: ../../Traceflow_tree.md
```

## Depends on / used by

Documentation only. Linked from the root [README](../../README.md).

## TODOs

None.
