# 08 · TraceFlow from the command line

Every button of the TraceFlow panel runs a `make` target from the repository
root, inside the `sca` conda environment. This file lists those targets, for
when VS Code is not at hand (a plain SSH terminal, a script), for the few steps
the panel has no button for, and for checking what a button did. The panel
tutorial is §1 of the root [README](../../README.md); the ASCAD project's own
commands are in [09_dlsca_ascad_command_line.md](09_dlsca_ascad_command_line.md).

```bash
conda activate sca               # every target below runs in it
cd TraceFlow                     # the repository root
```

## 1. Panel buttons and their commands

| Panel | Command |
|---|---|
| Run Single Experiment → Start | `make orchestrate ENV=… PROJECT=… DATASET=… MODEL_ID=…` (that line of `config/single_runs.yaml`) |
| Queues → Start | `make experiment ENV=… PROJECT=… QUEUE=<key> [FORCE=1]` |
| Queues → Log | `tail -f projects/<P>/artifacts/<ENV>/tmux_logs/<key>.txt` |
| Test (recorded) | `make test ENV=… PROJECT=… MODEL=<run>` (else the models of `config/test_config.yaml`) |
| Test (external) | `make test ENV=… PROJECT=… MODEL=<run> EXTERNAL=1` |
| Configs → model_judge.yaml → Judge | `make judge ENV=… PROJECT=…` |
| Configs → model_filter.yaml → Find models | `make find_model ENV=… PROJECT=…` |
| Configs → graph_config.yaml → Draw | `make graph ENV=… PROJECT=… GRAPH_ID=<id>` |
| Attach / Detach | `tmux attach -t <session>` / `tmux detach-client -s <session>` |
| Container log | `sudo nerdctl logs -f --tail 100 sidechannel` |
| Kill session | `tmux kill-session -t <session>` |
| Terminate Run | `tmux kill-session -t <session>`, then `make stack-down ENV=…` |
| Housekeeping → DELETE | `make delete_database ENV=… PROJECT=… MODEL=<names> JSON=1` (the dry run), then the same with `CONFIRM=1` |
| SNAPSHOTS → Scan | `make snapshots ENV=… PROJECT=…` |
| SNAPSHOTS → Reproduce | `make reproduce ENV=… PROJECT=… MODEL=<name> BATCH=<batch> [REPRODUCE_ROOT=<folder>] [FORCE=1]` |
| params.yaml → model → Import a Model → Import now | `make import ENV=… PROJECT=… [MODEL=<names>] [FORCE=1]` (the models of `config/import_config.yaml`) |
| Docker tear down | `make docker-teardown ENV=…` |
| Share icon → Start / Stop | `make status_share ENV=dev` (master.yaml `status_share` host and port) in tmux `tf-status` / `tmux kill-session -t tf-status` |
| Share icon → Add a person / Withdraw | `python -m services.visualization.server.status share add NAME` / `share remove NAME` (`share list` shows the tokens; the link is `http://<HOST>:<PORT>/?t=<token>`) |
| (install the panel) | `make panel ENV=dev`, then *Developer: Reload Window* |
| (a new project's config files) | `make scaffold ENV=dev` |

The panel starts long runs in the tmux session `tf-<project>` and runs
`sudo chown -R "$(id -u):$(id -g)" projects/` first (the container writes
there as root). From a terminal, do the same:

```bash
tmux new -s fig26                                  # a session survives a closed SSH connection
sudo chown -R "$(id -u):$(id -g)" projects
mkdir -p projects/DLSCA_ASCAD/artifacts/dev/tmux_logs
make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26 2>&1 | tee >(sed -u -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' -e 's/.*\r//' >> projects/DLSCA_ASCAD/artifacts/dev/tmux_logs/fig26.txt)
# detach: Ctrl-b, then d; back: tmux attach -t fig26
```

No button: `make batch_label`, `make graph FACE_ID=`, `make rerun`,
`make rerun-judged`, `make build-image`, `make run-stack`, `make stack-nuke`,
`make healthcheck`, `make reset`, `make lint`.

## 2. A first run

The reference project reproduces Benadjila et al., *Deep learning for
side-channel analysis and introduction to ASCAD database* (JCEN 2020); its
guide is [`projects/DLSCA_ASCAD/README.md`](../../projects/DLSCA_ASCAD/README.md).

```bash
conda activate sca
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_CNNbest   # train + test
make judge       ENV=dev PROJECT=DLSCA_ASCAD                                        # select the best model
make graph       ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=1                             # draw a figure; GRAPH_ID=0 plots the top_models of make judge
```

Shortlist the sweep lists in `projects/DLSCA_ASCAD/config/params.yaml`
before the first run: every list is a sweep dimension, and the file as
shipped lists the paper's full benchmark ranges.

## 3. Make targets and variables

Run every target from the repository root with the `sca` environment
active. `ENV` is required for all of them.

### 3.1 Targets

| Target | What it does |
|---|---|
| `orchestrate` | Stops the old stack, builds the training image (skipped if it exists), starts the container stack and runs every pending trial of the project's sweep: data, training, test, key recovery, registries. Runs the health check and GPU reset on success; keeps volumes on failure |
| `rerun` | Re-runs a finished experiment by model name, `REPEAT` times |
| `test` | Re-tests trained models by name (`MODEL=` or `test_request.model` in `projects/<P>/config/test_config.yaml`) through `core/cli.py test-cmd`, each on the exact parameters it was trained with; the result replaces the model's recorded test: its test metric row, `metrics/key_recovery/test.json` and `artifacts/test_registry.json`. With `EXTERNAL=1`: an external test with `test_config.yaml`'s `perform_attack` settings; reports in `<run>/test/<model_hash>_et_<index>/`, the outcome under `external_tests` → `<model_hash>_et_<index>` of the model's `test_registry.json` entry, the recorded test untouched |
| `judge` | Ranks the project's models with `config/model_judge.yaml`; writes `model_selection/` and job 0's `top_models` |
| `find_model` | Lists the trained models that have given parameters (`config/model_filter.yaml`: architecture, optimizer, learning rate, layers, dataset, …), from `run_map.jsonl`; writes `artifacts/filtered_models.yaml` for graph jobs. Metrics are not involved |
| `rerun-judged` | Re-runs the candidates the judge selected (`model_selection/latest_selection.json`) |
| `delete_database` | Deletes the runs of one environment of one project, optionally only those of a dataset (`DATASET=`) or named models (`MODEL=`): registries, run map, model names, `artifacts/<ENV>/generated_models/<model>`, and the `make find_model` / `make judge` outputs. Shared files and directories go only when every run they hold is deleted: when no run of any environment or dataset is left, `artifacts/`, `model_selection/` and `graphs/` are removed completely; otherwise emptied registries are deleted and `graphs/` is kept. Dry run unless `CONFIRM=1` |
| `graph` | Draws one job of `config/graph_config.yaml` into `projects/<P>/graphs/` |
| `verify` | Checks the seal of every artifact of the project (`artifacts/`), or of the runs `MODEL=` names, per run folder; exit 1 when a file was changed outside TraceFlow ([tamper-evident artifacts](../../projects/DLSCA_ASCAD/README.md#tamper-evident-artifacts)) |
| `seal` | Signs, once, the files of runs written before seals existed (same scope as `verify`); a file whose seal is broken is reported, never re-signed |
| `seal-key` | Creates the signing key if missing (`TRACEFLOW_SEAL_DIR`, default `~/.config/traceflow/seal`) and prints its id; `run-stack` runs it first and mounts the key read-only into the container |
| `build-image` | Builds the training image only |
| `run-stack` | Starts the compose stack without building |
| `stack-down` | Stops the stack, keeps volumes |
| `stack-nuke` | Stops the stack and deletes its volumes (destructive) |
| `docker-clean` | Same as `stack-nuke`, plus the orchestrator lock file |
| `docker-teardown` | `docker-clean`, then removes the `sidechannel` containers and images, prunes the build cache, dangling images and unused volumes of the namespace, empties `deployment/docker/tmp/` (the panel's Docker tear down) |
| `healthcheck` | Runs `scripts/healthcheck.sh` |
| `reset` | Resets the GPU (`scripts/gpu_reset.sh`) |
| `lint` | Pylint over `side_channel_core` |
| `preflight-rootful` | Checks that rootful containerd is active (run by `orchestrate`, `rerun` and `test`) |
| `panel` | Builds the TraceFlow Monitor (`services/visualization/vscode_monitor/`) and installs it into this machine's VS Code; run it from a VS Code terminal, then reload the window |
| `scaffold` | Copies the blueprint config files every project of `master.yaml` lacks into `projects/<P>/config/` (never overwrites) |
| `status_share` | Serves the read-only status page on this machine's address (found by itself, or master.yaml `status_share.host`) and `status_share.port` (`HOST=` / `PORT=` override; off when the block is absent or `enabled: false`) until Ctrl-C (`services/visualization/server/status.py`) |

### 3.2 Variables

Append them to any target: `make <target> ENV=dev PROJECT=DLSCA_ASCAD FORCE=1`.

| Variable | Values | Effect |
|---|---|---|
| `ENV` | `dev` \| `staging` \| `prod` | **Required.** Selects `config/<env>.yaml`, the artifact tree and the image policy (`prod` pushes an immutable tag) |
| `PROJECT` | a project `name` in `master.yaml` | Project to run; defaults to `default_project` in `master.yaml` |
| `DATASET` | a `data[].name` of the project | Dataset when `params.yaml` lists several; default `ascad` |
| `MODEL_ID` | a `model[].model_id` | Architecture when `params.yaml` lists several |
| `FORCE` | `0` \| `1` | `1` rebuilds the image even if it exists, and re-runs trials the registry already marks DONE |
| `ALIAS` | a name | Fixes the generated model name instead of a random one |
| `MODELS` | model names | Restricts a run to the listed models |
| `HY` | `H1` … `H4` | Hypothesis selector; required only when `auto_mlops.enabled: true` |
| `GRAPH_ID` | a job `id` | Job to draw (`graph`) |
| `FILTER` | path | Another filter file, relative to the project root (`find_model`); default `config/model_filter.yaml` |
| `RERUN_EXPERIMENT`, `REPEAT` | model name, int | Experiment and repeat count (`rerun`); the name can also follow `rerun` directly |
| `MODEL` | model name(s) or hash(es), comma-separated | Models to test (`test`); names are in `projects/<P>/artifacts/done_model_names.yaml` |
| `TEST_CONFIG` | path | Another test request file, relative to the project root (`test`); default `config/test_config.yaml` |
| `EXTERNAL` | `1` | An external test (`test`): `test_config.yaml`'s attack settings, beside the recorded test |
| `STAGING_REMOTE`, `PUSH_STAGING` | `0` \| `1` | Remote-staging pull policy and optional push |
| `NAMESPACE`, `IMAGE_REPO`, `IMAGE_REGISTRY` | strings | containerd namespace and image naming |

Examples:

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync50 MODEL_ID=ASCAD_CNNbest
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD MODEL_ID=ASCAD_MLPbest FORCE=1
make rerun       ENV=dev PROJECT=DLSCA_ASCAD <model_name> REPEAT=3
make test        ENV=dev PROJECT=DLSCA_ASCAD MODEL=<model_name>
make test        ENV=dev PROJECT=DLSCA_ASCAD MODEL=<model_name> EXTERNAL=1
make graph       ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=2
make find_model  ENV=dev PROJECT=DLSCA_ASCAD
make stack-nuke  ENV=dev
```
