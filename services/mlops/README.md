# services/mlops
Hyper-parameter search layer on top of the Side-Channel Core (SCC). SCC owns data, training, testing, naming, hashing and registries; this service samples configurations, hands each one to SCC and records the objective. Two paths exist:

- **AutoSearch / OptunaLite** (in use): `core/optuna_lite_entry.py` -> `core/optuna_lite.py` -> `utils/scc_bridge_lite.py`. Driven by each project's `config/auto_search.yaml`.
- **AutoMLOps** (undeveloped): `core/auto_mlops_runner_entry.py` -> `utils/runner.py` -> `core/optuna_runner.py` -> `utils/scc_bridge.py`. Driven by `config/auto_mlops.yaml`. Disabled in master.yaml ("FEATURE UNDEVELOPED ... Please use auto_search instead").

Authors: Shraddha Mishra.

## Contents
```text
services/mlops/
├── __init__.py     # package marker
├── README.md       # this file
├── core/           # search runners, Optuna engine, entry points (see core/README.md)
├── utils/          # SCC bridges, MLOpsConfig, registry repair, cleanup (see utils/README.md)
├── config/         # auto_mlops.yaml, meta_mlops.yaml, config_tracking_registry.yaml
├── schemas/        # validators for auto_mlops.yaml
├── adapters/       # disk registry backend, SCC training backend (not on a live path)
└── contracts/      # TrialSpec contract (unused)
```

## How it is started

`make orchestrate` (repo-root Makefile) builds the image and runs `nerdctl compose up`; the container command is `services/side_channel_core/scripts/orchestrator_auto_mlops.sh` (docker-compose.yaml and all Dockerfile stages). That script picks one mode:

1. `RERUN_EXPERIMENT` set -> `orchestrator_rerun.sh` (SCC).
2. `SEARCH=1` or `AUTO_SEARCH_FORCE=1` -> AutoSearch.
3. `MLOPS=1` or `AUTO_MLOPS_FORCE=1` -> AutoMLOps.
4. Otherwise from master.yaml for `PROJECT` / `default_project`: `auto_search.enabled` -> AutoSearch; else `auto_mlops.enabled` -> AutoMLOps; else manual SCC (`orchestrator.sh`). AutoSearch wins if both are enabled.

The Makefile forwards `MLOPS` to the container but not `SEARCH`, so under `make orchestrate` AutoSearch is selected through master.yaml.

AutoSearch command run by the script:
```bash
python -u -m services.mlops.core.optuna_lite_entry --project "$PROJECT" --master-yaml "$SCC_MASTER_YAML" --mode orchestrate
```
Before start, and after a non-zero exit or SIGINT/SIGTERM, it runs:
```bash
python -m services.mlops.utils.registry_repair --project-root "$PROJECT_ROOT" --env "$ENV" --reason <reason>
```

Direct use (inside the container, repo at `/app`):
```bash
python -m services.mlops.core.optuna_lite_entry --project DLSCA_ASCAD --env dev --mode train --max-trials 10
```
`auto_mlops_runner_entry` also accepts `--project --env --dry-run --resume --max-trials`, but see the blocking issues under AutoMLOps below.

Makefile note: `AUTO_MLOPS_ENABLED` is computed from master.yaml; when it is 1, `HY={H1..H4}` is required and the container is named `sidechannel-<ENV>-<HY>`. With `auto_mlops` commented out it is 0.

## AutoSearch / OptunaLite
- Project config comes from SCC `config_loader.get_project_config(project)`, which merges `projects/<P>/config/auto_search.yaml` onto params.yaml when `auto_search.enabled: true` (`hard_replace_params` selects override vs fill-missing).
- `search.engine` must be `optuna`; `search.method` `bo` (TPE) or `rs` (random); `search.budget.max_trials` int or `auto` (all combinations); `search.objective.metric` / `.direction` (defaults `guessing_entropy`, `minimize`). Pruner is always `none`.
- Search space: model ids in `cfg["model"]`, list-valued `trainer` fields, SCA variants. `MODEL_ID` limits to one model. Points already completed in the run map are skipped unless `FORCE=1`.
- No separate trial registry: `model_hash` is the identity in SCC's `artifacts/run_registry.json` / `test_registry.json`.

## SCC bridge (`utils/scc_bridge_lite.py`)
For each trial: resolve model contract, freeze `model_name`/`model_hash`, create registry entries, run SCC `train()`; with `--mode orchestrate` also run SCC `test()` and print one combined validation/test key-recovery summary (`side_channel_core/utils/key_recovery_helper`). Statuses go `in_progress` -> `DONE`/`FAILED`. The objective is then read from `metrics/scalars/*_metrics_validation_history.jsonl` (fallback: train history). `cleanup_trial_runtime()` frees CUDA cache, runs gc and reaps child processes after each trial.

## AutoMLOps (undeveloped)
- master.yaml: the `auto_mlops:` block is commented out with `#FEATURE UNDEVELOPED` (`services/side_channel_core/config/master.yaml:18`).
- `utils/runner.py` loads `auto_mlops.yaml` only through `MLOpsConfig`, requires `search.engine: optuna`, and runs `OptunaRunner`.
- Blocking issues found in code: `MLOpsConfig()` without `core_cfg` needs `Brahman.utils.administration.federation_config_loader`, which is not in this repository; `runner.py` needs `--project` or `AUTO_MLOPS_PROJECT`/`PROJECT_NAME`, but the orchestrator passes only `--master-yaml` and exports `PROJECT`.
- `core/auto_mlops.py` (AutoMLOpsRunner) is a second, unwired orchestrator; `config/meta_mlops.yaml` (v2) is not read by any code.

## Configuration files
| File | Read by |
|---|---|
| `services/side_channel_core/config/master.yaml` (`project[].auto_search`, `project[].auto_mlops`, `default_project`) | orchestrator script, Makefile, SCC config_loader, `optuna_lite_entry` |
| `projects/<P>/config/auto_search.yaml` | SCC config_loader -> OptunaLite |
| `services/mlops/config/auto_mlops.yaml` | MLOpsConfig; SCC config_loader when `auto_mlops.enabled` |
| `services/mlops/config/config_tracking_registry.yaml` | MLOpsConfig |
| `services/mlops/config/meta_mlops.yaml` | nothing |

Environment: `ENV`, `PROJECT`, `DATASET`, `MODEL_ID`, `FORCE`, `SEARCH`/`AUTO_SEARCH_FORCE`, `MLOPS`/`AUTO_MLOPS_FORCE`, `SCC_MASTER_YAML`, `PROJECTS_ROOT`, `AUTO_SEARCH_RUNNER_MODULE`, `AUTO_MLOPS_RUNNER_MODULE`.

## Used by
- `services/side_channel_core/scripts/orchestrator_auto_mlops.sh` — `core.optuna_lite_entry`, `core.auto_mlops_runner_entry`, `utils.registry_repair`
- `services/side_channel_core/src/sca/train.py:101`, `src/sca/test.py:45` — `utils.zombie_reaper.cleanup_trial_runtime`
- `services/side_channel_core/utils/config_loader.py:444` — `schemas.auto_mlops.validate_auto_mlops_yaml` (only if `auto_mlops.enabled`)
- `services/side_channel_core/deployment/docker/docker-compose.yaml:60` — bind mount; `Dockerfile:107` — copied into the staging image

## TODOs
None. `grep -rn "TODO\|FIXME" services/mlops` finds nothing. `utils/hashing.py` is an empty stub ("NEED TO DEVELOP CONTENTS").
