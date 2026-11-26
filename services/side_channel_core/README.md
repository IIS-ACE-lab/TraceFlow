# services/side_channel_core
Side-channel core (SCC): the TraceFlow training/testing runtime. It turns one project configuration into concrete runs, trains and tests each run inside a container, and writes artifacts, metrics and registries under `projects/<P>/`.

Authors: Shraddha Mishra.

## What the service does

| Concern | Modules |
| --- | --- |
| Config loading | `utils/config_loader.py` (`get_project_config`): merges `config/master.yaml` (project block) with `projects/<P>/config/params.yaml`, the `extra_configs` (e.g. `graph`, `test`), the `env_configs` file for `$ENV` (dev/staging/prod; existing keys only, given values only, else `ConfigError`: `apply_env_overrides`), each file's keys checked against what the code reads (`utils/config_schema.py`), optional `auto_search.yaml`, optional `projects/<P>/config/model_config.yaml`, and `distributed` settings |
| Sweep expansion | `utils/sweep_engine.py` (grid, run identity, `run_registry.json`), `utils/sca_sweep.py` (conditional `sca:` block: attack byte, leakage model, candidate ranking, key-enumeration profiles), `utils/model_naming.py` (hash + name) |
| Data contract and splits | `src/sca/ascad_data_loader.py` (config-specific HDF5 with `sca_data_contract_json` fingerprint), `utils/data_split.py` (train/val/test HDF5 splits), `utils/data_reproducibility.py` (config snapshot, generate or verify data) |
| Model building | `src/core/model_contract_resolver.py` (select `model_id`, resolve output head, validate the initialisers against the framework), `src/core/model.py` (`build_model` from YAML layers/nodes), `src/core/framework_switch.py` (`deep_learning.framework`: PyTorch or Keras 3 layers, activations, initialisers, loss, optimizer, schedules), `src/model_architectures/` (hard-coded networks) |
| Training | `src/sca/train.py`, `utils/epoch_utils.py` (`run_training_loop`, `run_test_loop`), `utils/checkpoint_policy.py` |
| 10-fold cross-validation | `utils/cross_validation.py` (plan, folds, manifest), `utils/cross_validation_pipeline.py` (folds + final refit), `utils/cross_validation_helpers.py`, `utils/cross_validation_diagnostics.py` |
| Attacks and metrics | `src/sca/perform_attack.py` (repeated full-population attacks), `src/sca/metrics.py`, `src/sca/rank_curve.py`, `src/sca/ldd.py`, `src/sca/leakage_models.py` |
| Key recovery and enumeration | `src/sca/key_recovery.py` (reports), `src/sca/key_enumeration.py` (enumeration algorithms), `utils/key_recovery_helper.py` (final summary), `utils/key_enumeration_helper.py` (completed-cycle enumeration) |
| Registries | `utils/sweep_engine.py` (`run_registry.json`), `utils/test_registry.py`, `utils/lifecycle_outcome.py` (key-recovery outcome in both), `utils/run_map.py`, `utils/run_lifecycle_logger.py`, `utils/atomic_io.py` |
| Orchestration | `scripts/orchestrator_auto_mlops.sh` (dispatch), `scripts/orchestrator.sh` (retry/resume loop), `scripts/orchestrator_rerun.sh` (snapshot replay), `src/core/cli.py` (Typer CLI), `src/core/orchestrator.py` (train then test per model hash) |
| Containerised execution | `deployment/docker/` (`Dockerfile`, `docker-compose*.yaml`) |

## Running

Everything is driven by the root `Makefile`, from the repository root:

```bash
make ENV=dev PROJECT=<P> orchestrate          # build image, run all pending runs, clean up
make ENV=dev PROJECT=<P> rerun RERUN_EXPERIMENT=<model_name> REPEAT=1
make ENV=dev PROJECT=<P> judge                # services/model_judge
make ENV=dev PROJECT=<P> GRAPH_ID=1 graph     # services/graph_manager
make ENV=dev healthcheck | reset              # scripts/healthcheck.sh | scripts/gpu_reset.sh
```

`PROJECT` defaults to `default_project` in `config/master.yaml`. Experiments run in containers via rootful containerd + nerdctl + compose v2 (`deployment/docker/docker-compose.yaml` + `docker-compose.<ENV>.yaml`, image from `deployment/docker/Dockerfile`). The host needs the `sca` conda environment for the Makefile, the orchestration helpers, the graph manager and the model judge: `infra/environment/setup_sca_env.sh` installs it (Python 3.12, PyTorch 2.9.1 cu128, `infra/environment/requirements.txt`, the same pins as the image), and `scripts/setup.sh` is the entry point.

Call chain inside the container:
`scripts/orchestrator_auto_mlops.sh` -> (auto_search -> `services.mlops.core.optuna_lite_entry` | auto_mlops -> `services.mlops.core.auto_mlops_runner_entry` | `RERUN_EXPERIMENT` -> `scripts/orchestrator_rerun.sh` | otherwise `scripts/orchestrator.sh`) -> `python -m core.cli orchestrate-cmd` -> `core.orchestrator.main` -> `utils.cli_helpers.train_and_test` -> `sca.train.train` in process, then subprocess `core.cli test-cmd -n <model_hash>`, per model hash. `core.cli train-cmd -n <registered model_name|model_hash>` takes the same path through `core.orchestrator.main(registered_only=True)`: params_out snapshot, retry budget (`SCC_MAX_FAILED_ATTEMPTS`, `-f` overrides), paired test; a failure exits non-zero. New runs start only with `make orchestrate`.

## Run lifecycle (one model hash)

1. Config and identity: `get_project_config`; sweep expansion yields one concrete run; `model_hash`/`model_name` assigned and recorded in `artifacts/run_registry.json` and `artifacts/run_map.jsonl`.
2. Model contract: `resolve_model_contract` picks the model block, sets the output head (ID 256, HW 9, MSB 2) and checks `architecture_args.weight_init` (required when `strict_graph_args` is true); resolved model preview saved.
3. Data contract: `prepare_artifacts_data` writes the `params_out` snapshot (atomic, `snapshot_sha256`, relative paths), copies enabled parent configs, and generates (`data.generate_data_out: true`) or verifies `paths.data_out`, fingerprinted by `sca_data_contract_json`.
4. Splits: `train.h5` from `Profiling_traces`; `val.h5` / `test.h5` from disjoint `Attack_traces` ranges. With cross-validation only `train.h5` and `test.h5` are built.
5. Training, one of:
   - `cross_validation.enabled: true`: T folds over profiling data only (fresh model per fold; the refit trains with `seed` and the folds with the next T seeds, one each, so no fold shares the refit's weight init and data order), aggregate, then refit a fresh model on all profiling traces for `trainer.epochs`; the CV result is the aggregate under `cross_validation/aggregate/`. The refit's training traces are never attacked: its rows carry loss, accuracy and fit time only.
     With `cross_validation.report_aggregate: true`, `utils/cross_validation_diagnostics.py` judges the folds with one profile per `sca.leakage_model`. ID, HW and MSB each have label-prior chance accuracy and cross-entropy, and a stability rule: stable needs a fraction of folds at GE <= 1 plus an endpoint-rank std in ranks (ID 90%/2, HW 80%/6, MSB 50%/16). The training trend is taken from the epoch of minimum validation loss (`possible_overfitting` when the training loss fell and the fold mean of final - min exceeds 0.5% of the chance cross-entropy). The report is non-authoritative: it never changes the refit or the test.
   - otherwise: train with per-epoch validation (metrics, key-recovery evidence, optional best-validation checkpoint).
6. Test (skipped, with status `SKIPPED` and the reason in `artifacts/test_registry.json`, when `perform_attack.test.attack_times` is 0): `test-cmd` loads the checkpoint chosen by `checkpoint_policy`, runs `run_test_loop` on `test.h5` (attack traces), writes its metric row and the test key-recovery report, updates `artifacts/test_registry.json`.
7. Key-recovery summary: `key_recovery_helper` runs completed-cycle key enumeration (if enabled), reloads the validation/test reports, prints one combined summary and patches the outcome into the registries. Failures here are logged, not fatal.
8. Registries: run marked DONE in `run_map.jsonl` / `run_registry.json`; model appended to `artifacts/done_model_names.yaml` for graphs.

## Outputs

Paths come from `paths:` in `projects/<P>/config/params.yaml` and are relative to `projects/<P>/` (values from `projects/DLSCA_ASCAD`):

```text
artifacts/{{env}}/generated_models/{{model_name}}/   # artifacts_root
  config/{{model_name}}_params.yaml                  # params_out: resolved config snapshot (the run's only config copy; relative paths; starts with snapshot_sha256)
  data/{{dataset_alias}}.h5, data/test.h5            # data_out, test_data (+ train.h5, val.h5)
  model/                                             # model_out
  checkpoints/                                       # checkpoints
  logs/                                              # logs
  graphs/                                            # model_graphs
  metrics/attack_metrics/{scalars,arrays}/           # metrics (JSONL + .npy)
  metrics/key_recovery/                              # key_recovery reports
  cross_validation/                                  # fold artifacts and manifest
artifacts/run_registry.json, test_registry.json, run_map.jsonl, done_model_names.yaml, model_name_map.json
```

The history rows under `metrics/**/scalars/` store a metric only when `metrics_to_log` names it; what a requested metric is derived from or read with goes under `dependencies` (`src/sca/metrics.py` `split_requested`), and nothing else is stored. The run folder is the only record of a run: there is no separate results index, and `services/model_judge` reads the run folders (`run_reader.py`).

## Configuration

- `config/master.yaml`: project list, project root, params file, env configs, extra configs, auto_search/auto_mlops switches, visualization (ClearML, graph manager), distributed.
- `projects/<P>/config/params.yaml`: the experiment.
- `config/blueprint/params.yaml`: annotated reference of every params key (sweep behaviour, defaults, rejected values). Nothing reads it at runtime; copy what you need.

## Tests

From the repository root, in the `sca` environment:

```bash
PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
  python -m pytest services/side_channel_core/tests/unit
```

## Contents

```text
side_channel_core/
├── ci_cd/          # placeholder CI/CD stage scripts
├── config/         # master.yaml, blueprints, architectures, requirements
├── data/           # source datasets (ASCAD .h5); not documented here
├── deployment/     # docker (used), harbor registry, k8s/helm (unused)
├── infra/          # host conda env installer, machine secrets
├── scripts/        # container orchestration and GPU hygiene scripts
├── src/            # core (CLI, orchestrator, model), sca (train/test/attack), architectures
├── tests/          # unit tests, dummy project fixture
├── utils/          # config, sweep, data, CV, registries, logging helpers
├── .dockerignore   # image build exclusions
├── .gitignore      # service-level ignores
└── __init__.py     # package marker
```

## Used by

- Root `Makefile` (`orchestrate`, `rerun`, `judge`, `graph`, `healthcheck`, `reset`, `lint`).
- `services/mlops` (`core/optuna_lite.py`, `core/trial_compiler.py`, `utils/scc_bridge*.py`, `utils/registry_repair.py`): calls `core.orchestrator`, `sca.train`, `sca.test`, config, sweep and registry helpers.
- `services/graph_manager` and `services/model_judge`: `utils.config_loader`, `utils.path_utils`, `utils.logger`, `utils.graph_config_sync`.
- `services/visualization/clearml/clearml_tracker.py`: `utils.logger`, `utils.secrets`.

## TODOs

- `src/sca/train.py:65` — cross-device / cross-key testing.
- `src/sca/leakage_models.py:699` — add HD leakage model after defining the state transition.
- `config/pipeline_config.yaml:27` — `trigger_mode` ui/cli.
- `deployment/harbor/README.md:440` — LAN firewall rule for port 443.
- Open defects from the 2026-09-29 audit (swallowed checkpoint save errors, per-dataset sweep lists, disjoint validation attacks under cross-validation, crossings on tied true keys): repo-root `TODO.md` §20–§23.
