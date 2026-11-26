# services/side_channel_core/src/core
Generic runtime entry points: the Typer CLI, the train-then-test orchestrator, the YAML-driven model builder and model contract resolver and a live metrics monitor.

## Contents
```text
core/
├── __init__.py                   # package marker (empty)
├── cli.py                        # Typer CLI: train-cmd, test-cmd, gen-data, orchestrate-cmd
├── keras_init.py                 # Keras-compatible weight initialisers (glorot/he/lecun uniform+normal, lecun_normal_keras = Keras 2.1.1 rule, random_normal) and the weight_init requirement (configured_weight_init), applied by model.py to layer models and named architectures
├── model.py                      # build_model from YAML layers/nodes; architecture registry
├── model_contract_resolver.py    # select model_id, resolve layers and output head, validate weight_init
├── monitor.py                    # a run's live metrics, one line per metric, its value per process; while it tests, attack i of n and the test values so far (utils/attack_progress.py); the latest key-recovery report (snapshot); the TraceFlow Monitor's [Running] list
├── orchestrator.py               # select runs, train_and_test per hash, reconcile registries
├── README.md                     # this file
└── README.pdf                    # old rendered README
```

## Used by
- `cli.py`: `scripts/orchestrator.sh` (`python -m core.cli orchestrate-cmd`), `utils/cli_helpers.py` (subprocess `core.cli test-cmd -n <hash>`), `scripts/orchestrator_auto_mlops.sh` (`test-cmd --test-request` for `make test`), `core/orchestrator.py`.
- `orchestrator.py`: `core/cli.py` (`orchestrate-cmd`; `train-cmd -n` via `main(registered_only=True)`).
- `model.py`: `src/sca/train.py`, `src/sca/test.py`, `utils/cross_validation_helpers.py`, `utils/flat_size.py`.
- `model_contract_resolver.py`: `src/sca/train.py`, `utils/cli_helpers.py`, `utils/sweep_engine.py`, `utils/model_preview.py` (`is_named_architecture`), `services/mlops/core/optuna_lite.py`, `services/mlops/utils/scc_bridge_lite.py`.
- `monitor.py`: the TraceFlow Monitor (`services/visualization/vscode_monitor/extension.js` liveMetrics: `python -m core.monitor --run-dir <run> --json`), which shows it under each run of the [Running] list; from a terminal, `python -m core.monitor --run-dir <run>`.

## TODOs
None.
