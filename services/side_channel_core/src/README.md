# services/side_channel_core/src
Python source of the SCC runtime: generic core (CLI, orchestrator, model builder), SCA-specific training/testing/attack code, and hard-coded architectures. On `PYTHONPATH` as `src` so modules import as `core.*`, `sca.*`, `model_architectures.*`.

## Contents
```text
src/
├── __init__.py            # package marker (empty)
├── core/                  # CLI, orchestrator, model builder, graphs, monitor
├── model_architectures/   # hard-coded PyTorch networks
├── sca/                   # data, train, test, attacks, metrics, key recovery
├── README.md              # this file
└── README.pdf             # old rendered README
```

## Used by
- `scripts/orchestrator.sh` (`python -m core.cli orchestrate-cmd`), `scripts/orchestrator_rerun.sh` (`sca.train.train`, `sca.test.test`).
- `utils/` (see `utils/README.md`), `services/mlops`, `services/graph_manager`.
- `master.yaml` `scripts:` paths (`src/sca/train.py`, `src/sca/test.py`, `src/core/model.py`, `src/sca/ascad_data_loader.py`).

## TODOs
- `sca/train.py:65` — cross-device / cross-key testing.
- `sca/leakage_models.py:699` — add HD leakage model after defining the state transition.
