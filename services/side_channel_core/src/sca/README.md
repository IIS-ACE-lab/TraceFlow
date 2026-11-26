# services/side_channel_core/src/sca
SCA-specific code: ASCAD data generation and loading, training and testing entry points, leakage models, attacks, metrics, rank curves, key-recovery reports and key enumeration.

## Contents
```text
sca/
├── __init__.py              # package marker (empty)
├── ascad_data_loader.py     # generate config-specific HDF5 with data contract; split datasets
├── converters/              # one module per dataset family: its official files -> the ASCAD layout (CHES_CTF_2018, AES_RD, DPAv4, AES_HD)
├── dataset_types.py         # database types: data[].type, the keys each type takes, sca.variant from the type, sources
├── key_enumeration.py       # AES-byte key enumeration algorithms (pure)
├── key_recovery.py          # key-recovery reports from attack evidence
├── ldd.py                   # LDD and Lm metrics
├── leakage_models.py        # AES intermediate, ID/HW/MSB leakage labels
├── metrics.py               # classification and SCA metrics (GE, SR, rank)
├── perform_attack.py        # repeated full-population attacks
├── rank_curve.py            # one-based key-rank curves
├── test.py                  # test(): load checkpoint, run_test_loop on test.h5; --model = one-off request
├── test_request.py          # make test: model names -> model_name_map -> run_map -> test-cmd on the training snapshot
├── train.py                 # train(): contract, data, CV or standard training
└── README.md                # this file
```

## Used by
- `train.py`: `core/cli.py train-cmd` (loaded from `master.yaml` `scripts.train`), `utils/cli_helpers.py`, `services/mlops/utils/scc_bridge.py`, `services/mlops/utils/scc_bridge_lite.py`, `scripts/orchestrator_rerun.sh`.
- `test.py`: `services/mlops/utils/scc_bridge_lite.py`, `scripts/orchestrator_rerun.sh`; `core/cli.py` via `scripts.test`; `core/cli.py test-cmd --test-request` (from `make test`).
- `test_request.py`: `core/cli.py test-cmd --test-request`; repo-root `Makefile` `test` (host-side config check).
- `ascad_data_loader.py`: `test.py`, `perform_attack.py`, `utils/data_split.py`; `core/cli.py gen-data` via `scripts.data_loader`.
- `metrics.py`: `utils/epoch_utils.py`, `utils/h5_metadata_utils.py`.
- `perform_attack.py`: `metrics.py`, `utils/cross_validation.py`. `rank_curve.py`: `metrics.py`, `perform_attack.py`. `ldd.py`: `metrics.py`.
- `dataset_types.py`: `utils/config_loader.py` (every data entry checked at load), `utils/model_naming.py` (hash), and every reader of `sca.variant` (`ascad_data_loader.py`, `perform_attack.py`, `ldd.py`, `metrics.py`, `key_recovery.py`, `utils/data_split.py`).
- `leakage_models.py`: most of `sca/`, `core/model_contract_resolver.py`, `utils/{data_split,epoch_utils,h5_utils}.py`.
- `key_recovery.py`: `utils/epoch_utils.py`, `utils/key_recovery_helper.py`, `utils/key_enumeration_helper.py`, `tests/unit/test_key_enumeration.py`.
- `key_enumeration.py`: `utils/key_enumeration_helper.py`, `utils/key_recovery_helper.py`, `utils/sca_sweep.py`, `core/model_contract_resolver.py`, `tests/unit/test_key_enumeration.py`.

## TODOs
- `train.py:65` — use cross-device / cross-key testing (train on device A, test on B).
- `leakage_models.py:699` — add HD only after defining the exact state transition.
