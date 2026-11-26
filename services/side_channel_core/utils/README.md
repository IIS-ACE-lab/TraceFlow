# services/side_channel_core/utils
Helper modules of the SCC runtime: config loading, sweeps, data splits, training loops, cross-validation, key recovery, registries, logging and paths. Imported as `utils.<module>` (with `services/side_channel_core` on `PYTHONPATH`) or `services.side_channel_core.utils.<module>`.

## Contents
```text
utils/
├── __init__.py                       # package marker (empty)
├── aes.py                            # AES_SBOX table
├── artifact_utils.py                 # create/rename run artifact directories
├── atomic_io.py                      # atomic text writes for the records under artifacts/
├── checkpoint_policy.py              # best-validation checkpoint and test-model selection
├── cli_helpers.py                    # registered run's config from its snapshot; train_and_test (train in process, test-cmd subprocess)
├── config_loader.py                  # get_project_config: master.yaml + params.yaml + env/extra configs
├── container_guard.py                # assert code runs inside a container
├── cross_validation.py               # CV plan, fold indices, manifest, finalization
├── cross_validation_diagnostics.py   # optional non-authoritative fold report
├── cross_validation_helpers.py       # per-fold/refit model, optimizer, loaders
├── cross_validation_pipeline.py      # full CV path: folds, aggregation, final refit
├── data_reproducibility.py           # params_out snapshot (atomic, snapshot_sha256, relative paths), enabled parent configs, generate or verify data_out and splits
├── data_split.py                     # train/val/test HDF5 splits and DataLoaders
├── dataloader_utils.py               # DataLoader kwargs from cfg
├── trace_scaler.py                   # trace_normalization.mode fitted: sklearn scalers fitted on the training traces; scaler.json
├── dataset_fetch.py                  # a missing registered database: submodule, else official download; SHA-256 checked; converted
├── delete_database.py                # make delete_database: remove one env's (or named models') runs from artifacts/
├── device.py                         # choose device, build model; a lost GPU fails the run (GPU FAILED), no CPU fallback
├── epoch_utils.py                    # run_training_loop, run_test_loop
├── error_handler.py                  # catch_exceptions decorator
├── flat_size.py                      # compute Linear in_features for a CNN YAML (standalone; python utils/flat_size.py [arch.yaml])
├── graph_config_sync.py              # artifacts/done_model_names.yaml, graph_config top models
├── h5_metadata_utils.py              # batch/metadata unpacking, metric gating
├── h5_utils.py                       # HDF5 inspection and sample counting
├── key_enumeration_helper.py         # completed-cycle key enumeration integration
├── key_recovery_helper.py            # final combined key-recovery summary
├── label_checks.py                   # verify stored labels match leakage model
├── lifecycle_outcome.py              # key-recovery outcome patched into run_registry.json / test_registry.json
├── logger.py                         # logging, terminal printing
├── loss_util.py                      # CrossEntropyLoss / NLLLoss / FocalLoss from config
├── metrics_logger.py                 # per-epoch JSONL scalars + .npy arrays (one row per epoch; reruns replace)
├── model_naming.py                   # experiment hash and model name
├── model_preview.py                  # save/print resolved model preview
├── monitor_metric.py                 # monitored metric and better-score test
├── optimizer_util.py                 # optimizer/scheduler from config
├── output_transformation_util.py     # Identity/Softmax/LogSoftmax on logits
├── path_utils.py                     # resolve {{env}}/{{model_name}} path templates
├── resolver_mode.py                  # when to run the model contract resolver
├── run_lifecycle_logger.py           # run_map start/complete/fail/interrupt
├── run_map.py                        # artifacts/run_map.jsonl store
├── run_utils.py                      # registry scope/selection, latest model lookup
├── save_utils.py                     # save model state_dict
├── sca_sweep.py                      # conditional sca: search-space compiler
├── secrets.py                        # read secrets from .env files / env vars
├── seed.py                           # seed Python/NumPy/PyTorch
├── status_store.py                   # CI pipeline status dict
├── sweep_engine.py                   # grid expansion, run identity, run_registry.json
├── sweep_runner.py                   # train sweep with retry/resume/filters
├── sweep_test_runner.py              # test sweep over trained runs
├── test_registry.py                  # artifacts/test_registry.json store
├── timestamps.py                     # every written timestamp, in master.yaml's timezone (city -> IANA zone)
├── tree_view.py                      # print directory tree (standalone script)
├── type_coerce.py                    # strict config type coercion
└── README.pdf                        # old rendered README
```

## Used by
- `src/sca/train.py`, `src/sca/test.py`, `src/core/{cli,orchestrator,monitor,graph_manager,model}.py`, `src/sca/{ascad_data_loader,metrics,key_recovery}.py`.
- `services/mlops` (`core/optuna_lite.py`, `core/trial_compiler.py`, `core/optuna_runner*.py`, `utils/scc_bridge*.py`, `utils/registry_repair.py`).
- `services/graph_manager/{graph_manager,metric_catalog}.py`, `services/model_judge/model_judge.py`, `services/visualization/clearml/clearml_tracker.py`.
- `ci_cd/pipeline_utils.py` (`status_store`).
- `tests/unit/test_key_enumeration.py`, `tests/unit/test_graph_manager.py`.
- Standalone scripts, not imported: `flat_size.py`, `tree_view.py`.

## TODOs
None.
