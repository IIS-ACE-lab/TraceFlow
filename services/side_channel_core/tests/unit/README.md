# services/side_channel_core/tests/unit
Pytest unit tests of TraceFlow and the TraceFlow Monitor's Python helpers: 63 files with tests (978 cases with
parametrisation); `test_data_loader.py`, `test_train.py` and `test_utils.py` are empty.

Run (repository root, `sca` environment):

```bash
PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" CUDA_VISIBLE_DEVICES= \
  python -m pytest services/side_channel_core/tests/unit -q
```

`CUDA_VISIBLE_DEVICES=` keeps the Keras tests on the CPU (with a GPU visible, five `test_framework_switch.py`
cases fail on a cpu-vs-cuda device mismatch in the test). Tests that read `projects/DLSCA_ASCAD` skip what is
not on the machine (a database, a finished run).

## Contents

| Area | Files |
|---|---|
| Configuration | `test_config_schema.py` (unread keys refused), `test_config_env_overrides.py` (dev/staging/prod over params.yaml), `test_config_freeze.py` (a run reads its frozen config), `test_config_view.py` (snapshot and logged CONFIG), `test_retired_keys.py`, `test_rank_base_retired.py` |
| Data | `test_dataset_types.py` (data[].type, ASCADf back-compat), `test_dataset_converters.py`, `test_dataset_fetch.py` (fetch once, SHA-256), `test_data_files_contract.py`, `test_label_targets.py` (sbox, sbox_masked, inv_sbox_hd), `test_trace_scaler.py` |
| Models and training | `test_model.py`, `test_final_models.py` (construction, weight init), `test_layer_list_limits.py`, `test_framework_switch.py` (PyTorch / Keras from their libraries), `test_keras_clipped_loss.py`, `test_focal_loss.py`, `test_optimizer_args_wiring.py`, `test_precision.py`, `test_monitor_metric.py`, `test_device.py`, `test_device_record.py`, `test_checkpoint_dir.py`, `test_model_save.py` |
| Sweeps, CV, orchestration | `test_linked_step_sweep.py` (attack_step with rank_step), `test_final_cv.py`, `test_final_cv_diagnostics.py`, `test_final_orchestration.py`, `test_experiment_queue.py` (queue order, FORCE, stop at the first failure), `test_batch.py`, `test_final_identity.py` (run identity, snapshots) |
| Attack and key recovery | `test_inference.py` (float64 log-likelihood, key ranks), `test_final_rank.py` (ties), `test_classification_metrics.py`, `test_key_enumeration.py`, `test_unattacked_test.py` (attack_times 0), `test_trained_settings.py` |
| Registries and artifacts | `test_registry_records.py`, `test_seal.py` (Ed25519 seals), `test_heartbeat.py`, `test_live_monitor.py`, `test_timestamps.py`, `test_delete_database.py`, `test_delete_database_tmux.py`, `test_reproduce.py` |
| Testing, judging, figures | `test_test_request.py` (make test), `test_model_filter.py`, `test_model_judge.py`, `test_graph_manager.py`, `test_final_graph.py`, `test_graph_export.py`, `test_graph_rules.py`, `test_figure_sources.py`, `test_model_explanation.py` |
| TraceFlow Monitor (Python) | `test_vscode_monitor_py.py` (layout, scaffold, choices, model check), `test_yamledit.py` (span-preserving edits), `test_option_lists.py` (menus = what the code accepts), `test_trace_view.py` (trace lines, ranges, trained runs, recorded key recovery), `test_status_share.py` (status page: links, facts without paths, GET only), `test_model_diagram.py` (the planner's Download: every style, orientation and theme, text rules, trace lines, saved as graph_manager saves, copied into the runs with these layers) |
| Audits | `test_audit_2026_09_28.py` |
| Smoke | `test_addition.py` |

The panel's JavaScript tests are in `services/visualization/vscode_monitor/test/` (node `--test`).

## Used by
- `ci_cd/test.sh`; run by hand.
