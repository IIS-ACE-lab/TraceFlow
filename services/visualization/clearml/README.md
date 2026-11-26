# services/visualization/clearml
ClearML experiment-tracking adapter used by SCC train/test, plus a manual smoke test. Live: imported by the SCC runtime.

## Contents
```text
clearml/
├── clearml_tracker.py     # ClearMLSettings, ClearMLTracker, NullTracker, build_tracker_from_run_context()
└── clearml_smoke_test.py  # manual connectivity check against the default project's ClearML profile
```

- `ClearMLSettings.from_cfg()` reads `visualization.clearml` (top-level or under `project`): `enabled`, `scope` (`local`/`global`) and the matching `profiles.<scope>` block (URLs, key names, `clearml_env_file`, healthcheck).
- `ClearMLTracker` is a no-op (`enabled == False`, reason in `disabled_reason`) when: disabled in config, URLs missing, `clearml_env_file` missing, `clearml` package not importable, credentials not found, or TCP probe fails and `fail_fast_on_unreachable_api` is true (default). A relative `clearml_env_file` is resolved against `/app` (container path).
- Credentials are read through `services.side_channel_core.utils.secrets.get_secret` from the env file; they are exported as `CLEARML_API_*` environment variables before `Task.init()`.
- Logging helpers: `log_scalar`, `log_metrics`, `log_curve`, `log_curve_from_2d_array`, `log_table`, `log_mixed_metrics`, `upload_file`, `upload_object`, `upload_json_file`, `connect_config`, `connect_parameters`, `flush`, `close`.
- `build_tracker_from_run_context()` names the task `<model_name>_<process>[_trial_N]` and tags it `MLSCA`, `group:`, `process:`, `env:`, `model_hash:`.
- Smoke test (not a pytest test): `python -m services.visualization.clearml.clearml_smoke_test` from the repo root.

## Used by
- `services/side_channel_core/src/sca/train.py:91` — `build_tracker_from_run_context`
- `services/side_channel_core/src/sca/test.py:35` — `build_tracker_from_run_context`
- `services/side_channel_core/utils/epoch_utils.py:54` — `ClearMLTracker` (per-epoch metric/file logging)
- `clearml_smoke_test.py` imports `clearml_tracker.py`; nothing imports the smoke test.

## TODOs
None.
