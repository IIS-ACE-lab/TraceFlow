# services/mlops/utils
Runtime glue: SCC bridges, config loading, logging, registry repair and process cleanup.

## Contents
```text
utils/
├── __init__.py          # empty
├── scc_bridge_lite.py   # run_trial_via_scc_lite(): one AutoSearch trial through SCC train (+ test), key-recovery summary
├── scc_bridge.py        # run_trial_via_scc(): SCC train for the AutoMLOps OptunaRunner
├── runner.py            # run_auto_mlops_from_master(): validate auto_mlops via MLOpsConfig, dispatch OptunaRunner
├── auto_config.py       # MLOpsConfig: registry-based YAML loader, schema validation, get_map(), path resolve
├── path_resolver.py     # {{var}} / ${var} path template expansion; fails on unresolved placeholders
├── logger.py            # mlops_logger(): append line to a path resolved from auto_mlops.yaml
├── registry_repair.py   # CLI: mark stale run/test registry entries INTERRUPTED
├── zombie_reaper.py     # cleanup_trial_runtime(): reap child processes, gc, free CUDA cache
└── hashing.py           # docstring only: "NEED TO DEVELOP CONTENTS" (stub)
```

`scc_bridge_lite.run_trial_via_scc_lite(run_cfg, mode, trial_id)`:
- sets `AUTO_SEARCH=1`, `AUTO_SEARCH_TRIAL_ID`, `MODEL_ID` for the call; resolves the model contract unless already resolved;
- freezes `model_name` / `model_hash` (the only registry identity; `trial_id` is progress metadata);
- ensures entries in `artifacts/run_registry.json` and, for `mode="orchestrate"`, `artifacts/test_registry.json`; sets status `in_progress` -> `DONE` / `FAILED`;
- calls SCC `train()`, then for `orchestrate` SCC `test()`, then prints one combined validation/test key-recovery summary via `side_channel_core/utils/key_recovery_helper._log_completed_cycle_key_recovery_summary`;
- returns `{trial_id, model_name, model_hash, run_dir, status, elapsed_s, metrics_dir}`; always calls `cleanup_trial_runtime()`.

## Used by
- `scc_bridge_lite.py`: `services/mlops/core/optuna_lite.py:51`
- `scc_bridge.py`: `services/mlops/core/optuna_runner.py:55`
- `runner.py`: `services/mlops/core/auto_mlops_runner_entry.py:46`
- `auto_config.py`: `utils/runner.py`, `utils/logger.py`, `utils/scc_bridge.py`, `core/optuna_runner.py`, `core/trial_compiler.py`, `core/context.py`, `core/auto_mlops.py`
- `path_resolver.py`: `utils/auto_config.py:52`
- `logger.py`: `utils/runner.py:42`, `core/auto_mlops.py:9`
- `registry_repair.py`: `services/side_channel_core/scripts/orchestrator_auto_mlops.sh:100` (`python -m services.mlops.utils.registry_repair --project-root --env --reason`)
- `zombie_reaper.py`: `services/side_channel_core/src/sca/train.py:101`, `src/sca/test.py:45`, `core/optuna_lite.py:52`, `utils/scc_bridge_lite.py:36`
- Not imported anywhere: `hashing.py`.

## TODOs
None. (`hashing.py` has the note "NEED TO DEVELOP CONTENTS", not tagged TODO.)
