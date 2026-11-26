# services/mlops/core
Search engines and entry points. Two paths live here: AutoSearch (`optuna_lite*`, the path used in practice) and hypothesis-driven AutoMLOps (`auto_mlops_runner_entry` -> `optuna_runner`, marked undeveloped in master.yaml).

## Contents
```text
core/
├── __init__.py                  # empty
├── optuna_lite_entry.py         # CLI for AutoSearch: --project --env --mode {train,orchestrate} --max-trials --master-yaml
├── optuna_lite.py               # OptunaLiteRunner: builds space from cfg, samples, runs trials via scc_bridge_lite
├── optuna_engine.py             # OptunaEngine: ask/tell wrapper around optuna (no I/O)
├── auto_mlops_runner_entry.py   # CLI for AutoMLOps: --master-yaml --project --env --dry-run --resume --max-trials
├── optuna_runner.py             # OptunaRunner: AutoMLOps loop (compile -> scc_bridge -> report -> judge -> tell)
├── trial_compiler.py            # hypothesis + sampled params -> SCC run_cfg; writes auto_params.yaml
├── trial_reporter.py            # finds SCC metrics CSV, extracts objective values
├── trial_judge.py               # accept/reject/qualify/promote decisions (pure)
├── pruning.py                   # pruning rules -> PruneDecision (pure)
├── auto_mlops.py                # AutoMLOpsRunner / run_auto_mlops(): alternative orchestrator (unused)
├── context.py                   # compile auto_mlops.yaml + project overrides + hypotheses (pure)
├── hypothesis_selector.py       # hypothesis allocation policies (pure)
├── auto_params_emitter.py       # deterministic params from search spaces (pure)
├── benchmark_runner.py          # benchmark plan builder, no execution (pure)
├── metrics_reader.py            # normalize metrics -> MetricsSnapshot (pure)
├── registry.py                  # TrialRegistry, HypothesisRegistry, RegistryBackend protocol
└── training_launcher.py         # TrainingBackend protocol + request/result types
```

AutoSearch (`optuna_lite.py`):
- Config: `get_project_config(project)` from SCC; requires `auto_search.enabled: true` for the project in master.yaml. Reads `search.engine` (only `optuna`), `search.method` (`bo` -> TPE, `rs` -> random), `search.budget.max_trials` (int or `auto`), `search.objective.metric` (default `guessing_entropy`) and `.direction` (default `minimize`).
- Space: model ids from `cfg["model"]`, list-valued `trainer` fields, SCA variants; `MODEL_ID` env restricts to one model.
- Skips points already in the project's run map unless `FORCE=1` / `AUTO_SEARCH_FORCE_RUN_ALL=1`.
- Each trial: `run_trial_via_scc_lite()` then objective read from the last record of `metrics/scalars/*_metrics_validation_history.jsonl` (fallback: train history); result told to `OptunaEngine`; run map updated.
- Exit code of `optuna_lite_entry`: 0 unless no trial succeeded and at least one failed (1); 2 on fatal error; 130 on Ctrl-C.

## Used by
- `services/side_channel_core/scripts/orchestrator_auto_mlops.sh:60` — `python -m services.mlops.core.optuna_lite_entry --project ... --master-yaml ... --mode orchestrate`
- `services/side_channel_core/scripts/orchestrator_auto_mlops.sh:61` — `python -m services.mlops.core.auto_mlops_runner_entry --master-yaml ...`
- `services/mlops/utils/runner.py:276` — `OptunaRunner`
- Internal: `optuna_lite` <- `optuna_lite_entry`; `optuna_engine` <- `optuna_lite`, `optuna_runner`, `auto_mlops`; `trial_compiler`, `trial_reporter`, `trial_judge` <- `optuna_runner`; `pruning` <- `trial_judge`, `auto_mlops`; `registry` <- `auto_mlops`, `hypothesis_selector`, `resume`, `adapters/disk_registry_backend`.
- Not imported anywhere: `auto_mlops.py` (and therefore `context`, `hypothesis_selector`, `auto_params_emitter`, `benchmark_runner`, `metrics_reader`, `training_launcher` are reachable only through it).

## TODOs
None.
