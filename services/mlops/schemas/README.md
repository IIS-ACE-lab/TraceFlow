# services/mlops/schemas
Pure validators (dataclass specs, no I/O) for `config/auto_mlops.yaml`. Entry point: `validate_auto_mlops_yaml()` in `auto_mlops.py`.

## Contents
```text
schemas/
├── __init__.py        # empty
├── auto_mlops.py      # top-level validator -> AutoMLOpsSpec; calls the block validators below
├── common.py          # shared primitives, guards, normalization
├── project.py         # project list, overrides, hypotheses, allocation, promotion
├── objectives.py      # objective directions, primary, pareto
├── constraints.py     # hard/soft constraints, budgets, resources
├── search.py          # search.* (Optuna engine, sampler, pruner, budget)
├── retention.py       # trial retention and artifact cleanup
├── registry.py        # registry.* (auto_run_map, leaderboard, pareto_front)
├── visualization.py   # visualization.* (tensorboard, static_reports)
├── learning.py        # learning method/strategy
├── model.py           # model family, architecture hints
├── trainer.py         # trainer hyper-params and search spaces
├── pruning.py         # pruning / early-stop rules
└── benchmarking.py    # benchmark and qualification rules
```

## Used by
- `services/mlops/utils/auto_config.py:59,66` — `AutoMLOpsSpec`, `validate_auto_mlops_yaml`
- `services/mlops/core/context.py:26` — optional validation
- `services/side_channel_core/utils/config_loader.py:444` — only when a project sets `auto_mlops.enabled: true` (currently commented out in master.yaml)
- Internal: `auto_mlops.py` imports project, objectives, constraints, retention, search, visualization, registry; `project.py` imports learning, model, trainer, pruning, benchmarking; all import `common.py`.

## TODOs
None.
