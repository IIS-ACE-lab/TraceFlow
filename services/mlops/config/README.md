# services/mlops/config
YAML configuration for the hypothesis-driven AutoMLOps path and the planned v2 meta-learning layer. Not used by AutoSearch (OptunaLite), which reads each project's `config/auto_search.yaml` through SCC's config_loader.

## Contents
```text
config/
├── auto_mlops.yaml                # v1 AutoMLOps spec: paths, visualization, search, objectives, constraints, retention, registry, project/hypotheses
├── config_tracking_registry.yaml  # name -> file map read by MLOpsConfig (auto_mlops, meta_mlops, master)
└── meta_mlops.yaml                # v2 plan: meta-learning, failure theory, cross-project transfer (not read by code)
```

## Used by
- `config_tracking_registry.yaml`: default registry of `MLOpsConfig` (`services/mlops/utils/auto_config.py:315`).
- `auto_mlops.yaml`: loaded by `MLOpsConfig` (key `auto_mlops`) and validated by `services/mlops/schemas/auto_mlops.py`; also by `services/side_channel_core/utils/config_loader.py` when a project sets `auto_mlops.enabled: true` in master.yaml. In master.yaml that block is commented out and marked "FEATURE UNDEVELOPED" (`services/side_channel_core/config/master.yaml:18-21`).
- `meta_mlops.yaml`: only listed in `config_tracking_registry.yaml`; no code loads it. The `services/mlops/state/` paths it names do not exist.

## TODOs
None.
