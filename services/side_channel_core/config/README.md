# services/side_channel_core/config
Service-level configuration: `master.yaml` (which projects exist and how their configs combine), blueprint templates for project configs and reference architecture YAMLs. The Python dependency lists moved to `infra/environment/`.

## Contents
```text
config/
├── __init__.py            # package marker (empty)
├── master.yaml            # projects, params/env/extra configs, auto_search, visualization, distributed
├── pipeline_config.yaml   # ci_cd stages (build/test/deploy) and ci_cd log dir
├── architectures/         # reference model YAMLs (copy into params.yaml model:)
├── blueprint/             # templates for projects/<P>/config/*.yaml
└── README.md              # this file
```

`master.yaml` project block (keys read by `utils/config_loader.py` unless noted):

```text
name, root                 # project key; root relative to projects/
params                     # params.yaml, relative to root
env_configs.{dev,staging,prod}  # deep-merged over params for $ENV
extra_configs.<name>       # loaded as cfg[<name>] (graph, test); file must exist
auto_search / auto_mlops   # enabled, config, hard_replace_params
scripts.<key>              # train/test/model/data_loader paths used by core/cli.py
use_core_src, code_dir, logs, models, metrics   # copied into cfg["project"]
visualization              # clearml profiles, graph_manager, dashboards
distributed_override       # merged into top-level distributed when use: true
exp_rerun, print_config    # read by scripts/orchestrator_rerun.sh / logger
default_project            # Makefile default for PROJECT
```

Annotated reference of every `params.yaml` key: `blueprint/params.yaml`.

## Used by
- `master.yaml`: `utils/config_loader.py`, root `Makefile` (`default_project`, auto_mlops flag, graph_manager), `scripts/orchestrator_auto_mlops.sh`, `scripts/orchestrator.sh`, `scripts/orchestrator_rerun.sh`, `src/core/{cli,orchestrator,graph_manager,monitor}.py`, `src/sca/{train,test}.py`, `utils/{cli_helpers,logger}.py`, `services/mlops`, `services/graph_manager/graph_manager.py`, `services/visualization/utils/*`, `deployment/docker/docker-compose.yaml` (`SCC_MASTER_YAML`).
- `pipeline_config.yaml`: `ci_cd/pipeline_utils.py`, `scripts/run_cicd.sh`.

## TODOs
- `pipeline_config.yaml:27` — `trigger_mode` options ui/cli.
