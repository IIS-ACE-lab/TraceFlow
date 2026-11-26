# services/visualization
Experiment-tracking and visualization service for TraceFlow. Three parts work: the ClearML integration (`clearml/` plus three helpers in `utils/`), which SCC train/test use to log runs; the TraceFlow Monitor (`vscode_monitor/`), a VS Code side panel that watches, configures and launches the experiments of every project in `master.yaml`; and the status page (`server/status.py` serving `mobile/`), a read-only view of the runs for phones and shared links. The web dashboard, ingest, reports, schemas and CLI directories are empty placeholders.

Authors: Shraddha Mishra.

## What is live and what is not

| Part | State | Evidence |
|---|---|---|
| `clearml/clearml_tracker.py` | live | imported by `side_channel_core/src/sca/train.py:91`, `src/sca/test.py:35`, `utils/epoch_utils.py:54` |
| `utils/ensure.py` | live | imported by `src/sca/train.py:102`, `src/sca/test.py:47` |
| `utils/helpers.py` | live | imported by `clearml/clearml_tracker.py:62` |
| `clearml/clearml_smoke_test.py` | manual tool | run by hand; not referenced anywhere |
| `utils/get_clearml_scope.py` | unused | not referenced by any script or Makefile |
| `vscode_monitor/` | live | the TraceFlow Monitor; VS Code only; installed by `make panel` and by `services/side_channel_core/infra/environment/setup_sca_env.sh` when VS Code is the editor; reads `projects/<P>/artifacts/<env>/heartbeat.json` written by `services/side_channel_core/utils/heartbeat.py`; guide: [`vscode_monitor/README.md`](vscode_monitor/README.md) |
| `server/status.py`, `mobile/` | live | the status page: started from the panel's share button (`make status_share`), one link per person; guides: [`server/README.md`](server/README.md), [`mobile/README.md`](mobile/README.md) |
| `cli/`, `ingest/`, `reports/`, `schemas/`, `web/`, `server/{app.py,routes,security}`, `utils/{io,jsonl,paths,time}.py` | placeholder | empty packages or 0-byte files; nothing imports them |

The only server is the read-only status page, started by hand (the panel's share button or `make status_share`); the TraceFlow Monitor runs inside VS Code.

## Contents
```text
services/visualization/
├── __init__.py                 # empty
├── README.md                   # this file
├── readme_clear_ml_setup.md    # notes for deploying a self-hosted ClearML server stack
├── clearml/                    # ClearML tracker + smoke test (live)
├── utils/                      # ensure.py, helpers.py (live); get_clearml_scope.py (unused); 4 empty files
├── vscode_monitor/             # the TraceFlow Monitor: VS Code side panel and the planner tab (live)
├── cli/                        # placeholder (README only)
├── ingest/                     # placeholder (README only)
├── mobile/                     # the status page (index.html, status.js, status.css) and its notes (live)
├── reports/                    # placeholder (README only)
├── schemas/                    # empty placeholder (events, metrics, status, dashboard)
├── server/                     # status.py: the status page's server (live); app.py, routes/, security/ empty
└── web/                        # empty placeholder (templates, static, components, assets)
```

## ClearML integration

Flow during an SCC run:
1. `train.py` / `test.py` call `_ensure_visualization_cfg(run_cfg, project_name)`, which replaces `run_cfg["visualization"]` with the matching project's `visualization` block from master.yaml (`$SCC_MASTER_YAML` or the default path).
2. They call `build_tracker_from_run_context(cfg, process_name, model_name, model_hash, ...)`, which returns a `ClearMLTracker`.
3. The tracker is a no-op unless `visualization.clearml.enabled` is true and the active profile resolves (URLs, env file, credentials, reachable API). The failure reason is stored in `tracker.disabled_reason`; SCC runs continue without ClearML.
4. `epoch_utils.run_training_loop` logs per-epoch results through `tracker.log_metrics` / `log_curve_from_2d_array` and uploads files through `upload_file`.

Manual check (from the repo root, with the `clearml` package installed):
```bash
python -m services.visualization.clearml.clearml_smoke_test
```
It builds a tracker for `default_project` in master.yaml and prints `scope`, `enabled` and `failure_reason`.

Server deployment notes for a self-hosted ClearML stack are in `readme_clear_ml_setup.md` (host-specific paths).

## Configuration

Per project, in `services/side_channel_core/config/master.yaml`:
```yaml
visualization:
  clearml:
    enabled: false
    scope: "global"                 # selects profiles.<scope>
    fail_fast_on_unreachable_api: true
    profiles:
      global: {web_url, api_url, files_url, verify_certificate,
               access_key_name, secret_key_name, clearml_env_file,
               healthcheck_enabled, healthcheck_path, healthcheck_timeout_sec}
      local:  {... same keys, 127.0.0.1:8080/8008/8081 ...}
```
- Credential files are referenced by `clearml_env_file` (repo `secrets/` directory: `clearml_global.env`, `clearml_local.env`). A relative path is resolved against `/app`.
- master.yaml has no dashboard keys (`visualization.ui`, `graph_manager.ui` were removed on 2026-10-19): the TraceFlow Monitor (`vscode_monitor/`) is TraceFlow's UI.
- `visualization.graph_manager` in master.yaml belongs to `services/graph_manager` (Make target `graph`), not to this service.
- Python dependency: `clearml==1.18.0` (pinned in `services/side_channel_core/infra/environment/requirements.txt`).

## Entry points
- `make panel ENV=dev` (build and install the TraceFlow Monitor), `make scaffold ENV=dev` (blueprint configs for new projects), `make status_share ENV=dev` (the status page, on master.yaml `status_share`).
- Library use only: `services.visualization.clearml.clearml_tracker`, `services.visualization.utils.ensure`.
- Manual: `python -m services.visualization.clearml.clearml_smoke_test`, `python services/visualization/utils/get_clearml_scope.py` (repo root).

## Used by
- `services/side_channel_core/src/sca/train.py` (lines 91, 102)
- `services/side_channel_core/src/sca/test.py` (lines 35, 47)
- `services/side_channel_core/utils/epoch_utils.py` (line 54)

Deployment note: `services/side_channel_core/deployment/docker/docker-compose.yaml` bind-mounts `services/side_channel_core` and `services/mlops` but not `services/visualization`, and the Dockerfile `staging` stage copies only those two services. Only the `prod` stage (`COPY . /app`) includes this directory, although train/test import it unconditionally.

## TODOs
None. `grep -rn "TODO\|FIXME" services/visualization` finds nothing.
