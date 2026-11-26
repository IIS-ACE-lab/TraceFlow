# services/side_channel_core/scripts
Shell entry points: in-container orchestration (dispatch, retry/resume loop, snapshot rerun), GPU/process hygiene, and host setup. Container scripts assume the image layout (`/app`, `/venv`).

## Contents
```text
scripts/
├── __init__.py                    # package marker (empty)
├── orchestrator_auto_mlops.sh     # container entrypoint: dispatch test request / auto_search / auto_mlops / rerun / manual
├── orchestrator.sh                # manual SCC loop: core.cli orchestrate-cmd until all trials DONE; retry/resume/force
├── orchestrator_rerun.sh          # replay RERUN_EXPERIMENT from its config snapshot (SHA256), REPEAT trials
├── cleanup.sh                     # in-container cleanup of python/torch/GPU state; --lock-file, --no-exit-on-error
├── cleanup_all.sh                 # kill all python processes, torch cleanup
├── pre_training_cleanup.sh        # kill own python processes, clear CUDA caches
├── gpu_reset.sh                   # GPU recovery: persistenced, stale contexts, cache clean
├── gpu_watchdog.sh                # periodic GPU check; runs cleanup.sh + gpu_reset.sh on failure
├── healthcheck.sh                 # nvidia-smi / nvcc / torch.cuda checks
├── setup.sh                       # host setup entry point; runs infra/environment/setup_sca_env.sh
├── run_cicd.sh                    # runs ci_cd stage scripts
├── init.sh                        # empty
├── README.md                      # this file
└── README.pdf                     # old rendered README
```

Host setup: `bash services/side_channel_core/scripts/setup.sh` creates the `sca` conda environment (Python 3.12) through `infra/environment/setup_sca_env.sh`.

## Used by
- `orchestrator_auto_mlops.sh`: `deployment/docker/docker-compose.yaml` (`command`), `deployment/docker/Dockerfile` (`CMD`).
- `orchestrator.sh`, `orchestrator_rerun.sh`: `orchestrator_auto_mlops.sh`.
- `cleanup.sh`: `orchestrator.sh` (as `./scripts/cleanup.sh` and `/app/scripts/cleanup.sh`), `gpu_watchdog.sh`.
- `gpu_reset.sh`: root `Makefile` (`reset`), `gpu_watchdog.sh`.
- `healthcheck.sh`: root `Makefile` (`healthcheck`).
- `setup.sh`: run by hand.
- Not called anywhere: `cleanup_all.sh`, `pre_training_cleanup.sh`, `gpu_watchdog.sh`, `run_cicd.sh`, `init.sh`.

## TODOs
None.
