# services/side_channel_core/ci_cd
Placeholder CI/CD pipeline: a Python stage runner driven by `config/pipeline_config.yaml` and three stage scripts. Build and deploy stages only write marker files; experiments are built and run by the root `Makefile` instead.

## Contents
```text
ci_cd/
├── __init__.py          # package marker (empty)
├── pipeline_utils.py    # run stages from config/pipeline_config.yaml, log success/failure
├── build.sh             # stub: writes ./artifacts/build.log
├── test.sh              # pytest on ../tests, log to ../logs/test_results.log
├── deploy.sh            # stub: touches ./logs/deploy_success_<ts>.log
├── pytest.ini           # pytest options (testpaths ../tests, pythonpath ../backend)
└── README.md            # this file
```

Local run of the runner (from `services/`): `PYTHONPATH=$(pwd) python side_channel_core/ci_cd/pipeline_utils.py`. It also reads `config/params.yaml`, which does not exist in `side_channel_core/config/`, so it currently fails at import.

## Used by
- `pipeline_utils.py` stage paths: `config/pipeline_config.yaml`.
- `scripts/run_cicd.sh` calls `test.sh`, `build.sh`, `deploy.sh` (and `python ci_cd/__init__.py`, which is empty).
- `scripts/setup.sh` makes `*.sh` here executable.

## TODOs
None.
