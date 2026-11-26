# services/side_channel_core/tests/dummy_project
Leftover sample project from an early (June 2025) run: an old params file and per-epoch metric CSVs. No test or script references it.

## Contents
```text
dummy_project/
├── config/     # dummy_project_params.yaml (old params, no comments)
├── data/       # params.yaml (same params, commented)
├── metrics/    # train/validation/test metric history CSVs
├── logs/       # run logs (git-ignored)
└── README.md   # this file
```

## Used by
- Nothing (`grep -rn dummy_project services` finds no references).

## TODOs
None.
