# services/visualization/utils
Helpers for the ClearML integration (config refresh from master.yaml, URL probing, trial-suffix naming) and the colour themes of TraceFlow's figures. Four files are live; the rest are empty placeholders.

## Contents
```text
utils/
├── __init__.py            # empty
├── ensure.py              # _ensure_visualization_cfg(): overwrite run_cfg["visualization"] from master.yaml (live)
├── fig_theme.py           # colour themes of TraceFlow's figures: page, panel, ink, muted and eleven palette slots (live)
├── get_clearml_scope.py   # script: prints visualization.clearml.scope for $PROJECT/default_project (unused)
├── helpers.py             # _extract_trial_suffix, _probe_tcp, debug_resolve (live, via clearml_tracker)
├── io.py                  # empty placeholder
├── jsonl.py               # empty placeholder
├── paths.py               # empty placeholder
└── time.py                # empty placeholder
```

- `ensure.py` finds master.yaml via `$SCC_MASTER_YAML`, else `<repo>/services/side_channel_core/config/master.yaml`.
- `get_clearml_scope.py` uses a relative path `services/side_channel_core/config/master.yaml`, so it must run from the repo root.
- `fig_theme.py` (moved here from `paper/draft_4/figures/system/` on 2026-10-21): 24 themes, `python services/visualization/utils/fig_theme.py` lists them. A figure maps its roles to palette slots (`resolve`), so a theme added here reaches every figure. Added with the move: `traceflow` (the planner's light colours, the Download's default), `cyberpunk` (TraceFlow at night: the panel's dark ground, its neon status lights and the share button's glowing blue, with magenta and cyan), `okabe_ito` (colour-blind safe), `grayscale` (black-and-white print), `nord`; then `solarized`, `solarized_dark`, `dracula`, `monokai`, `gruvbox`, `tokyo_night` (the editors' palettes; the dark ones for slides) and `viridis` (perceptually even, readable in grey).

## Used by
- `ensure.py`: `services/side_channel_core/src/sca/train.py:102`, `services/side_channel_core/src/sca/test.py:47` (called at `test.py:143`, `test.py:390`).
- `helpers.py`: `services/visualization/clearml/clearml_tracker.py:62`.
- `fig_theme.py`: `paper/draft_4/figures/system/*.py` (every system figure; `paper/draft_4/Makefile` sets each one's theme), `services/visualization/vscode_monitor/model_diagram.py` (the planner's Download) and `choices.py` (its theme menu). Older drafts (`paper/draft_2`, `paper/draft_3`) keep their own copies.
- `get_clearml_scope.py`: not referenced by any script, Makefile or module.
- `io.py`, `jsonl.py`, `paths.py`, `time.py`: not referenced; empty.

## TODOs
None.
