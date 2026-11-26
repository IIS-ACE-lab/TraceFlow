# 06 · Python imports and tests

TraceFlow imports modules three ways: `services.<svc>...` (from the
repository root), `utils...` (from `services/side_channel_core`) and
`core...` / `sca...` (from `services/side_channel_core/src`). All three
roots must be on `PYTHONPATH`. The Makefile sets this for its targets;
set it yourself when you run Python directly.

## 6.1 `ModuleNotFoundError: No module named 'services'` / `'utils'` / `'core'`

From the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src"
```

Check:

```bash
python -c "import services.side_channel_core.utils.logger, utils.logger, core.model; print('ok')"
```

## 6.2 Running the unit tests

```bash
conda activate sca
PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
  python -m pytest services/side_channel_core/tests/unit -q
```

One file: append its path, e.g.
`services/side_channel_core/tests/unit/test_key_enumeration.py`.

## 6.3 `make lint` prints many import errors

The `lint` target never fails the build (`; true`). Run pylint with the
`PYTHONPATH` above for meaningful results.

## 6.4 A running container uses an old version of my code

`side_channel_core`, `mlops` and `projects` are bind-mounted, so edits
apply to the next trial. Anything else is baked into the image; rebuild
with `FORCE=1` ([04](04_running_experiments.md#413-my-change-to-servicesvisualization-has-no-effect-in-the-container)).
