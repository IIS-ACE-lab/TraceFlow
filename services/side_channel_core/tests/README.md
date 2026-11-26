# services/side_channel_core/tests
Tests for the SCC service and the services that depend on it (graph manager, model judge).

## Contents
```text
tests/
├── __init__.py       # package marker (empty)
├── unit/             # pytest unit tests
├── integration/      # empty
├── dummy_project/    # old sample project outputs (not used by tests)
└── README.md         # this file
```

Run from the repository root, in the `sca` environment:

```bash
PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
  python -m pytest services/side_channel_core/tests/unit
```

## Used by
- `ci_cd/test.sh` (`pytest ../tests`).

## TODOs
None.
