# services/mlops/adapters
Concrete backend for the hypothesis-driven AutoMLOps core: registry persistence on disk. Not on any live execution path.

## Contents
```text
adapters/
├── __init__.py                 # empty
└── disk_registry_backend.py    # DiskRegistryBackend: <base_dir>/<namespace>.json, atomic write, .bak, lock
```

## Used by
- `disk_registry_backend.py`: imported only by `services/mlops/core/auto_mlops.py`, which nothing imports.

## TODOs
None.
