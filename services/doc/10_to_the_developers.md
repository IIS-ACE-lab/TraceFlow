# 10 · To the developers

How to change TraceFlow without breaking what it guarantees: every run is identified by its
hash, every configuration key it is given is one it reads, and every error is loud and early.
Read this before adding a key, a configuration file, a dataset type or a dependency. The
open work is in the repository root [`TODO.md`](../../TODO.md).

## Contents

1. [Ground rules](#1-ground-rules)
2. [Adding, renaming or removing a configuration key](#2-adding-renaming-or-removing-a-configuration-key)
3. [Adding a configuration file](#3-adding-a-configuration-file)
4. [The run hash](#4-the-run-hash)
5. [Errors](#5-errors)
6. [Adding a dataset type](#6-adding-a-dataset-type)
7. [The blueprint (a new project's config)](#7-the-blueprint-a-new-projects-config)
8. [Dependencies and the training image](#8-dependencies-and-the-training-image)
9. [The TraceFlow panel](#9-the-traceflow-panel)
10. [Before you merge](#10-before-you-merge)

---

## 1. Ground rules

- **Work in a git worktree**, never in the checkout an experiment runs from: the container
  mounts that checkout live, so an edit there reaches the next queued command.
  `git worktree add ../TraceFlow-<topic> -b <topic> dev`.
- **Merge between queue commands**, never while `make orchestrate|experiment|test` runs, and
  only after the full test suite passes (§10). `device.yaml` of every run records the commit.
- **Official library functions only**: activations, initialisers, losses, optimizers and
  schedulers come from `torch` / `keras`, never a hand-written port. A definition the
  installed library does not have is documented as unavailable, not written.
- **No silent fallbacks**: a setting that cannot be honoured fails with the reason and the fix
  (a CUDA device that cannot be used fails the run; it never falls back to the CPU).
- **File conventions**: every source file starts with its path as a comment
  (`# services/side_channel_core/...`) and names the authors (Shraddha Mishra); modules log
  through `utils/logger.py`.
- **`make -n` runs recipes that call `$(MAKE)`**: the Makefile sets `.ONESHELL`, so a dry run
  of such a target executes it. Read the recipe instead.

## 2. Adding, renaming or removing a configuration key

A key in a configuration file that no code reads fails the run with `ConfigError`
([guide 04 §4.20](04_running_experiments.md)). So a new key takes these steps, all in the same commit:

1. **The code that reads it**, with its default and the check of its value.
2. **Its schema entry** in [`utils/config_schema.py`](../side_channel_core/utils/config_schema.py):
   the key under its block (`PARAMS`, `MASTER_PROJECT`, `GRAPH`, `JUDGE`, `AUTO_SEARCH`). Without
   it, every configuration that uses the key is refused at load. A block whose keys are names or
   a function's arguments is `ANY` or `Named(...)`; a list of blocks is `Seq(...)`.
3. **The blueprint** ([§7](#7-the-blueprint-a-new-projects-config)): the key with an empty value
   and a commented example, or only a commented example if it is optional.
4. **The hash decision** ([§4](#4-the-run-hash)): does the key change what a run computes?
5. **A test**: the key read, its wrong values refused, and `test_config_schema.py` still green.
6. **The documentation**: the annotated `projects/DLSCA_ASCAD/config/params.yaml` comment if the
   reference project uses it, and the guide that covers its subject.

**Renaming** a key: keep the old name in the schema and make its reader refuse it with the new
name (as `perform_attack.*.attack_repetitions` and `trainer.lr` do). The schema then lets it
through, and the user reads where it went instead of "not a key".

**Removing** a key: remove the code, the schema entry and the blueprint line; the key then fails
as unknown. Check the reference project and the registered projects' files first
(`python -m pytest services/side_channel_core/tests/unit/test_config_schema.py` checks the
blueprint and `master.yaml`).

**ENV override files** (`dev/staging/prod.yaml`) need nothing extra: they may override any key
that exists in the project's `params.yaml` (`config_loader.apply_env_overrides`).

Run snapshots (`params_out`) are never schema-checked: they are read directly, hold resolved
keys, and must keep rebuilding as they were trained.

## 3. Adding a configuration file

1. Its schema in `utils/config_schema.py`, and `check_keys(...)` where the file is loaded
   (`config_loader.get_project_config` for files under `master.yaml` `extra_configs`).
2. Its template in the blueprint, every key empty (§7); its schema in `make_blueprint.py` `FILE_SCHEMAS`.
3. If a new project should get it: `layout.py` `CONFIG_ORDER` / `SCAFFOLD_FILES` and
   `make_blueprint.py` `FILES` (a test keeps the two equal).
4. Its reader raises `ConfigError` (or a subclass) for every wrong key or value.

## 4. The run hash

`utils/model_naming.make_experiment_hash` identifies a run by the sections that change what it
computes: the data entry, `model`, `trainer`, `sca.attack_byte` / `leakage_model` / the key
mode, `metrics_to_log`, `seed`, the project's `scripts` and `extra_configs`, `perform_attack`,
`evaluation`, `metrics_cfg`, `validation`, `key_recovery_reporting`, `device`, the resolved
`deep_learning` block and the cross-validation settings. A run whose hash is DONE is not trained
again.

- A key inside a hashed section changes every hash that section is part of. To keep existing
  runs comparable, add new settings so that their **absence keeps the old hash**: the hash reads
  the key only when it is present (as `deep_learning` joins only when configured, and a data
  entry's `type: ASCADf` is left out of it).
- Never change a hashed section's meaning in place: the registry would call two different
  experiments the same run.
- Check the recorded runs after such a change: `make snapshots ENV=dev PROJECT=<project>` checks
  every snapshot on disk as a re-run checks it (digest, folder, hash). A snapshot that no longer
  hashes to its recorded hash means its run would be trained again. The unit tests check the hash
  rules (`test_final_identity.py`, `test_precision.py`) on synthetic configurations, not your runs.

## 5. Errors

- Every configuration error is a `ConfigError`
  ([`utils/config_errors.py`](../side_channel_core/utils/config_errors.py)), a `ValueError`. A new
  reader's error class derives from it: `class MyReaderError(ConfigError)`. Import it from
  `services.side_channel_core.utils.config_errors`, so the `utils.*` and `services.*` import
  paths share one class.
- The message names the file, the key path and the fix ("did you mean ...?", "Set
  sca.attack_byte: [0]"). It is raised when the configuration loads, before any data is read
  or any epoch runs.
- A database that cannot be fetched or does not match its SHA-256 raises
  `DataFetchError` ("DATA FAILED: ..."), which the run records as its `status_reason`.

## 6. Adding a dataset type

[`services/side_channel_core/data/README.md`](../side_channel_core/data/README.md) "Adding a
database type" lists the steps: the type in `src/sca/dataset_types.py` `TYPES` (key mode,
required and optional entry keys, sources with SHA-256, conversion, label target, attack bytes),
its trace counts in `TRACE_COUNTS`, `window_bytes` when its stored window holds one key byte
only, a converter in `src/sca/converters/`, its submodule, and a section in
[`TUTORIAL.md`](TUTORIAL.md). Verify it as the tutorial's §8 did: the labels recomputed
independently, the attack recomputed from the saved weights, and a published model recovering
the key.

## 7. The blueprint (a new project's config)

`services/side_channel_core/config/blueprint/` is copied into `projects/<root>/config/` when a
project is declared in `master.yaml` (the panel on save, or `make scaffold ENV=dev`). The
templates are written by hand:

- every key the code reads, **every key empty** (a test checks it: only `paths`, the standard
  layout, and `metrics_cfg`, the metrics' constants, hold values), with a comment per key (the panel shows it as the key's hint); no project's
  own notes. Data entries, models, queues and single runs are empty lists: added in the panel;
  `templates.yaml` holds one data entry, model and optimizer (Add a Dataset's entry; an item's
  missing keys);
- an empty key is harmless: `config_loader` drops every empty (null) key of params.yaml after the
  ENV overrides, so it is the code's default (the panel's "(empty)"); a required key left empty is
  refused by its reader as a missing one. The ENV files' empty keys override nothing;
- **rearranging keys** (a flat key becomes a block, e.g. `trainer.early_stopping` 2026-10-22): the
  readers and the run hash keep reading the form the registered runs were hashed in; the new written
  form is folded into it by `config_loader.canonical_form` (`canonical_trainer`), the old flat keys go
  to `RETIRED` (accepted in a snapshot or an older file, not offered), the blueprint writes the new
  form, and `make_blueprint check` keeps template and schema in step. `test_config_freeze` proves the
  registered hashes unchanged;
- **adding a key the code reads** (one the panel must offer every project): its schema in
  `utils/config_schema.py`, then `python -m services.visualization.vscode_monitor.make_blueprint sync`
  puts it into the blueprint, empty (write its comment), and `check` must be clean
  (`test_config_schema` runs it: a schema key missing from the blueprint fails the tests). A key
  the readers still accept but a new project is not offered (renamed, retired) goes in
  `config_schema.RETIRED`. Every declared project's files then get the key on the next scaffold
  (`make_blueprint complete`, empty keys only; the panel's Load Blueprint for the rest). A true / false key goes in `choices.BOOLEAN_KEYS` (its menu); a key
  of a known set gets a rule in `choices.py`. The panel then draws the key, faded, in every
  project's form from the blueprint, with its menu and hint;
- `make_blueprint.py from-project` builds an emptied copy of DLSCA_ASCAD's config in memory
  only; `--write` would replace the hand-written templates.

## 8. Dependencies and the training image

- Pin every package the code imports in
  [`infra/environment/requirements.txt`](../side_channel_core/infra/environment/requirements.txt);
  PyTorch is pinned as `torch==2.9.1+cu128` (plus torchvision, torchaudio) in the same pip call
  (the Dockerfile, `setup_sca_env.sh`, `requirements.yaml`): one resolution installs every pin
  once. Check a new pin with a build log free of `Attempting uninstall` (except pip) and a clean
  `pip check`.
- The image tag is `dev-<commit>`: every new commit builds a new image on the next
  `make orchestrate` (about 5 minutes when the pip layer is cached).
- The image's clock is `master.yaml` `timezone`, resolved by `utils/timestamps.py` at build time;
  compose passes the same zone as `TZ`.
- Running the container from a worktree needs the untracked local files: copy
  `services/side_channel_core/deployment/docker/.env`; `secrets/`,
  `services/side_channel_core/deployment/harbor/harbor-certs/` and
  `services/side_channel_core/deployment/docker/tmp/` may be empty folders.

## 9. The TraceFlow panel

`services/visualization/vscode_monitor/`. After changing it, reinstall: `make panel ENV=dev`,
then *Developer: Reload Window*. Its JavaScript tests run with VS Code's bundled node:
`$NODE --test services/visualization/vscode_monitor/test/*.test.js`. Test a form change by
rendering a real configuration through the installed panel, not only the unit test.

## 10. Before you merge

```bash
PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
  CUDA_VISIBLE_DEVICES= python -m pytest -q services/side_channel_core/tests > /tmp/pytest.log 2>&1
echo "pytest exit $?"            # merge only on 0
python -m pyflakes <the files you changed>
```

- `CUDA_VISIBLE_DEVICES=` hides the GPU: with a GPU visible, five Keras tests fail on a
  device mismatch of the test itself.
- Never chain `pytest ... | tail && git commit`: the pipe hides the failure and the commit runs
  anyway. Read the exit code first.
- For a change that touches training, run one short real training of the affected path
  (`make orchestrate` with a small model) and check its `device.yaml`, its key-recovery report,
  and `make snapshots` before and after.
- Merge with a fast-forward into `dev` between queue commands; check first that none of the
  merged files has uncommitted changes in the main checkout.
