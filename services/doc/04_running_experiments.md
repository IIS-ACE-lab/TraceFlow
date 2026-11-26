# 04 · Running experiments (`make orchestrate`, `make rerun`)

Configuration errors stop a run before training. The error names the
key; the fixes below cover the common ones. Every key is documented in
[`services/side_channel_core/config/blueprint/params.yaml`](../side_channel_core/config/blueprint/params.yaml).

## 4.1 `ENV` missing or invalid

Every target needs `ENV=dev|staging|prod`:

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD
```

## 4.2 `HY is required because auto_mlops.enabled=true in master.yaml`

`auto_mlops.enabled` is `true` in
`services/side_channel_core/config/master.yaml`. Either pass `HY=H1`…`H4`,
or set `auto_mlops.enabled: false` — the AutoMLOps path is marked
undeveloped in `master.yaml`, so `false` is the normal setting.

## 4.3 `Project not found in master.yaml` / the wrong project runs

`PROJECT` must match a `name:` under `project:` in `master.yaml`. Without
`PROJECT`, `default_project` is used. Register a new project as described
in the root [README](../../README.md#3-configuration-files).

## 4.4 Nothing runs: all trials are DONE

The orchestrator skips every trial whose hash the registry
(`projects/<P>/artifacts/run_registry.json`) marks DONE, and prints
`ALL SCC trials complete for ENV=…`. To run them again:

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD FORCE=1          # re-run scoped trials
make rerun ENV=dev PROJECT=DLSCA_ASCAD <model_name> REPEAT=2   # one model, twice
```

`FORCE=1` retrains every scoped trial once per orchestration session and takes precedence over
the run_map resume/retry passes. At the start of the session every run of `run_registry.json`
in scope (ENV + MODEL_ID; DONE, FAILED or INTERRUPTED) is marked `queued` in `run_registry.json`
and `test_registry.json` and numbered 1..N (`trial_index`) in its previous `trial_index` order
(ties: earlier `started_at` first). The queue is retrained in that order; only then does the
planner generate new hashes. A trial that fails under `FORCE` is not retried in the same session
(`FORCE` bypasses the retry budget). Stopping the run (Ctrl-C) marks the waiting queue
INTERRUPTED; a DONE run whose config cannot be reproduced stays DONE.

A changed value in `params.yaml` gives a new hash and runs without `FORCE`.

A run reads the config as it was when the run started. `make experiment` copies the project's
`config/` folder at queue start to `config/.frozen/<batch_id>/`; a lone `make orchestrate`
(`make reproduce`) or `make test` copies it at its own start (`utils/config_freeze.py`). Every command, orchestrator pass, model and test of
the run reads `params.yaml`, the ENV file, `test_config.yaml`, `auto_search.yaml`,
`model_config.yaml` and the extra configs from that copy (`TRACEFLOW_FROZEN_CONFIG`, relative to
the project's root, for the project `TRACEFLOW_FROZEN_PROJECT` names), so an edit made while it
runs reaches the next run. The copy plans the same sweep and the same hashes as the files; it is
removed when the run ends, and each run's record stays its sealed `params_out` snapshot.

## 4.5 The wrong dataset or model is used

With several `data:` entries, `DATASET` chooses one; it defaults to
`ascad`. With several `model:` blocks, `MODEL_ID` chooses one:

```bash
make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 MODEL_ID=ASCAD_CNNbest
```

Names must match `data[].name` and `model[].model_id` exactly.

## 4.6 Far more trials than expected

Every YAML list in `params.yaml` is a sweep dimension, and the trial count
is the product of the list lengths. Shorten the lists to one value each
for a single run. Blocks that never sweep: `sca`, `trainer` pairs
(`betas`, …), `cross_validation`, `paths`, `logging`, `graph`,
`visualization`, data ranges, `metrics_to_log`, `ge_thresholds`.

The per-dataset keys under `data:` do not sweep either, although the
blueprint marks them `[sweep]`: a list there is not expanded, and for
`profiling_desync`, `attack_desync`, `multilabel` or
`trace_normalization.mode` it fails type coercion (repo-root
[`TODO.md`](../../TODO.md) §21). Give one value per dataset entry.

## 4.7 `sca.key_enumeration.enabled=true is not supported while cross_validation.enabled=true`

Cross-validation keeps key recovery per fold, so there is no single
validation ranking to enumerate. Disable one of the two:

```yaml
cross_validation:
  enabled: false        # to use key enumeration
# or
sca:
  key_enumeration:
    enabled: false      # to keep cross-validation
```

## 4.8 `best_rank` is retired

`sca.key_enumeration.strategy: best_rank` was removed. Use one of
`local_hamming`, `neighbour_sweep`, `support_weighted`.

## 4.9 Data errors

| Message | Cause | Fix |
|---|---|---|
| `Dataset '<name>' has invalid database path` | `data[].database` does not point to a file | Give data[].database relative to the repo root (services/side_channel_core/data/...) or the project root; it resolves from any checkout (host or container). Check the HDF5 file exists |
| `Dataset '<name>' is missing database: key` | `data[]` entry without `database` | Add it |
| Train/validation/test range overlap | `train_range`, `val_range`, `test_range` intersect | Make them disjoint within the file's trace count |

## 4.10 Named architectures (`ResNet50_1D`, `InceptionV3_1D`)

These take no `layers:` list; the resolver builds them and sets
`num_classes` from the leakage model (256 for ID, 9 for HW). A `layers:`
list or a mismatching `num_classes` is rejected.

`architecture_args.weight_init` (e.g. `glorot_uniform`) is applied after
construction to every Conv/Linear layer, with zero biases; BatchNorm keeps
γ = 1, β = 0. Without it the build logs a WARNING and keeps PyTorch's
default initialisation. An unknown `architecture_args` key is rejected with
the closest valid one.

## 4.11 CUDA out of memory

Lower `trainer.batch_size`, or run a smaller model first. Free the GPU
between runs:

```bash
make reset ENV=dev
nvidia-smi
```

## 4.12 The run failed; where is the log?

| Where | What |
|---|---|
| Terminal | the full orchestrator and training output |
| `projects/<P>/artifacts/<env>/generated_models/<model>/logs/` | per-model training log |
| `projects/<P>/artifacts/run_registry.json` | status of every trial (`DONE`, `FAILED`, …) |

After a failure the stack keeps its volumes; clean up with
`make stack-nuke ENV=dev` ([02](02_container_runtime_and_image_build.md#27-orchestrator-failed-volumes-preserved-for-debugging)).

## 4.13 My change to `services/visualization` has no effect in the container

The container bind-mounts `side_channel_core`, `mlops` and `projects`
from the host, but `services/visualization` comes from the image, copied
at build time. Rebuild after changing it:

```bash
make orchestrate ENV=dev PROJECT=<P> FORCE=1
```

## 4.14 `Extra config '<name>' not found at …` / `ENV config file for '<env>' not found`

Every file the project lists under `extra_configs` in `master.yaml`, and
`config/<env>.yaml`, must exist. Create the missing file from
`services/side_channel_core/config/blueprint/` or remove its entry from
`extra_configs`.

## 4.15 Re-testing models (`make test`)

```bash
make test ENV=dev PROJECT=DLSCA_ASCAD MODEL=model_a,model_b
make test ENV=dev PROJECT=DLSCA_ASCAD          # names listed in config/test_config.yaml
```

`projects/<P>/config/test_config.yaml` lists only model names
(`test_request.model`: one name or a list). Each name is looked up in
`artifacts/model_name_map.json` (hash) and `artifacts/run_map.jsonl` (trained
run); the model is tested through `core/cli.py test-cmd` from its own
training snapshot, so on exactly the parameters it was trained with. The
result replaces its recorded test: its test metric row,
`metrics/key_recovery/test.json` and `artifacts/test_registry.json`. The host checks every name before any image
is built.

Everything is fresh: the model is run on its test traces again, and key
recovery (with key enumeration, if the model uses it, under the strategy
fixed at training) is computed from that new ranking. Only the test is
reported (`KEY RECOVERY SUMMARY - TEST`); `validation_best.json` and
`run_registry.json` from training are not touched.

| Message | Fix |
|---|---|
| `No model to test` | Pass `MODEL=`, or list names under `test_request.model` |
| `<name>: not in …/model_name_map.json` | Use a name from `artifacts/done_model_names.yaml`; the message lists the trained models |
| `training status FAILED, not DONE` | Only finished models can be tested; re-run training (`make rerun`) |
| `trained with ENV=staging` | Run with the ENV the model was trained in |
| `Unknown key(s) in test_request` | The file takes only `model:` |
| `Refusing to run on bare metal` | Run through `make test`; the CLI runs only in the container |
| `model_name_map.json is unreadable or not valid JSON` / `maps X and Y to the same model_name` | Restore `artifacts/model_name_map.json` from a backup or fix it by hand; it is never rewritten while damaged |
| `params_out snapshot ... truncated or damaged` / `does not match its snapshot_sha256` | Restore the snapshot, or re-run the model as a new run (`make delete_database ... MODEL=<name> CONFIRM=1`, then `make orchestrate`) |
| `Snapshot of <name> hashes to X, not to its registered model_hash Y` | The snapshot or `master.yaml` `scripts:` changed since training; restore them or re-run as a new run |

The same snapshot checks run when `make orchestrate` retries or resumes a
registered run. A run registered before 2026-09-29 is refused there with
`registered before 2026-09-29 ... cannot be re-run under that hash`: its
old identity included absolute paths, so re-run it as a new experiment
(`make delete_database ... MODEL=<name> CONFIRM=1`, then `make orchestrate`).
It can still be tested with `make test`.

## 4.16 Timestamps are in the wrong timezone

Every timestamp TraceFlow writes (registries, reports, metric histories,
logs) uses the zone from `master.yaml`:

```yaml
timezone:
  current_location: "Taipei"    # a city, or an IANA name such as Asia/Taipei
  current_country: "Taiwan"     # a country name or ISO code
```

`utils/timestamps.py` resolves it to an IANA zone (Taipei, Taiwan ->
Asia/Taipei). `Cannot tell which zone of <country> contains <city>` means
the country has several clocks and the city is not a zone name: set
`current_location` to one of the zones the message lists. Check the result:

```bash
PYTHONPATH=services/side_channel_core python -c "from utils.timestamps import timezone_name, now_iso; print(timezone_name(), now_iso())"
```

## 4.17 `architecture_args.weight_init is required when strict_graph_args is true`

A layer model (`CNN`, `MLP`, …) with `strict_graph_args: true`, or without
the key (strict is the default), must name its initialisation:

```yaml
architecture_args:
  strict_graph_args: true
  weight_init: "glorot_uniform"   # the Keras default the ASCAD models trained with
```

The modes are listed in the message and in
`config/blueprint/params.yaml`: `kaiming_uniform` (PyTorch's default),
`glorot_uniform`, `glorot_normal`, `he_uniform`, `he_normal`,
`lecun_uniform`, `lecun_normal` (Keras >= 2.2.0 rule), `lecun_normal_keras`
(Keras 2.1.1 rule) and `random_normal`. `strict_graph_args: false` accepts
PyTorch's default initialisation with a WARNING. A misspelled key
(`weight_int`) is named with its closest valid key.

## 4.18 Cross-validation settings rejected or failing

| Message | Cause | Fix |
|---|---|---|
| `trainer.optimizer.scheduler.args ['total_steps'] cannot be used with cross_validation.enabled=true` | a OneCycleLR length written in `args` is one number, but the folds and the refit train on different numbers of traces | remove `total_steps`, `epochs`, `steps_per_epoch`: each loop gives OneCycleLR its own length. Or none, or a per-epoch scheduler |
| `trainer.optimizer.scheduler.type=CyclicLR cannot be used with cross_validation.enabled=true` | its step sizes count optimizer steps; folds and refit take different numbers per epoch | OneCycleLR without a length, none, or a per-epoch scheduler; CyclicLR without cross-validation |
| `trainer.early_stopping cannot be used with OneCycleLR` (any training) | stopping early leaves the cycle unfinished | `trainer.early_stopping: false`, or another scheduler |
| `... is not a keras.optimizers.schedules class: it is a torch.optim.lr_scheduler class, available under deep_learning.framework: pytorch` | OneCycleLR (or another torch scheduler) under Keras | `deep_learning.framework: pytorch`, or a `keras.optimizers.schedules` class |
| `perform_attack.validation.attack_times=<n> does not divide the held-out fold size <m>`, before any fold trains | `attack_distribution: disjoint` under cross-validation splits each held-out fold into `attack_times` attacks of equal size | Choose `attack_times` dividing the held-out fold size, or `attack_distribution: full_population` |
| `cross_validation.partition_seed is set but cross_validation.shuffle is false` | unshuffled folds are contiguous blocks and take no seed | `shuffle: true`, or remove `partition_seed` |
| `cross_validation.refit.enabled: false trains no model on all profiling traces, but the test attacks that model`, before any fold trains | the test attacks the refit's model and nothing else | `cross_validation.refit.enabled: true`, or `perform_attack.test.attack_times: 0` for the CV results only |
| WARNING `cross_validation.refit.enabled: true trains a final model ... perform_attack.test.attack_times: 0 never attacks it` | the refit (the default; absent means true) trains a model no result uses | `cross_validation.refit: {enabled: false}` skips it; it is not part of the `model_hash`, so no run trains again. Or test it: `attack_times >= 1` |
| `<model> cannot be tested: cross_validation.refit.enabled: false ...` (`make test`, an external test) | the run kept its CV results only | train the configuration without cross-validation (a new `model_hash`) to test a model |

## 4.19 `ConfigError: <env>.yaml: <key> is not a key of params.yaml ...`

`dev.yaml`, `staging.yaml` and `prod.yaml` (`master.yaml` `env_configs`) override `params.yaml`
when a command runs with that `ENV`, key by key (`utils/config_loader.apply_env_overrides`):

- only a key `params.yaml` (or the project's extra configs: `graph`, `test`) has, and only that key;
  the other keys of its block keep their `params.yaml` values;
- only a key given a value: `epochs:`, `[]`, `""` and `{}` override nothing; `0` and `false` are values;
- one value for a list is a one-value sweep, as in `params.yaml`.

| Message | Cause | Fix |
|---|---|---|
| `staging.yaml: trainer.epoch is not a key of params.yaml or of the project's extra configs: an override replaces an existing key only; did you mean trainer.epochs?` | a misspelled key, or a key nothing reads (`clearml`, `logging`, `debug`, `outputs` were such keys until 2026-10-18) | the key as `params.yaml` names it, or remove it |
| `dev.yaml: trainer is a mapping in params.yaml; its override must be one too (got a value)` | a value where `params.yaml` has a block, or a block where it has a value or a list | the block's own keys (`trainer: {epochs: [2]}`) |
| `dev.yaml must hold a mapping of params.yaml keys` | the file is a list or a value | keys at the top level, as in `params.yaml` |

## 4.20 `ConfigError: <file>: <key> is not a key TraceFlow reads ...`

Every configuration file is checked for keys no code reads, before anything runs
(`services/side_channel_core/utils/config_schema.py`): a misspelled key, or one that was
renamed or removed, would otherwise change nothing without a word. The message names the
file, the key path (list items with their index) and the closest known key:

```text
ConfigError: params.yaml: validation.rank_eval_intervall is not a key TraceFlow reads; did you mean validation.rank_eval_interval? Keys here: rank_eval_interval, rank_step
```

| File | Checked by | Blocks whose keys are free (names, function arguments) or checked by their own reader |
|---|---|---|
| `params.yaml` | `config_loader.get_project_config` | `data[]` entries (by `type`, §4.5 / `services/doc/TUTORIAL.md`), `model[].architecture_args`, `trainer.optimizer[]` and its scheduler, `trainer.loss.args`, `deep_learning.<framework>.*_init_args` and `optimizer_args`, `sca.key_enumeration`, `metrics_to_log` |
| `master.yaml` (the top and the selected project's entry) | `get_project_config` | `scripts`, `env_configs`, `extra_configs`, ClearML `profiles` names |
| `graph_config.yaml` | `get_project_config` (`extra_configs.graph`) | `legend.labels`, `relative` (checked by `graph_job_validator.py`) |
| `model_judge.yaml` | `make judge` | none |
| `auto_search.yaml` | `get_project_config` | as `params.yaml`, plus its `search` and `tracking` blocks |
| `dev/staging/prod.yaml` | `apply_env_overrides` (§4.19) | keys must exist in `params.yaml` |
| `test_config.yaml`, `model_filter.yaml` | their readers, as before | per their own rules |

Renamed keys whose own reader names the new one (`perform_attack.*.attack_repetitions`,
`trainer.lr`, a facet's `style`, `export.dpi`) pass this check, so that reader's message
still tells where the key went. Run snapshots (`params_out`) are never checked: they are read
directly and hold resolved keys. Every configuration error (`DatasetTypeError`,
`TestRequestError`, `ModelFilterError`, `GraphJobValidationError`, `FacetValidationError`) is a
`ConfigError` (`utils/config_errors.py`).
