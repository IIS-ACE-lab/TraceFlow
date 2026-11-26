# 05 · Model judge and graphs (`make judge`, `make graph`)

Both run on the host in the `sca` environment and read what the runs
recorded. Service guides:
[model_judge](../model_judge/README.md), [graph_manager](../graph_manager/README.md).

## 5.1 `PROJECT is required` / `Missing model judge config`

```bash
make judge ENV=dev PROJECT=DLSCA_ASCAD
```

needs `projects/DLSCA_ASCAD/config/model_judge.yaml`. Copy one from
another project if it is missing.

## 5.2 `Unknown ranking process` / a metric no model recorded

`ranking.primary` and `tie_breakers` take `process` ∈ {`train`,
`validation`, `test`, `test_budget`, `cross_validation`} (`test`: the run's
latest All test; `test_budget`: its latest Budget test, absent until one ran) and a metric that process
recorded. When a process or metric is missing, the judge prints a hint and
a colored "What you can choose, from your current data" table — pick a
row from it.

With `cross_validation.enabled: true` there is no `validation` process;
rank on `cross_validation` instead.

## 5.3 The judge finds no candidates

`filters.require_done: true` keeps only models whose train **and** test
are DONE, and `filters.require_processes` drops models without those
records. Check `projects/<P>/artifacts/run_registry.json`, or relax the
filters.

## 5.4 `GRAPH_ID is required`

```bash
make graph ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=1
```

`GRAPH_ID` is a job `id` in `projects/<P>/config/graph_config.yaml`.

## 5.5 `visualization.graph_manager missing for project`

The project entry in `master.yaml` has no
`visualization.graph_manager` block. Add one with `enabled: true` and
`config: "config/graph_config.yaml"`.

## 5.6 A job's models have no artifacts

The graph manager prints a red/green table of the job's models and the
names that exist. Job 0 plots `top_models`, which `make judge` rewrites;
every other job lists `models` by hand. Copy names from
`projects/<P>/artifacts/done_model_names.yaml`. A models entry may also be a
model_hash from `artifacts/run_registry.json`.

## 5.7 `GraphJobValidationError`

The job does not match the schema. The message names the field. Common
cases:

| Case | Fix |
|---|---|
| `source: cross_validation` with `process: test` | CV records only `train` and `validation` |
| `source: key_recovery` with `process: train` | key recovery exists for `validation` and `test` |
| `fold_rank_curves` or `key_candidate_scores` in an overlay job | these metrics need `type: solo` |
| Missing field | Every job lists the same fields; copy a working job |
| `cannot bind its models to trained runs of ENV=...` | each listed model_name/model_hash must be a run of that ENV in `artifacts/run_registry.json` and agree with `artifacts/model_name_map.json`; use a suggested close match, or rerun with the ENV the message names |
| `model entry ... is not text` | quote the entry; YAML reads an all-digit hash as a number |

## 5.8 A metric is missing from the plot

The metric was not recorded. Check `metrics_to_log` in the run's
`params.yaml`, then use a metric shown in the "What you can choose"
table.

Rank metrics (`mean_attack_rank_curve`, `guessing_entropy`,
`traces_to_ge_*`, …) exist only for `validation` (the CV folds) and `test`:
training traces are never attacked, so `process: train` has loss, accuracy
and fit time only.

## 5.9 `.graph_config.yaml.lock` stays next to `graph_config.yaml`

Left by an interrupted `make judge`. Delete it when no judge is running.

## 5.10 Matplotlib cache warnings

`make graph` sets `MPLCONFIGDIR=.cache/matplotlib` in the repository. If
it is not writable, fix ownership ([03](03_file_permissions.md#31-cannot-delete-or-write-project-artifacts)).

## 5.11 Choosing models by parameters (`make find_model`)

```bash
make find_model ENV=dev PROJECT=DLSCA_ASCAD            # config/model_filter.yaml
```

Lists the trained models whose parameters satisfy the `where` conditions
(architecture, optimizer, learning rate, layer count, dataset, …) and
writes them to `artifacts/filtered_models.yaml`, ready for a graph job's
`models` list. Metrics are not involved.

| Message | Fix |
|---|---|
| `Missing model filter config` | Copy `services/side_channel_core/config/blueprint/model_filter.yaml` to `projects/<P>/config/` |
| `must contain a top-level 'model_filter:' block (the file is empty)` | Fill the file from the template |
| `needs exactly one operator` | Each condition takes `path` and one operator (`equals`, `in`, `between`, …) |
| 0 matches, a column shows `null` | The path does not exist for that run; the table shows what each run has, adjust the path |
| `Permission denied: …/.filtered_models.yaml.tmp` | `artifacts/` is root-owned from container runs; see [03](03_file_permissions.md#31-cannot-delete-or-write-project-artifacts) |

## 5.12 A cross-validation curve averages the wrong folds

For `source: cross_validation` the graph manager averages every
`metrics/cross_validation/fold_*` directory of the run. It does not check the
CV manifest yet (repo-root [`TODO.md`](../../TODO.md) §19), so before
publishing such a figure check, under
`projects/<P>/artifacts/<env>/generated_models/<model>/`:

| Check | Where |
|---|---|
| the CV finished | `cross_validation/manifest.yaml`: `status: completed` and every `folds[].status` completed |
| no stale or extra folds | `metrics/cross_validation/` holds exactly `contract.n_folds` `fold_NN` directories |
| one epoch for all folds | with `array_epoch.mode: latest` each fold takes its own last epoch; use `mode: selected` (or check every fold reached the same epoch) |
