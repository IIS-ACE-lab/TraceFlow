# services/model_filter

Finds the trained models of a project that were trained with given
parameters. It selects models by their traits, not by their metrics, so you
can plot, re-test or compare "all CNNs trained with RMSprop below 1e-3", say,
whatever their results were. Configured by `projects/<P>/config/model_filter.yaml`.

## Usage

```bash
make find_model ENV=dev PROJECT=DLSCA_ASCAD                          # config/model_filter.yaml
make find_model ENV=dev PROJECT=DLSCA_ASCAD FILTER=config/deep.yaml  # another filter file
```

Runs on the host in the `sca` environment. Output:
`projects/<P>/artifacts/filtered_models.yaml`, overwritten on every run:

```yaml
models:                 # copy into a graph job's `models` list
- mesh_chicken
details:                # one entry per match: hash, model_id, env and the queried values
- model_name: mesh_chicken
  model_hash: 30f464fd
  values: {model.architecture: CNN, trainer.optimizer.type: RMSprop}
```

The terminal shows a table of every candidate run with the value each has at
every queried path (green = match, `null` = the path does not exist for that
run), so a filter that matches nothing can be adjusted from what is there.

For the panel's FIND MODEL form (`model_filter.yaml` under EXPERIMENT SETTINGS):

```bash
python services/model_filter/model_filter.py --project P --fields                  # JSON: every field the runs hold
python services/model_filter/model_filter.py --project P --write-where '[{"path": "batches", "contains": "fig26"}]'
```

`--fields` lists every path a condition can name, from every run of `run_map.jsonl`: where the
filter finds it (run entry, registry, sweep point, snapshot, device record; the first source that
has it wins, as a condition reads it), its kind (number, text, bool, list), how many runs have it,
whether it differs between them and its values (up to 40); a list of blocks shows as `*`
(`model.architecture_args.layers.*.Conv1d.kernel_size`). `--write-where` writes `model_filter.where`
in block style, as by hand, each condition checked as the filter reads it; the rest of the file
and the comments after the list stay.

## How a model is selected

```text
config/model_filter.yaml ──> load_filter_config ──> for each run in artifacts/run_map.jsonl
   (match, status, env,        (schema, operators)     RunView.values(path)  ──> evaluate ──> filtered_models.yaml
    where: [conditions])                               1. run_map entry
                                                       2. its `sampled` sweep point
                                                       3. the training snapshot
```

A parameter path is looked up in the run's `run_map.jsonl` entry first
(`env`, `status`, `model_id`, `trial_index`, …), then in its sweep point
(`model.*`, `trainer.*`, `sca.*`), then in the run's training snapshot
`generated_models/<model>/config/<model>_params.yaml` (`data.*`,
`cross_validation.*`, `perform_attack.*`, `evaluation.*`, `seed`, …). Any key
of the resolved experiment config can therefore be a condition. Only runs
whose training `status` is `DONE` (default) and whose `env` matches the make
call are candidates.

| Config field | Values |
|---|---|
| `match` | `all` (every condition must hold) or `any` |
| `status` | run_map status to consider; default `DONE`; `any` for every status |
| `env` | `null` = the ENV of the make call; `any` = every environment; or a name |
| `where[].path` | dotted path; `*` expands a list (`model.architecture_args.layers.*.Conv1d.kernel_size`), a number indexes it (`data.test_range.1`) |
| `where[].<operator>` | exactly one of the operators below |
| `where[].quantifier` | for `*` paths: `any` (default) value satisfies the operator, or `all` must |

| Operator | Takes | True when |
|---|---|---|
| `equals`, `not_equals` | any value | the value is (not) equal; numbers compare numerically |
| `in`, `not_in` | list | the value is (not) one of them |
| `gt`, `gte`, `lt`, `lte` | number | numeric comparison; non-numbers never match |
| `between` | `[low, high]` | low ≤ value ≤ high |
| `matches` | regular expression | the whole value, as text, matches |
| `contains` | value | a list contains it, or a string contains it as substring |
| `exists` | `true` / `false` | the path is (not) present for the run |
| `count` | N or `{min, max}` | the list (or mapping/string) length is in range |

Example:

```yaml
model_filter:
  match: all
  where:
    - path: model.architecture
      equals: CNN
    - path: model.architecture_args.layers
      count: {min: 10}
    - path: trainer.optimizer.learning_rate
      between: [1.0e-6, 1.0e-3]
    - path: sca.key_enumeration.enabled
      equals: false
```

## Contents

```text
services/model_filter/
├── model_filter.py   # CLI entry; config validation, path lookup, operators, table, output
├── README.md         # this file
└── __init__.py
```

## Depends on

`services/side_channel_core/utils`: `run_map` (reads `run_map.jsonl`),
`logger` (terminal table), `timestamps` (`generated_at`).

## Used by

- Repo-root `Makefile`: `make find_model`.
- The TraceFlow Monitor's FIND MODEL form (`--fields`, `--write-where`).
- `services/side_channel_core/tests/unit/test_model_filter.py`.
- `filtered_models.yaml` is meant for `projects/<P>/config/graph_config.yaml`
  jobs (`models:` lists) and `config/test_config.yaml` (`test_request.model`).

## TODOs

None.
