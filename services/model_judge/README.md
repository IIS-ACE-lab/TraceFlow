# services/model_judge

Ranks every trained model of a project and selects the best one, from the
run folders and the two lifecycle registries. Configured by `projects/<P>/config/model_judge.yaml`.

## Usage

```bash
make judge ENV=dev PROJECT=DLSCA_ASCAD           # rank and select
make rerun-judged ENV=dev PROJECT=DLSCA_ASCAD    # re-run the judged candidates
```

Runs on the host in the `sca` conda environment. Output:
`projects/<P>/model_selection/` — `latest_selection.json`, `leaderboard.json`,
`selected_model.yaml`, `selected_candidates.yaml` (with a plain
`candidate_list`). It also rewrites job 0's `top_models` in
`projects/<P>/config/graph_config.yaml`.

## How a model is chosen

```text
artifacts/run_registry.json, test_registry.json ─┐
<run>/model/<name>_checkpoint_manifest.json      ─┤
<run>/metrics/attack_metrics/scalars/*_history.jsonl ─┼─> run_reader ─> filters ─> ranking ─> report_writer
<run>/metrics/key_recovery/{validation_best,test}.json ─┤  (one candidate   (require_done,   (primary +
<run>/cross_validation/aggregate/                 ─┘   per run)         require_processes) tie-breakers)
```

There is no separate results index: `run_reader.py` reads each registered
run's own folder (`path` in `run_registry.json`), the files the graph manager
plots from. A run whose folder is missing, or that recorded nothing yet, is
not a candidate.

Rankable names, per process:

- `train`, `validation`: the columns of the history row of the tested epoch
  (the best-validation epoch when validation checkpointing is on and
  `evaluation.test.load_model` is `best_validation`, otherwise the final
  epoch), a row's `dependencies` included (`fit_time_s`), plus
  `selected_epoch`. `test`: the columns of the history row of the run's latest
  All test (the training's own, or a later Retest All / Test All);
  `test_budget`: those of its latest Budget test ([`utils/latest_test.py`](../side_channel_core/utils/latest_test.py)).
  A test that did not finish is never read, and the leaderboard's `tests` names
  the test each process was read from (`recorded` or `<model_hash>_et_<index>`).
- `validation`, `test`, `test_budget`: the key-recovery report as well:
  `final_true_key_rank`, `key_rank_reported`, `key_enumeration_position`,
  `key_recovery`, `key_recovered_through`, `final_ge`, `final_sr_at_1`,
  `first_rank1_trace`, `stable_rank1_trace`, `top1_top2_score_margin`.
- `cross_validation`: the headline of `aggregate/summary.yaml`
  (`mean_curve_final_ge`, `mean_curve_traces`, `traces_to_ge_1_mean_curve`,
  `accuracy_mean`, `mean_fit_time_s`, ...), the crossings of
  `mean_rank_curve.json` (`traces_to_ge_<k>_mean_curve`,
  `first_traces_to_ge_<k>_mean_curve`, `folds_reaching_ge_<k>_fraction`), the
  fold mean of every scalar (`<metric>_train`, `<metric>_validation`, and
  plain `<metric>` for the held-out value, a fit time being the training
  one), and from the diagnostics report `folds_better_than_chance`,
  `normalized_rank_advantage`, `endpoint_rank_std`.

A value that is not a number (an array pointer, a text, a bool) is not
rankable; `get_metric()` treats it as missing. A process is present when the
run recorded something for it: a cross-validation run has no `validation`
process, and a test that is not attacked and records no metric has no `test`
process.

| Config field | Values |
|---|---|
| `ranking.primary`, `ranking.tie_breakers[]` | `{process, metric}`; process `train`, `validation`, `test`, `test_budget` or `cross_validation` |
| `report_metrics[]` | shown in the leaderboard, never ranked |
| `filters.require_done` | model's train and test both DONE |
| `filters.require_processes` | processes a model must have recorded |
| `ranking.N_top_candidates` | leaderboard length, and the size of job 0's `top_models` |

Direction is automatic: accuracies, success rates, `folds_*` fractions and
`normalized_rank_advantage` are maximised, everything else minimised; a model
missing a ranking metric sorts last. The direction is read through the names
`run_reader.py` builds: `mean_`/`final_` and `_train`/`_validation`/`_test`/
`_mean` keep the metric's direction (`mean_accuracy_validation` is maximised),
and a spread (`accuracy_std`, `training_time_std_s`) is always minimised. Ranking on `test` or `test_budget` is allowed but flagged
as biasing the reported test result. Ranking on a `train` quality metric
(accuracy, loss) is allowed but warned and listed as
`selection_uses_train_quality_metrics` in `latest_selection.json`; train costs
such as `fit_time_s` and `trainable_parameters` are not flagged. Rank metrics are never computed on
training traces. A process or metric that no model recorded prints a hint and
a colored table of the available choices.

## Contents

```text
services/model_judge/
├── model_judge.py       # CLI entry; config validation, filters, ranking, hints
├── run_reader.py        # one candidate per registered run, read from its run folder
├── report_writer.py     # atomic writes of the four output files
└── __init__.py
```

## Depends on

`services/side_channel_core/utils`: `logger` (tables and hints),
`graph_config_sync` (job 0 `top_models`).

## Used by

- Repo-root `Makefile`: `make judge`, `make rerun-judged` (reads
  `model_selection/latest_selection.json`).
- `services/side_channel_core/tests/unit/test_model_judge.py`, `test_final_cv.py`
  (train quality metrics flagged), `test_final_orchestration.py` (a row with a
  `$ref` metric).

## TODOs

None.
