# tests/unit/test_data_files_contract.py
"""
What a run records, and where (2026-09-30):

- The run's own folder is the only record. There is no benchmark/ index:
  model_judge and the graph manager read the run folders.
- A per-epoch history row holds its header (epoch, timestamp, model_name,
  model_hash, process, env), exactly the metrics metrics_to_log names, and,
  under "dependencies", what a requested metric is derived from or read with
  (the base of a mean, exact counts, first crossings). Nothing else is stored.
- The training and test loops, and the final training of cross-validation,
  finish without writing anything outside the run folder.
"""
from __future__ import annotations

import json
import logging
import types

import pytest

from test_final_cv import harness  # noqa: F401  (the training-loop fixture)
from utils.seal import strip as strip_seal  # rows and registries carry their "_tfseal" (utils/seal.py)

CFG = {
    "metrics_to_log": ["mean_attack_rank_curve", "mean_fit_time_{s}"],
    "perform_attack": {"validation": {"attack_times": 1}, "test": {"attack_times": 0}},
}
CURVE = [[10, 97.0], [20, 1.0]]
TRAIN = {   # what the training loop hands over for a train row
    "mean_fit_time_s": 21.5, "fit_time_s": 21.5, "training_objective": 5.5,
    "train_loss_reduction": "mean", "running_train_accuracy": None, "evaluated_traces": 50000,
}


# ------------------------------------------------------------------ the split
def test_only_requested_columns_are_metrics():
    from sca.metrics import split_requested

    metrics, dependencies = split_requested(TRAIN, CFG)
    assert metrics == {"mean_fit_time_s": 21.5}
    assert dependencies == {"fit_time_s": 21.5}          # the base of the requested mean


def test_parameterised_requests_and_their_companions():
    from sca.metrics import split_requested

    cfg = {"metrics_to_log": [{"traces_to_ge_k": {"k": 5}}, "success_rate", "mean_accuracy",
                              {"rank_percentile": {"p": 25}}, {"sr_curve": {"k": 10}}, "guessing_entropy"]}
    results = {
        "traces_to_ge_5": 40, "first_traces_to_ge_5": 30, "success_rate": 1.0, "sr_at_1_successes": 1,
        "sr_at_1_trials": 1, "mean_accuracy": 0.5, "accuracy": 0.5, "accuracy_correct": 5, "accuracy_total": 10,
        "rank_25_percentile": 3.0, "sr_curve@10": CURVE, "guessing_entropy": 1.0, "selected_epoch": 50,
        "validation_objective": 5.4,
    }
    metrics, dependencies = split_requested(results, cfg)
    assert set(metrics) == {"traces_to_ge_5", "success_rate", "mean_accuracy", "rank_25_percentile",
                            "sr_curve@10", "guessing_entropy"}
    assert set(dependencies) == {"first_traces_to_ge_5", "sr_at_1_successes", "sr_at_1_trials", "accuracy",
                                 "accuracy_correct", "accuracy_total", "selected_epoch"}


# -------------------------------------------------------------- history rows
def test_history_row_holds_the_header_the_requested_metrics_and_their_dependencies(tmp_path, monkeypatch):
    from utils import metrics_logger

    monkeypatch.setattr(metrics_logger, "resolve_path", lambda value, *a, **k: value)
    cfg = dict(CFG, paths={"metrics": str(tmp_path)}, model_name="tulip_fish", model_hash="3745aeec", env="dev")
    metrics_logger.append_metrics_row(cfg, TRAIN, process="train", epoch=4)

    row = strip_seal(json.loads((tmp_path / "scalars" / "tulip_fish_metrics_train_history.jsonl").read_text()))
    assert sorted(row) == ["dependencies", "env", "epoch", "mean_fit_time_s", "model_hash", "model_name",
                           "process", "timestamp"]
    assert row["mean_fit_time_s"] == 21.5 and row["dependencies"] == {"fit_time_s": 21.5}
    assert "running_train_accuracy" not in json.dumps(row)
    assert metrics_logger.flat_history_row(row)["fit_time_s"] == 21.5


def test_history_row_without_dependencies_has_no_dependencies_key(tmp_path, monkeypatch):
    from utils import metrics_logger

    monkeypatch.setattr(metrics_logger, "resolve_path", lambda value, *a, **k: value)
    cfg = {"metrics_to_log": ["guessing_entropy"], "paths": {"metrics": str(tmp_path)}, "model_name": "m",
           "model_hash": "h", "env": "dev"}
    metrics_logger.append_metrics_row(cfg, {"guessing_entropy": 1.0, "validation_objective": 5.0},
                                      process="validation", epoch=1)
    row = strip_seal(json.loads((tmp_path / "scalars" / "m_metrics_validation_history.jsonl").read_text()))
    assert "dependencies" not in row and row["guessing_entropy"] == 1.0 and "validation_objective" not in row


def test_no_history_row_when_the_process_has_no_requested_metric(tmp_path, monkeypatch):
    from utils import metrics_logger

    monkeypatch.setattr(metrics_logger, "resolve_path", lambda value, *a, **k: value)
    cfg = {"metrics_to_log": ["mean_attack_rank_curve"], "paths": {"metrics": str(tmp_path)},
           "model_name": "m", "model_hash": "h", "env": "dev"}
    metrics_logger.append_metrics_row(cfg, {"training_objective": 5.5, "train_loss_reduction": "mean"},
                                      process="train", epoch=1)
    assert not (tmp_path / "scalars" / "m_metrics_train_history.jsonl").exists()


def test_fold_mean_rows_follow_the_history_row_layout(tmp_path):
    from utils.cross_validation import save_fold_mean_accuracy

    rows = {"train": [{"epoch": 1, "process": "train", "n_folds": 2, "fit_time_s": [20.0, 22.0],
                       "mean_fit_time_s": 21.0, "mean_fit_time_s_std": 1.41}]}
    (path,) = save_fold_mean_accuracy(types.SimpleNamespace(fold_metrics=tmp_path), model_name="m", rows=rows)
    assert strip_seal(json.loads(path.read_text())) == {
        "epoch": 1, "process": "train", "mean_fit_time_s": 21.0,
        "dependencies": {"n_folds": 2, "fit_time_s": [20.0, 22.0], "mean_fit_time_s_std": 1.41},
    }


# ----------------------------------------------------------- readers of rows
def test_graph_loader_reads_a_dependency_column(tmp_path):
    from services.graph_manager.metric_catalog import _scalar_fields
    from services.graph_manager.metric_loader import _load_table

    path = tmp_path / "m_metrics_train_history.jsonl"
    path.write_text("".join(json.dumps(
        {"epoch": e, "process": "train", "mean_fit_time_s": 2.0 * e, "dependencies": {"fit_time_s": 2.0 * e}}
    ) + "\n" for e in (1, 2)))

    frame = _load_table(path)
    assert list(frame["fit_time_s"]) == [2.0, 4.0] and list(frame["epoch"]) == [1, 2]
    assert "dependencies" not in frame.columns
    assert _scalar_fields([path])[0] == {"mean_fit_time_s", "fit_time_s"}


# --------------------------------- the real loops write only the run folder
def test_cv_final_training_writes_its_rows_and_nothing_outside_the_run(harness, tmp_path, monkeypatch):
    """The refit of a CV run (the rank curve and mean_fit_time_{s} requested)."""
    import utils.metrics_logger as metrics_logger
    from test_final_cv import _loop_cfg, _run, _stack

    eu, _rec = harness
    monkeypatch.setattr(metrics_logger, "resolve_path", lambda template, cfg=None, **k: str(tmp_path / template))
    monkeypatch.setattr(eu, "append_metrics_row", metrics_logger.append_metrics_row)
    cfg = dict(_loop_cfg(["mean_attack_rank_curve", "mean_fit_time_{s}"], refit=True), env="dev")
    result = _run(eu, cfg, *_stack(cfg), tmp_path)

    rows = [strip_seal(json.loads(line)) for line in
            (tmp_path / "metrics" / "scalars" / "m_metrics_train_history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]
    for row in rows:
        assert sorted(row) == ["dependencies", "env", "epoch", "mean_fit_time_s", "model_hash", "model_name",
                               "process", "timestamp"]
        assert row["dependencies"] == {"fit_time_s": row["mean_fit_time_s"]}
    assert result["selected_epoch"] == 2 and result["train"]["mean_fit_time_s"] == rows[-1]["mean_fit_time_s"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["metrics"]      # no benchmark/
    assert not hasattr(eu, "update_registry")


def test_cv_final_refit_runs_to_completion(monkeypatch, tmp_path):
    """
    _run_final_refit from its first to its last line. Its heavy collaborators
    are stubbed; a name that does not exist in the function (the 2026-09-30
    NameError on fold_results) fails here instead of after the folds trained.
    """
    from utils import cross_validation_pipeline as pipeline

    calls = []
    plan = types.SimpleNamespace(spec=types.SimpleNamespace(seed=42, n_folds=2), paths="cv-paths")
    refit_result = {"epochs_completed": 3, "selected_epoch": 3}
    monkeypatch.setattr(pipeline, "update_manifest", lambda paths, patch: calls.append(("manifest", patch["status"])))
    monkeypatch.setattr(pipeline, "build_refit_runtime_config",
                        lambda cfg, resolved_epochs: {"trainer": {"epochs": resolved_epochs}})
    monkeypatch.setattr(pipeline, "build_fresh_training_stack",
                        lambda cfg, model_id, logger: (42, "model", "cpu", "optimizer", "scheduler"))
    monkeypatch.setattr(pipeline, "get_full_profiling_loader", lambda cfg, trainer, seed: "train-loader")
    monkeypatch.setattr(pipeline, "get_full_profiling_eval_loader", lambda cfg, trainer, seed: "eval-loader")
    monkeypatch.setattr(pipeline, "run_training_loop",
                        lambda **kwargs: calls.append(("train", kwargs["train_loader"], kwargs["val_loader"]))
                        or refit_result)
    monkeypatch.setattr(pipeline, "mark_cross_validation_completed",
                        lambda paths, refit_result: calls.append(("completed", paths, refit_result)))
    monkeypatch.setattr(pipeline, "cleanup_training_runtime", lambda *loaders: calls.append(("cleanup", loaders)))

    out = pipeline._run_final_refit(
        {"model_hash": "3745aeec", "paths": {}}, plan=plan, finalized={"resolved_refit_epochs": 3},
        logger=logging.getLogger("test"), model_name="meh_lollypop", model_id="ASCAD_CNNbase", env="dev",
        base_dir=tmp_path, task=None, tracker=None,
    )
    assert out is refit_result
    assert calls == [("manifest", "refitting"), ("train", "train-loader", "eval-loader"),
                     ("completed", "cv-paths", refit_result), ("cleanup", ("train-loader", "eval-loader"))]
    assert not list(tmp_path.iterdir())             # the refit itself writes no index beside the run


@pytest.mark.parametrize("module", [
    "utils.epoch_utils", "utils.cross_validation", "utils.cross_validation_pipeline",
    "utils.key_recovery_helper", "utils.delete_database", "sca.metrics",
])
def test_no_module_refers_to_a_benchmark_registry(module):
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))
    for name in ("benchmark_registry", "update_registry", "paths\"][\"benchmark", "write_benchmark",
                 "cross_validation_registry"):
        assert name not in source, f"{module} still refers to {name}"
