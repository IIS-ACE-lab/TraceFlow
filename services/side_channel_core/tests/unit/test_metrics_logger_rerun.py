# services/side_channel_core/tests/unit/test_metrics_logger_rerun.py
"""A rerun of the same model (FORCE, a resumed run) writes into the same run folder: its history rows from the
rewritten epoch on replace the old run's, and its epoch arrays follow them (audit 2026-10-25). Before, an earlier
run's later arrays stayed, and graph_manager's array_epoch: latest read the old run's curve beside the new rows."""

import numpy as np

from services.graph_manager.metric_loader import load_array_metric, load_scalar_history
from utils.metrics_logger import append_metrics_row


def _cfg(tmp_path):
    return {"paths": {"metrics": str(tmp_path / "metrics")}, "model_name": "m", "model_hash": "h", "env": "dev",
            "metrics_to_log": ["mean_attack_rank_curve", "traces_to_ge_1"]}


def _curve(value):
    return np.column_stack([np.arange(1, 4), np.full(3, value, float)])


def _arrays(tmp_path):
    return sorted(str(p.relative_to(tmp_path / "metrics")) for p in (tmp_path / "metrics").rglob("ep*.npy"))


def test_a_shorter_rerun_leaves_no_array_of_the_old_run(tmp_path):
    cfg = _cfg(tmp_path)
    for epoch in (1, 2, 3):                                            # the first run: three epochs
        append_metrics_row(cfg, {"mean_attack_rank_curve": _curve(100.0 + epoch), "traces_to_ge_1": None}, "validation", epoch)
    append_metrics_row(cfg, {"mean_attack_rank_curve": _curve(1.0), "traces_to_ge_1": 2}, "validation", 1)   # rerun: one epoch
    history, _, _ = load_scalar_history(metrics_dir=tmp_path / "metrics", model="m", process="validation", metric="traces_to_ge_1")
    assert history["epoch"].tolist() == [1]
    curve, path, _ = load_array_metric(metrics_dir=tmp_path / "metrics", model="m", process="validation",
                                       metric="mean_attack_rank_curve")
    assert path.name == "ep0001.npy" and curve[:, 1].tolist() == [1.0, 1.0, 1.0]
    assert [p.rsplit("/", 1)[-1] for p in _arrays(tmp_path)] == ["ep0001.npy"]


def test_a_resumed_run_keeps_the_arrays_before_its_first_epoch_and_other_processes_keep_theirs(tmp_path):
    cfg = _cfg(tmp_path)
    for epoch in (1, 2, 3):
        append_metrics_row(cfg, {"mean_attack_rank_curve": _curve(10.0 + epoch), "traces_to_ge_1": None}, "validation", epoch)
    append_metrics_row(cfg, {"mean_attack_rank_curve": _curve(5.0), "traces_to_ge_1": None}, "test", 1)
    append_metrics_row(cfg, {"mean_attack_rank_curve": _curve(7.0), "traces_to_ge_1": None}, "validation", 2)   # resumed at 2
    found = _arrays(tmp_path)
    assert [p for p in found if "/validation/" in p] == ["arrays/validation/mean_attack_rank_curve/ep0001.npy",
                                                         "arrays/validation/mean_attack_rank_curve/ep0002.npy"]
    assert [p for p in found if "/test/" in p] == ["arrays/test/mean_attack_rank_curve/ep0001.npy"]   # another process
    first = np.load(tmp_path / "metrics" / "arrays" / "validation" / "mean_attack_rank_curve" / "ep0001.npy")
    assert first[:, 1].tolist() == [11.0, 11.0, 11.0]                 # epoch 1 of the run that was resumed
