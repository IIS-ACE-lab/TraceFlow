# services/side_channel_core/tests/unit/test_heartbeat.py
"""utils/heartbeat.py: the run's beating file for the TraceFlow Monitor."""
from __future__ import annotations

import json
import time

import pytest

from utils import heartbeat as hb


def _cv(**kw):
    base = {"n_folds": 10, "fold_epochs": 100, "refit_epochs": 100, "batches_per_epoch": 50}
    base.update(kw)
    return base


def test_progress_runs_from_zero_to_one_through_folds_refit_and_test():
    total = 10 * 100 + 100 + 1
    assert hb.progress_fraction(_cv(phase="starting")) == 0.0
    assert hb.progress_fraction(_cv(phase="fold", fold=1, epoch=1, batch=0)) == 0.0
    assert hb.progress_fraction(_cv(phase="fold", fold=1, epoch=1, batch=25)) == pytest.approx(0.5 / total)
    assert hb.progress_fraction(_cv(phase="fold", fold=3, epoch=1, batch=0)) == pytest.approx(200 / total)
    assert hb.progress_fraction(_cv(phase="refit", epoch=101, batch=0)) <= 1.0
    assert hb.progress_fraction(_cv(phase="refit", epoch=1, batch=0)) == pytest.approx(1000 / total)
    assert hb.progress_fraction(_cv(phase="test")) == pytest.approx(1100 / total)
    assert hb.progress_fraction(_cv(phase="test", status="done")) == 1.0
    points = [
        _cv(phase="fold", fold=f, epoch=e, batch=b)
        for f in (1, 5, 10) for e in (1, 50, 100) for b in (0, 25, 50)
    ] + [_cv(phase="refit", epoch=e, batch=50) for e in (1, 100)] + [_cv(phase="test")]
    values = [hb.progress_fraction(p) for p in points]
    assert values == sorted(values)


def test_progress_without_cross_validation():
    fields = {"n_folds": 0, "epochs": 10, "batches_per_epoch": 4, "phase": "train", "epoch": 6, "batch": 2}
    assert hb.progress_fraction(fields) == pytest.approx(5.5 / 11)


def test_heartbeat_beats_and_finishes(tmp_path):
    path = tmp_path / "artifacts" / "dev" / "heartbeat.json"
    beat = hb.Heartbeat(path, {"model_name": "m", "n_folds": 2, "fold_epochs": 3, "refit_epochs": 3}, interval=0.05).start()
    first = json.loads(path.read_text())
    assert first["status"] == "running" and first["model_name"] == "m"
    beat.update(phase="fold", fold=2, epoch=1, batch=0, batches_per_epoch=1)
    time.sleep(0.2)
    later = json.loads(path.read_text())
    assert later["written_ns"] > first["written_ns"]
    assert later["progress"] == pytest.approx(3 / 10)
    beat.finish("done")
    final = json.loads(path.read_text())
    assert final["status"] == "done" and final["progress"] == 1.0
    assert not beat._thread.is_alive()


def test_a_failing_write_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    beat = hb.Heartbeat(blocker / "heartbeat.json", {}, interval=0.05).start()   # parent is a file
    beat.update(phase="fold")
    beat.finish("failed")                                                        # still no exception


def test_module_calls_are_no_ops_without_a_run():
    hb.update(epoch=3)
    hb.finish("done")


def test_begin_reads_the_run_config(tmp_path):
    cfg = {"project": {"name": "P"}, "model_name": "m", "model_hash": "abcd1234", "model_id": "MLP",
           "data": {"name": "ascad"}, "cross_validation": {"enabled": True, "n_folds": 10}}
    beat = hb.begin(cfg, tmp_path, "dev", batches=["fig26"])
    try:
        data = json.loads((tmp_path / "artifacts" / "dev" / "heartbeat.json").read_text())
        assert (data["project"], data["dataset"], data["n_folds"], data["batches"]) == ("P", "ascad", 10, ["fig26"])
    finally:
        hb.finish("done")
    assert json.loads((tmp_path / "artifacts" / "dev" / "heartbeat.json").read_text())["status"] == "done"


def test_progress_never_goes_back_when_the_refit_starts(tmp_path):
    beat = hb.Heartbeat(tmp_path / "heartbeat.json", {"n_folds": 5, "fold_epochs": 10}, interval=60)
    beat.update(phase="fold", fold=5, epoch=10, batch=4, batches_per_epoch=4)
    end_of_folds = beat.snapshot()["progress"]
    assert end_of_folds == pytest.approx(50 / 61)                # refit taken as one fold's 10 epochs
    beat.update(phase="refit", epoch=1, batch=0, refit_epochs=40)
    assert beat.snapshot()["progress"] >= end_of_folds          # the real refit is longer: held, not dropped
    beat.update(epoch=40, batch=4)
    assert beat.snapshot()["progress"] == pytest.approx(90 / 91)


def test_heartbeat_is_published_read_only(tmp_path):
    import stat
    from utils.heartbeat import Heartbeat
    hb = Heartbeat(tmp_path / "heartbeat.json", {"status": "running"})
    hb.write(); hb.write()                                   # replaced over a read-only file
    assert stat.S_IMODE((tmp_path / "heartbeat.json").stat().st_mode) == 0o444
