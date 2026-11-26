# services/side_channel_core/tests/unit/test_graph_marks.py
"""Graph job marks: a run's validation and test mean rank curves with the TraceFlow Monitor's three Done ranks
marked: the validation's last point, the test at the validation's trace count (the like-for-like value) and the
test's last point."""

import numpy as np
import pytest

MARKS = ["validation_final", "test_at_validation", "test_final"]


def _job(**extra):
    job = {"id": 9, "name": "ranks", "type": "process_overlay", "enabled": True, "metric": "mean_attack_rank_curve",
           "source": "attack_metrics", "models": ["run"], "processes": ["validation", "test"], "output_file_name": "r"}
    job.update(extra)
    return job


def _curves():
    """An AES_HD-shaped pair: validation 2,500 traces ending at 107, test 22,500 traces, 107.83 at 2,500, 36 last."""
    test_x = np.arange(1, 22501, dtype=float)
    test = np.column_stack([test_x, np.interp(test_x, [1, 2500, 22500], [128, 107.83, 36])])
    val_x = np.arange(1, 2501, dtype=float)
    validation = np.column_stack([val_x, np.interp(val_x, [1, 2500], [128, 107])])
    return validation, test


def test_the_marks_are_the_validation_end_the_test_at_its_count_and_the_test_end():
    from services.graph_manager.plotters import rank_marks

    validation, test = _curves()
    assert rank_marks(validation, test, MARKS) == [
        {"mark": "validation_final", "traces": 2500.0, "rank": 107.0},
        {"mark": "test_at_validation", "traces": 2500.0, "rank": pytest.approx(107.83)},
        {"mark": "test_final", "traces": 22500.0, "rank": 36.0}]
    assert rank_marks(validation, None, MARKS) == [{"mark": "validation_final", "traces": 2500.0, "rank": 107.0}]
    short = test[:1000]                                           # a test shorter than the validation: no like-for-like
    assert [m["mark"] for m in rank_marks(validation, short, MARKS)] == ["validation_final", "test_final"]


def test_marks_are_validated_against_the_job():
    from services.graph_manager.graph_job_validator import GraphJobValidationError, validate_graph_job

    assert validate_graph_job(_job(marks=MARKS))["marks"] == MARKS
    assert validate_graph_job(_job(marks="test_final"))["marks"] == ["test_final"]
    assert "marks" not in validate_graph_job(_job(marks=[]))
    for job, message in [
        (_job(marks=["validation_end"]), r"marks \['validation_end'\]: use any of"),
        (_job(type="model_overlay", marks=MARKS), "use type: process_overlay"),
        (_job(metric="mean_accuracy", marks=MARKS), "use metric: mean_attack_rank_curve"),
        (_job(processes=["validation"], marks=["test_at_validation"]), r"add \['test'\] to processes"),
        (_job(processes=["test"], marks=["validation_final"]), r"add \['validation'\] to processes"),
    ]:
        with pytest.raises(GraphJobValidationError, match=message):
            validate_graph_job(job)


def test_the_figure_draws_each_mark_with_its_rank_and_trace_count_in_the_legend(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from services.graph_manager.plotters import plot_curve_overlay

    validation, test = _curves()
    captured = {}
    real_close = plt.close
    plt.close = lambda fig=None: captured.setdefault("fig", fig)
    try:
        plot_curve_overlay(series={"validation": validation, "test": test},
                           job={"legend": {"enabled": True}, "export": {"formats": ["png"]}}, out_base=tmp_path / "o",
                           metric="mean_attack_rank_curve", x_label="traces", y_label="rank",
                           order=["validation", "test"], marks=MARKS)
    finally:
        plt.close = real_close
    ax = captured["fig"].axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert "validation final: 107 @ 2,500" in labels
    assert "test at the same traces: 107.8 @ 2,500" in labels
    assert "test final: 36 @ 22,500" in labels
    assert any(list(line.get_xdata()) == [2500, 2500] and line.get_linestyle() == ":" for line in ax.get_lines())
    assert (tmp_path / "o.png").exists()


def test_a_process_overlay_job_with_marks_draws_them_from_the_runs_stored_curves(tmp_path, monkeypatch):
    """GraphManager end to end: the run's stored validation and test curves reach the plotter with the job's marks."""
    import services.graph_manager.graph_manager as gm
    from test_final_graph import _manager, _project

    _project(tmp_path)
    validation, test = _curves()
    arrays = tmp_path / "artifacts" / "dev" / "generated_models" / "mesh_chicken" / "metrics" / "attack_metrics" / "arrays"
    for process, epoch, curve in (("validation", 20, validation), ("test", 1, test)):
        folder = arrays / process / "mean_attack_rank_curve"
        folder.mkdir(parents=True)
        np.save(folder / f"ep{epoch:04d}.npy", curve)
    manager = _manager(tmp_path, monkeypatch, [_job(models=["mesh_chicken"], marks=MARKS)])
    seen = {}
    real = gm.plot_curve_overlay

    def spy(**kwargs):
        seen.update(series=sorted(kwargs["series"]), marks=kwargs["marks"])
        return real(**kwargs)

    monkeypatch.setattr(gm, "plot_curve_overlay", spy)
    manager.execute(9)
    assert seen == {"series": ["test", "validation"], "marks": MARKS}
    run_graphs = tmp_path / "artifacts" / "dev" / "generated_models" / "mesh_chicken" / "graphs"   # a run's own figure
    assert list(run_graphs.rglob("*__mean_attack_rank_curve.svg"))
