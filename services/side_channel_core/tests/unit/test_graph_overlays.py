# services/side_channel_core/tests/unit/test_graph_overlays.py
"""Graph jobs' overlays and plot_type strip (TODO.md section 29): the Model Performance Report's GE graph (the
geometric mean and the median over the attacks beside the GE, each training's N̄tGE at GE 1 with their median, the
N_val and N_budget lines) and its key-confidence strip (a dot per run at log10(1 + the best key's score margin),
teal when that key is the true key), drawn by graph jobs."""

import json

import numpy as np
import pytest
import yaml


def _ge_job(**extra):
    job = {"id": 5, "name": "ge", "type": "model_overlay", "enabled": True, "metric": "mean_attack_rank_curve",
           "source": "attack_metrics", "models": ["mesh_chicken", "ultimate_jewel"], "processes": ["test"],
           "output_file_name": "ge", "overlays": {"ge_averages": ["geometric", "median"], "ntge": True,
                                                  "trace_lines": ["n_val", "budget"]}}
    job.update(extra)
    return job


def _strip_job(**extra):
    job = {"id": 6, "name": "kc", "type": "model_overlay", "enabled": True, "metric": "top1_top2_score_margin",
           "source": "key_recovery", "models": ["mesh_chicken", "ultimate_jewel"], "processes": ["test"],
           "output_file_name": "kc", "plot_type": "strip"}
    job.update(extra)
    return job


GROUP = [{"label": "W, N100", "models": ["mesh_chicken", "ultimate_jewel"], "band": "std"}]


# ------------------------------------------------------------------ validation of the job
def test_overlays_are_normalised_and_empty_means_none():
    from services.graph_manager.graph_job_validator import validate_graph_job

    job = validate_graph_job(_ge_job(overlays={"ge_averages": "Median", "ntge": True, "trace_lines": ["budget", "budget"]}))
    assert job["overlays"] == {"ge_averages": ["median"], "ntge": True, "trace_lines": ["budget"]}
    for empty in (None, {}, {"ge_averages": [], "ntge": False}):
        assert "overlays" not in validate_graph_job(_ge_job(overlays=empty))
    assert validate_graph_job(_ge_job(processes=["validation", "test"], test="budget"))["overlays"]["ntge"] is True
    paper = {"geometric": "#08306B", "median": "#FF00FF", "ntge": "#222222"}       # Fig. 8's colours (2026-10-30)
    assert validate_graph_job(_ge_job(overlays={"ntge": True, "colors": paper}))["overlays"]["colors"] == paper


@pytest.mark.parametrize("extra, message", [
    ({"overlays": ["ntge"]}, "must be a mapping"),
    ({"overlays": {"ntge": True, "marks": True}}, r"unknown key\(s\) \['marks'\]"),
    ({"overlays": {"ntge": True, "colors": {"mean": "red"}}}, "colors maps any of"),
    ({"overlays": {"ntge": True, "colors": {"median": "fuchsiaish"}}}, "not a Matplotlib colour"),
    ({"overlays": {"ge_averages": ["harmonic"]}}, r"ge_averages \['harmonic'\]: use any of"),
    ({"overlays": {"ge_averages": [1]}}, "ge_averages must be a list"),
    ({"overlays": {"trace_lines": ["n_test"]}}, r"trace_lines \['n_test'\]: use any of"),
    ({"overlays": {"ntge": "yes"}}, "ntge is true or false"),
    ({"overlays": {"ntge": True}, "trace_budget": 100}, "drop trace_budget"),
    ({"type": "process_overlay", "models": ["mesh_chicken"]}, "use type: model_overlay"),
    ({"metric": "sr_curve"}, "use metric: mean_attack_rank_curve"),
    ({"processes": ["train", "test"]}, "processes validation and/or test"),
    ({"array_epoch": {"mode": "overlay", "epochs": ["ep0001"]}}, "array_epoch.mode latest or selected"),
])
def test_overlays_a_job_cannot_draw_are_refused(extra, message):
    from services.graph_manager.graph_job_validator import GraphJobValidationError, validate_graph_job

    with pytest.raises(GraphJobValidationError, match=message):
        validate_graph_job(_ge_job(**extra))


def test_the_config_schema_refuses_an_unread_overlay_key():
    from services.side_channel_core.utils import config_schema as cs
    from services.side_channel_core.utils.config_errors import ConfigError

    cs.check_keys({"jobs": [_ge_job()]}, cs.GRAPH, source="graph_config.yaml")
    with pytest.raises(ConfigError, match="overlays"):
        cs.check_keys({"jobs": [_ge_job(overlays={"ntg": True})]}, cs.GRAPH, source="graph_config.yaml")


def test_a_strip_is_the_tests_key_recovery_margin_and_takes_groups():
    from services.graph_manager.graph_job_validator import GraphJobValidationError, validate_graph_job
    from services.graph_manager.metric_router import plot_types_for

    assert "strip" in plot_types_for("model_overlay", ["top1_top2_score_margin"])
    assert "strip" not in plot_types_for("solo", ["top1_top2_score_margin"])
    assert "strip" not in plot_types_for("model_overlay", ["mean_attack_rank_curve"])
    assert validate_graph_job(_strip_job(groups=GROUP))["groups"] == GROUP
    for extra, message in [
        ({"type": "process_overlay", "models": ["mesh_chicken"]}, "is a model_overlay"),
        ({"source": "attack_metrics"}, "source key_recovery"),
        ({"metric": "final_true_key_rank"}, "metric top1_top2_score_margin"),
        ({"processes": ["validation"]}, r"processes \[test\]"),
    ]:
        with pytest.raises(GraphJobValidationError, match=message):
            validate_graph_job(_strip_job(**extra))


# ------------------------------------------------------------------ the averages over the attacks
def test_the_geometric_mean_and_the_median_over_the_attacks():
    from services.graph_manager.plotters import ge_average_curves

    x = np.array([1.0, 2.0, 3.0])
    ranks = np.array([[0.5, 4.0, 1.0], [8.0, 4.0, 1.0], [2.0, 16.0, 9.0]])      # rank 0.5: counted as 1
    out = ge_average_curves(np.stack([np.column_stack([x, r]) for r in ranks]))
    np.testing.assert_allclose(out["geometric"], np.column_stack([x, [16 ** (1 / 3), 256 ** (1 / 3), 9 ** (1 / 3)]]))
    np.testing.assert_allclose(out["median"], np.column_stack([x, [2.0, 4.0, 1.0]]))
    with pytest.raises(ValueError, match=r"\[attacks, traces, 2\]"):
        ge_average_curves(np.zeros((3, 2)))


# ------------------------------------------------------------------ drawing
def _capture(monkeypatch):
    """The figure a plotter saves, kept open for the test."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    return captured


def _legend(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def _dashed(lines, dashes):
    """The curves drawn with this dash pattern (plotters.GE_AVERAGE_STYLES), not the vertical trace lines."""
    return [line for line in lines if line._unscaled_dash_pattern[1] == dashes and len(line.get_xdata()) > 2]


def _marks(lines, marker):
    """(colour, x, y) of each mark drawn alone (no line through it)."""
    return [(line.get_color(), line.get_xdata()[0], line.get_ydata()[0]) for line in lines
            if line.get_marker() == marker and line.get_linestyle() == "None"]


GEOMETRIC, MEDIAN = (3, 1.5), (4, 1.2, 1, 1.2)


def test_the_overlays_draw_in_the_runs_colours_with_one_legend_entry_each(tmp_path, monkeypatch):
    from services.graph_manager.plotters import NTGE_MEDIAN_COLOUR, plot_curve_overlay

    captured = _capture(monkeypatch)
    x = np.arange(1.0, 6.0)
    series = {"a": np.column_stack([x, 6 - x]), "b": np.column_stack([x, 7 - x]), "c": np.column_stack([x, 8 - x])}
    overlays = {"ge_averages": {"geometric": {"a": np.column_stack([x, 5 - x]), "c": np.column_stack([x, 4 - x])},
                                "median": {"a": np.column_stack([x, 3 - x])}},
                "ntge": {"a": 2.0, "b": float("nan"), "c": 4.0},
                "trace_lines": {"n_val": [3.0, 3.0], "budget": [4.0]}}
    plot_curve_overlay(series=series, job={"legend": {"enabled": True}, "export": {"formats": ["png"]}},
                       out_base=tmp_path / "o", metric="mean_attack_rank_curve", x_label="traces", y_label="GE",
                       order=list(series), overlays=overlays)
    ax = captured["fig"].axes[0]
    assert _legend(ax) == ["a", "b", "c", "geometric mean over the attacks", "median over the attacks",
                           "each training's $\\bar{N}_{tGE}$", "their median", "never stably at rank 1",
                           "$N_\\mathrm{val}$: validation traces", "$N_\\mathrm{budget}$: traces per Budget-test attack"]
    drawn = [line for line in ax.lines if len(line.get_xdata())]
    assert [(line.get_color(), list(line.get_ydata())) for line in _dashed(drawn, GEOMETRIC)] == [
        ("C0", list(5 - x)), ("C2", list(4 - x))]                                   # a run without one: none
    assert [(line.get_color(), list(line.get_ydata())) for line in _dashed(drawn, MEDIAN)] == [("C0", list(3 - x))]
    assert _marks(drawn, "o") == [("C0", 2.0, 1.0), ("C2", 4.0, 1.0)]
    assert _marks(drawn, ">") == [("C1", 1.0, 1.0)]                                   # x: the axis' end (axes units)
    assert _marks(drawn, "D") == [(NTGE_MEDIAN_COLOUR, 3.0, 1.0)]                     # the median of 2 and 4
    verticals = sorted((line.get_color(), line.get_xdata()[0]) for line in drawn
                       if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1])
    assert verticals == [("#9a4a00", 3.0), ("#c00000", 4.0)]                        # one line per distinct count


def test_with_groups_the_overlays_are_the_groups(tmp_path, monkeypatch):
    from services.graph_manager.plotters import NTGE_MEDIAN_COLOUR, plot_curve_overlay

    captured = _capture(monkeypatch)
    x = np.arange(1.0, 4.0)
    series = {k: np.column_stack([x, x + i]) for i, k in enumerate(("a", "b", "c"))}
    overlays = {"ge_averages": {"geometric": {"a": np.column_stack([x, x]), "b": np.column_stack([x, x + 2])}},
                "ntge": {"a": 1.0, "b": 5.0, "c": 7.0}}
    plot_curve_overlay(series=series, job={"legend": {"enabled": True}, "export": {"formats": ["png"]}},
                       out_base=tmp_path / "o", metric="mean_attack_rank_curve", x_label="traces", y_label="GE",
                       order=list(series), groups=[{"label": "G", "models": ["a", "b"], "band": "none"}],
                       overlays=overlays)
    drawn = [line for line in captured["fig"].axes[0].lines if len(line.get_xdata())]
    assert [(line.get_color(), list(line.get_ydata())) for line in _dashed(drawn, GEOMETRIC)] == [
        ("C0", list(x + 1))]                                                        # their mean
    assert [m[:2] for m in _marks(drawn, "o")] == [("C0", 1.0), ("C0", 5.0), ("C1", 7.0)]   # the group's colour, c's own
    assert [m[:2] for m in _marks(drawn, "D")] == [(NTGE_MEDIAN_COLOUR, 3.0), (NTGE_MEDIAN_COLOUR, 7.0)]  # group, rest


def test_the_overlays_take_their_own_colours_and_the_groups(tmp_path, monkeypatch):
    """Fig. 8's GE (2026-10-30): the group #4A148C, the geometric mean #08306B, each N̄tGE #222222."""
    from services.graph_manager.plotters import plot_curve_overlay

    captured = _capture(monkeypatch)
    x = np.arange(1.0, 4.0)
    series = {k: np.column_stack([x, x + i]) for i, k in enumerate(("a", "b"))}
    overlays = {"ge_averages": {"geometric": {"a": np.column_stack([x, x]), "b": np.column_stack([x, x + 2])}},
                "ntge": {"a": 1.0, "b": 5.0}}
    job = {"legend": {"enabled": True}, "export": {"formats": ["png"]},
           "overlays": {"colors": {"geometric": "#08306B", "ntge": "#222222"}}}
    plot_curve_overlay(series=series, job=job, out_base=tmp_path / "o", metric="mean_attack_rank_curve",
                       x_label="traces", y_label="GE", order=list(series), overlays=overlays,
                       groups=[{"label": "G", "models": ["a", "b"], "band": "none", "members": False, "color": "#4A148C"}])
    drawn = [line for line in captured["fig"].axes[0].lines if len(line.get_xdata())]
    assert [line.get_color() for line in drawn if line.get_label().startswith("G (")] == ["#4A148C"]
    assert [line.get_color() for line in _dashed(drawn, GEOMETRIC)] == ["#08306B"]
    assert {m[0] for m in _marks(drawn, "o")} == {"#222222"}


def test_a_strip_draws_a_row_per_group_coloured_by_the_best_key(tmp_path, monkeypatch):
    import pandas as pd

    from services.graph_manager.plotters import STRIP_DOTS, plot_strip

    captured = _capture(monkeypatch)
    frame = lambda margin, best: pd.DataFrame({"epoch": [1], "top1_top2_score_margin": [margin], "best_is_key": [best]})  # noqa: E731
    series = {"a": frame(9.0, 1), "b": frame(99.0, 0), "c": frame(0.0, None),
              "d": pd.DataFrame({"epoch": [1], "top1_top2_score_margin": [float("nan")]})}
    out = plot_strip(series=series, job={"legend": {"enabled": True}, "export": {"formats": ["png"]}},
                     out_base=tmp_path / "s", metric="top1_top2_score_margin",
                     groups=[{"label": "G", "models": ["a", "b"], "band": "none"}], order=list(series))
    assert out
    ax = captured["fig"].axes[0]
    dots = [(line.get_color(), line.get_xdata()[0], line.get_ydata()[0]) for line in ax.lines if len(line.get_xdata())]
    assert dots == [(STRIP_DOTS[True][0], 1.0, 0), (STRIP_DOTS[False][0], 2.0, 0), (STRIP_DOTS[None][0], 0.0, 1)]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["G", "other runs"]
    assert _legend(ax) == ["the best key is the true key", "it is not", "the record names no best key"]
    assert ax.get_xlabel() == "log(1 + score margin)"
    assert STRIP_DOTS[True][0] == "#00897B" and STRIP_DOTS[False][0] == "#E53935"


# ------------------------------------------------------------------ the key-recovery loader
def test_the_margin_carries_best_is_key_and_other_metrics_are_unchanged(tmp_path):
    from services.graph_manager.metric_loader import _load_key_recovery_dir

    (tmp_path / "test.json").write_text(json.dumps({
        "epoch": None, "key": {"top_key_hypothesis": 7, "true_key_byte": 224, "final_true_key_rank": 3},
        "diagnostics": {"top1_top2_score_margin": 12.5}}))
    margin, _ = _load_key_recovery_dir(tmp_path, "test", "top1_top2_score_margin")
    assert margin.to_dict("records") == [{"epoch": 1, "top1_top2_score_margin": 12.5, "best_is_key": 0}]
    rank, _ = _load_key_recovery_dir(tmp_path, "test", "final_true_key_rank")
    assert list(rank.columns) == ["epoch", "final_true_key_rank"]
    (tmp_path / "test.json").write_text(json.dumps({"epoch": None, "diagnostics": {"top1_top2_score_margin": 1.0}}))
    assert list(_load_key_recovery_dir(tmp_path, "test", "top1_top2_score_margin")[0].columns) == [
        "epoch", "top1_top2_score_margin"]                                          # a record naming no best key


# ------------------------------------------------------------------ GraphManager end to end
def _run(tmp_path, name):
    return tmp_path / "artifacts" / "dev" / "generated_models" / name


def _store(metrics_dir, process, metric, array, epoch=1):
    folder = metrics_dir / "arrays" / process / metric
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"ep{epoch:04d}.npy", array)


def _per_attack(seed, n=40, attacks=6):
    rng = np.random.default_rng(seed)
    x = np.arange(1, n + 1, dtype=float)
    return np.stack([np.column_stack([x, rng.integers(1, 257, n)]) for _ in range(attacks)]).astype(float)


def _write_run(tmp_path, name, seed, *, ntge, validation_range=(30, 50), budget=25, per_attack=True):
    root = _run(tmp_path, name)
    metrics = root / "metrics" / "attack_metrics"
    curves = _per_attack(seed)
    _store(metrics, "test", "mean_attack_rank_curve", np.column_stack([curves[0, :, 0], curves[:, :, 1].mean(0)]))
    if per_attack:
        _store(metrics, "test", "per_attack_curves", curves)
    (metrics / "scalars" / f"{name}_metrics_test_history.jsonl").write_text(
        json.dumps({"epoch": 1, "per_attack_traces_to_ge_1": ntge}) + "\n")
    data = {"name": "w", "attack_trace_budget": budget}
    if validation_range:
        data["validation_range"] = list(validation_range)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / f"{name}_params.yaml").write_text(yaml.safe_dump({"data": data}))
    return curves


def _spy(monkeypatch, name):
    import services.graph_manager.graph_manager as gm

    seen = {}
    real = getattr(gm, name)

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(gm, name, spy)
    return seen


def test_a_job_reads_its_overlays_from_the_curves_record(tmp_path, monkeypatch, capsys):
    from test_final_graph import _manager, _project

    from services.graph_manager.plotters import ge_average_curves

    _project(tmp_path)
    curves = _write_run(tmp_path, "mesh_chicken", 0, ntge=12.5)
    _write_run(tmp_path, "ultimate_jewel", 1, ntge=None, validation_range=None, budget=None, per_attack=False)
    _store(_run(tmp_path, "ultimate_jewel") / "metrics" / "attack_metrics", "validation", "mean_attack_rank_curve",
           np.column_stack([np.arange(1.0, 801.0), np.ones(800)]), epoch=3)          # N_val from the curve: 800
    seen = _spy(monkeypatch, "plot_curve_overlay")
    _manager(tmp_path, monkeypatch, [_ge_job(groups=GROUP)]).execute(5)
    overlays = seen["overlays"]
    np.testing.assert_array_equal(overlays["ge_averages"]["geometric"]["mesh_chicken"], ge_average_curves(curves)["geometric"])
    assert set(overlays["ge_averages"]["median"]) == {"mesh_chicken"}                  # the other: no per_attack_curves
    assert overlays["ntge"]["mesh_chicken"] == 12.5 and np.isnan(overlays["ntge"]["ultimate_jewel"])
    assert overlays["trace_lines"] == {"n_val": [20.0, 800.0], "budget": [25.0]}
    assert "overlays.ge_averages: model=ultimate_jewel process=test has no per_attack_curves: left out of it" in capsys.readouterr().out
    assert list((tmp_path / "graphs" / "jobs").rglob("*.svg"))


def test_a_job_without_overlays_passes_none(tmp_path, monkeypatch):
    from test_final_graph import _manager, _project

    _project(tmp_path)
    for i, name in enumerate(("mesh_chicken", "ultimate_jewel")):
        _write_run(tmp_path, name, i, ntge=3.0)
    seen = _spy(monkeypatch, "plot_curve_overlay")
    _manager(tmp_path, monkeypatch, [_ge_job(overlays=None)]).execute(5)
    assert seen["overlays"] is None


def test_a_strip_job_draws_the_tests_key_recovery_by_group(tmp_path, monkeypatch):
    from test_final_graph import _manager, _project

    _project(tmp_path)
    for name, best, margin in (("mesh_chicken", 224, 40.0), ("ultimate_jewel", 3, 2.0)):
        kr = _run(tmp_path, name) / "metrics" / "key_recovery"
        kr.mkdir(parents=True)
        (kr / "test.json").write_text(json.dumps({"epoch": None, "key": {"top_key_hypothesis": best, "true_key_byte": 224},
                                                  "diagnostics": {"top1_top2_score_margin": margin}}))
    seen = _spy(monkeypatch, "plot_strip")
    _manager(tmp_path, monkeypatch, [_strip_job(groups=GROUP)]).execute(6)
    assert {k: v["best_is_key"].iloc[-1] for k, v in seen["series"].items()} == {"mesh_chicken": 1, "ultimate_jewel": 0}
    assert seen["groups"] == GROUP
    assert list((tmp_path / "graphs" / "jobs").rglob("*top1_top2_score_margin*.svg"))
    with pytest.raises(ValueError, match="holds one value per run"):                # groups: curves, or a strip
        _manager(tmp_path, monkeypatch, [_strip_job(groups=GROUP, plot_type="auto", id=7)]).execute(7)
