# services/side_channel_core/tests/unit/test_graph_groups.py
"""Graph job groups (2026-10-24): runs pooled into one series of a model_overlay curve, as Wouters et al. (TCHES
2020(3), Figs. 1-2) draw ten trainings of one configuration: each run's curve thin, their mean bold, and the band
the job names (min_max | std | none: his paper never says which, so there is no default)."""

import numpy as np
import pytest


def _job(**extra):
    job = {"id": 7, "name": "w", "type": "model_overlay", "enabled": True, "metric": "mean_attack_rank_curve",
           "source": "attack_metrics", "models": ["z0", "z1", "z2", "s0", "s1", "s2", "other"], "processes": ["test"],
           "output_file_name": "w"}
    job.update(extra)
    return job


GROUPS = [{"label": "Zaid N0", "models": ["z0", "z1", "z2"], "band": "min_max"},
          {"label": "Simplified N0", "models": ["s0", "s1", "s2"], "band": "std"}]


def test_groups_are_validated_and_the_band_is_never_assumed():
    from services.graph_manager.graph_job_validator import GraphJobValidationError, validate_graph_job

    assert validate_graph_job(_job(groups=GROUPS))["groups"] == GROUPS
    assert "groups" not in validate_graph_job(_job(groups=[]))
    for bad, message in [
        ([{"label": "a", "models": ["z0", "z1"]}], "band must be one of"),
        ([{"label": "a", "models": ["z0", "z1"], "band": "range"}], "band must be one of"),
        ([{"label": "", "models": ["z0", "z1"], "band": "none"}], "label is required"),
        ([{"label": "a", "models": ["z0", "nope"], "band": "none"}], "'nope' is not one of the job's models"),
        ([{"label": "a", "models": ["z0"], "band": "std"}], "needs two runs at least"),
        ([{"label": "a", "models": ["z0", "z1"], "band": "none"}, {"label": "b", "models": ["z1"], "band": "none"}],
         "already in group 'a'"),
        ([{"label": "a", "models": ["z0"], "band": "none", "colour": "red"}], "unknown key"),
        ([{"label": "a", "models": ["z0", "z1"], "band": "std", "members": "no"}], "members must be true or false"),
        ([{"label": "a", "models": ["z0", "z1"], "band": "std", "color": "purplish"}], "must be a Matplotlib colour"),
    ]:
        with pytest.raises(GraphJobValidationError, match=message):
            validate_graph_job(_job(groups=bad))
    hidden = [{**GROUPS[0], "members": False}]
    assert validate_graph_job(_job(groups=hidden))["groups"] == hidden
    with pytest.raises(GraphJobValidationError, match="use type: model_overlay"):
        validate_graph_job(_job(type="solo", groups=GROUPS))


def _draw(tmp_path, series, groups, dropped=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from services.graph_manager.plotters import plot_curve_overlay

    captured = {}
    real_close = plt.close
    plt.close = lambda fig=None: captured.setdefault("fig", fig)
    try:
        plot_curve_overlay(series=series, job={"legend": {"enabled": True}, "export": {"formats": ["png"]}},
                           out_base=tmp_path / "o", metric="mean_attack_rank_curve", x_label="traces",
                           y_label="rank", order=list(series) + list(dropped or {}), dropped=dropped, groups=groups)
    finally:
        plt.close = real_close
    fig = captured["fig"]
    ax = fig.axes[0]
    out = (ax, [t.get_text() for t in ax.get_legend().get_texts()])
    real_close(fig)
    return out


def test_a_group_draws_its_runs_thin_their_mean_bold_and_the_named_band(tmp_path):
    x = np.arange(1, 6, dtype=float)
    z = {f"z{i}": np.column_stack([x, 10.0 - x + i]) for i in range(3)}      # offsets 0, 1, 2
    s = {f"s{i}": np.column_stack([x, 5.0 + i * x]) for i in range(3)}
    other = {"other": np.column_stack([x, np.full(5, 7.0)])}
    ax, texts = _draw(tmp_path, {**z, **s, **other}, GROUPS)
    assert texts == ["Zaid N0 (mean of 3 runs; band: min to max)", "Simplified N0 (mean of 3 runs; band: mean ± 1 std)",
                     "other"]
    lines = [line for line in ax.lines if len(line.get_xdata())]
    thin = [line for line in lines if line.get_alpha() == 0.35]
    assert len(thin) == 6 and {line.get_color() for line in thin[:3]} == {"C0"} and {line.get_color() for line in thin[3:]} == {"C1"}
    means = {line.get_label(): line for line in lines if line.get_label().startswith(("Zaid", "Simplified"))}
    np.testing.assert_allclose(means["Zaid N0 (mean of 3 runs; band: min to max)"].get_ydata(), 10.0 - x + 1.0)
    np.testing.assert_allclose(means["Simplified N0 (mean of 3 runs; band: mean ± 1 std)"].get_ydata(), 5.0 + x)
    assert [line.get_color() for line in lines if line.get_label() == "other"] == ["C2"]   # after the groups
    bands = ax.collections
    assert len(bands) == 2
    zaid_band = bands[0].get_paths()[0].vertices[:, 1]
    assert zaid_band.min() == pytest.approx(10.0 - 5 + 0) and zaid_band.max() == pytest.approx(10.0 - 1 + 2)
    simplified_std = np.std(np.vstack([5.0 + i * x for i in range(3)]), axis=0, ddof=1)
    top = bands[1].get_paths()[0].vertices[:, 1].max()
    assert top == pytest.approx((5.0 + x + simplified_std).max())


def test_a_group_names_runs_it_could_not_read_and_refuses_runs_on_other_traces(tmp_path):
    x = np.arange(1, 4, dtype=float)
    series = {"z0": np.column_stack([x, x]), "z1": np.column_stack([x, x + 1])}
    _, texts = _draw(tmp_path, series, [{"label": "Z", "models": ["z0", "z1", "z2"], "band": "none"}],
                     dropped={"z2": "not recorded"})
    assert texts == ["Z (mean of 2 runs, 1 not recorded)"]
    series["z1"] = np.column_stack([x + 1, x])
    with pytest.raises(ValueError, match="a group pools runs attacked on the same test traces"):
        _draw(tmp_path, series, [{"label": "Z", "models": ["z0", "z1"], "band": "min_max"}])


def test_a_group_pools_per_epoch_metrics_one_series_each_and_can_hide_its_runs(tmp_path):
    """The Model Performance Report's loss (2026-10-30): ten trainings' training and validation loss as two means
    with their +-1 std band, in the metrics' colours, no curve per training (members: false)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from services.graph_manager.plotters import plot_scalar_overlay

    epochs = np.arange(1, 6, dtype=float)
    runs = ["r0", "r1", "r2"]
    series = {f"{r}:{m}": pd.DataFrame({"epoch": epochs, "value": base - epochs / 10 + i})
              for i, r in enumerate(runs) for m, base in (("training_objective", 5.0), ("validation_objective", 6.0))}
    colours = {"training_objective": "#666666", "validation_objective": "#00B4D8"}
    job = {"metrics": list(colours), "legend": {"enabled": True}, "export": {"formats": ["png"]},
           "style": {"series": {f"{r}:{m}": {"color": c} for r in runs for m, c in colours.items()}}}
    groups = [{"label": "W", "models": runs, "band": "std", "members": False}]
    captured = {}
    real_close = plt.close
    plt.close = lambda fig=None: captured.setdefault("fig", fig)
    try:
        plot_scalar_overlay(series=series, job=job, out_base=tmp_path / "loss", metric="loss", x_col="epoch",
                            y_col="value", order=list(series), groups=groups)
    finally:
        plt.close = real_close
    ax = captured["fig"].axes[0]
    lines = [line for line in ax.lines if len(line.get_xdata())]
    assert len(lines) == 2 and not [line for line in lines if line.get_alpha() == 0.35]     # no curve per run
    for line, (metric, base) in zip(lines, (("training_objective", 5.0), ("validation_objective", 6.0))):
        assert line.get_label() == f"W: {metric} (mean of 3 runs; band: mean ± 1 std)"
        assert matplotlib.colors.to_hex(line.get_color()) == colours[metric].lower()
        np.testing.assert_allclose(line.get_ydata(), base - epochs / 10 + 1.0)
    assert len(ax.collections) == 2                                                         # the two bands
    real_close(captured["fig"])


def test_a_std_band_stays_within_its_runs_range(tmp_path):
    """Two trainings at rank 1, one at 100 (VI.b, 2026-10-30): mean - std is below rank 1, the band stops at 1."""
    x = np.arange(1, 6, dtype=float)
    series = {f"r{i}": np.column_stack([x, np.full(5, rank)]) for i, rank in enumerate((1.0, 1.0, 100.0))}
    ax, _ = _draw(tmp_path, series, [{"label": "W", "models": list(series), "band": "std", "members": False}])
    band = ax.collections[0].get_paths()[0].vertices[:, 1]
    spread = np.std([1.0, 1.0, 100.0], ddof=1)
    assert band.min() == pytest.approx(1.0) and band.max() == pytest.approx(34.0 + spread)
    assert not [line for line in ax.lines if line.get_alpha() == 0.35]                     # no curve per run


def test_a_group_takes_its_own_colour_and_band_colour(tmp_path):
    """Fig. 8's GE (2026-10-30): the mean #4A148C, its band #7E57C2, not the colour cycle's slot."""
    import matplotlib

    x = np.arange(1, 6, dtype=float)
    series = {f"r{i}": np.column_stack([x, 10.0 - x + i]) for i in range(3)}
    group = {"label": "W", "models": list(series), "band": "std", "members": False, "color": "#4A148C",
             "band_color": "#7E57C2"}
    ax, _ = _draw(tmp_path, series, [group])
    mean = [line for line in ax.lines if line.get_label().startswith("W (")][0]
    assert matplotlib.colors.to_hex(mean.get_color()) == "#4a148c"
    assert matplotlib.colors.to_hex(ax.collections[0].get_facecolor()[0]) == "#7e57c2"
