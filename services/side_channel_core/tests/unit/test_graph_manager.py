"""GraphManager: model lists in job 0, sources, fold aggregation."""

import numpy as np
import pandas as pd
import pytest
import yaml

from services.graph_manager.graph_job_validator import (
    GraphJobValidationError,
    get_graph_job_by_id,
)
from services.graph_manager.metric_loader import combine_folds
from services.side_channel_core.utils.graph_config_sync import (
    append_model_name,
    update_default_job_models,
)


CONFIG = """\
# header comment
defaults:
  export: &export {formats: ["png"]}

jobs:
  # job 0 comment
  - id: 0
    name: "default"
    type: "model_overlay"
    enabled: true
    metric: "mean_attack_rank_curve"
    top_models: []
    processes: ["test"]
    output_file_name: "default"
    export: *export

  # job 1 comment
  - id: 1
    name: "cv"
    type: "model_overlay"
    enabled: true
    source: "cross_validation"
    metric: "guessing_entropy"
    models: ["extra"]
    processes: ["validation"]
    output_file_name: "cv"
"""


def test_judge_rewrites_top_models_keeping_rest_of_file(tmp_path):
    path = tmp_path / "graph_config.yaml"
    path.write_text(CONFIG)
    assert update_default_job_models(path, "top_models", ["b", "a"], replace=True) == ["b", "a"]
    assert update_default_job_models(path, "top_models", ["a"], replace=True) == ["a"]
    text = path.read_text()
    assert "# header comment" in text and "# job 1 comment" in text and "&export" in text
    jobs = yaml.safe_load(text)["jobs"]
    assert jobs[0]["top_models"] == ["a"]
    assert jobs[1]["models"] == ["extra"]        # other jobs are left to the user


def test_job_0_plots_top_models_others_their_own(tmp_path):
    jobs = yaml.safe_load(CONFIG)["jobs"]
    jobs[0]["top_models"] = ["b"]
    assert get_graph_job_by_id(jobs, 0)["models"] == ["b"]
    assert get_graph_job_by_id(jobs, 1)["models"] == ["extra"]
    jobs[0]["top_models"] = []
    get_graph_job_by_id(jobs, 1)                                     # job 0 not requested
    with pytest.raises(GraphJobValidationError, match="make judge"):
        get_graph_job_by_id(jobs, 0)


def test_generated_model_names_file(tmp_path):
    assert append_model_name(tmp_path, "one") == ["one"]
    assert append_model_name(tmp_path, "two") == ["one", "two"]
    assert append_model_name(tmp_path, "two") == ["one", "two"]
    text = (tmp_path / "artifacts" / "done_model_names.yaml").read_text()
    assert text.splitlines()[0].startswith("# tfseal1:")          # its seal (utils/seal.py)
    assert text.splitlines()[2:] == ["- one", "- two"]
    assert yaml.safe_load(text) == ["one", "two"]


def test_source_restricts_processes_and_solo_metrics():
    jobs = yaml.safe_load(CONFIG)["jobs"]
    jobs[0]["top_models"] = ["a"]
    jobs[1]["processes"] = ["test"]
    with pytest.raises(GraphJobValidationError, match="source='cross_validation'"):
        get_graph_job_by_id(jobs, 1)
    jobs[1]["processes"], jobs[1]["metric"] = ["validation"], "fold_rank_curves"
    with pytest.raises(GraphJobValidationError, match="type: solo"):
        get_graph_job_by_id(jobs, 1)


def test_combine_folds_mean_std_and_strict_missing():
    parts = {
        "fold_01": pd.DataFrame({"epoch": [1, 2], "ge": [10.0, 4.0]}),
        "fold_02": pd.DataFrame({"epoch": [1, 2], "ge": [20.0, np.nan]}),
    }
    out = combine_folds("ge", parts)
    assert out["ge"].tolist()[0] == 15.0 and np.isnan(out["ge"].tolist()[1])  # a fold without a value
    assert out["n_folds"].tolist() == [2, 2]

    x = np.arange(1, 4, dtype=float)
    curves = {"f1": np.column_stack([x, [3.0, 2.0, 1.0]]), "f2": np.column_stack([x, [5.0, 4.0, 1.0]])}
    mean = combine_folds("mean_attack_rank_curve", curves)
    assert mean.shape == (3, 3)
    assert mean[:, 1].tolist() == [4.0, 3.0, 1.0] and mean[2, 2] == 0.0

    pooled = combine_folds("final_ranks", {"f1": np.array([7]), "f2": np.array([1])})
    assert pooled.tolist() == [7.0, 1.0]

    with pytest.raises(ValueError, match="trace grids"):
        combine_folds("mean_attack_rank_curve", {"f1": curves["f1"], "f2": curves["f2"][:2]})


def test_combine_folds_without_std_loads_the_mean_only():
    parts = {
        "fold_01": pd.DataFrame({"epoch": [1, 2], "ge": [10.0, 4.0]}),
        "fold_02": pd.DataFrame({"epoch": [1, 2], "ge": [20.0, 6.0]}),
    }
    out = combine_folds("ge", parts, with_std=False)
    assert "ge_std" not in out.columns and out["ge"].tolist() == [15.0, 5.0]

    x = np.arange(1, 4, dtype=float)
    curves = {"f1": np.column_stack([x, [3.0, 2.0, 1.0]]), "f2": np.column_stack([x, [5.0, 4.0, 1.0]])}
    mean = combine_folds("mean_attack_rank_curve", curves, with_std=False)
    assert mean.shape == (3, 2) and mean[:, 1].tolist() == [4.0, 3.0, 1.0]

    nested = combine_folds("ge", {"f1": {"ep1": parts["fold_01"]}, "f2": {"ep1": parts["fold_02"]}}, with_std=False)
    assert "ge_std" not in nested["ep1"].columns


def test_style_std_band_switch_and_alpha():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import _draw_band

    assert validate_graph_job(_job())["style"]["std_band"] == {"enabled": True, "alpha": 0.15}
    off = validate_graph_job(_job(style={"std_band": {"enabled": False, "alpha": 0.4}}))
    assert off["style"]["std_band"] == {"enabled": False, "alpha": 0.4}
    with pytest.raises(GraphJobValidationError, match="style.std_band"):
        validate_graph_job(_job(style={"std_band": "off"}))

    x, y, std = np.arange(3.0), np.ones(3), np.full(3, 0.5)
    for style, bands, alpha in (({"std_band": {"alpha": 0.4}}, 1, 0.4), ({"std_band": {"enabled": False}}, 0, None)):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        _draw_band(ax, x, y, std, job={"style": style})
        assert len(ax.collections) == bands
        if bands:
            assert ax.collections[0].get_alpha() == alpha
        plt.close(fig)


def test_barh_latest_keeps_the_last_epoch_and_names_bars_in_the_legend(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from services.graph_manager.graph_manager import GraphManager
    from services.graph_manager.metric_loader import LoadedMetric

    def loaded():
        df = pd.DataFrame({"epoch": [1, 2, 3], "mean_fit_time_s": [5.0, 10.0, 15.0]})
        return LoadedMetric("mean_fit_time_s", "mean_fit_time_s", "m", "train", None, df, None)

    barh = GraphManager._select_scalar_epoch(loaded(), {"plot_type": "barh", "array_epoch": {"mode": "latest"}})
    assert barh.data["epoch"].tolist() == [3] and barh.data["mean_fit_time_s"].tolist() == [15.0]
    line = GraphManager._select_scalar_epoch(loaded(), {"plot_type": "line", "array_epoch": {"mode": "latest"}})
    assert len(line.data) == 3  # a line keeps the whole history
    selected = GraphManager._select_scalar_epoch(
        loaded(), {"plot_type": "barh", "array_epoch": {"mode": "selected", "epoch": 2}}
    )
    assert selected.data["mean_fit_time_s"].tolist() == [10.0]

    from services.graph_manager.plotters import plot_scalar_overlay

    frames = {m: pd.DataFrame({"epoch": [3], "t": [v]}) for m, v in (("m1", 5.0), ("m2", 9.0))}
    labels = {"m1": "batch size=50", "m2": "batch size=500"}
    for enabled in (True, False):
        job = {"plot_type": "barh", "legend": {"enabled": enabled, "position": "bottom-right",
               "mode": "custom", "labels": labels}, "export": {"formats": ["png"]}}
        captured = {}
        real_close = plt.close
        plt.close = lambda fig=None: captured.setdefault("fig", fig)
        try:
            plot_scalar_overlay(series=frames, job=job, out_base=tmp_path / f"bars_{enabled}", metric="t")
        finally:
            plt.close = real_close
        ax = captured["fig"].axes[0]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["", ""]  # never on the axis
        legend = ax.get_legend()
        if enabled:
            assert [t.get_text() for t in legend.get_texts()] == ["batch size=50", "batch size=500"]
            assert legend._loc == 4  # lower right
        else:
            assert legend is None
        real_close(captured["fig"])


def test_legend_handle_shape_and_thickness():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plot_style import parse_legend_handle
    from services.graph_manager.plotters import legend_handles

    assert parse_legend_handle(["line", 2]) == {"shape": "line", "marker": None, "size": 2.0}
    assert parse_legend_handle(["s", 8])["marker"] == "s"
    assert parse_legend_handle(["\u25a0", 9])["marker"] == "$\u25a0$"  # a character is drawn as text
    for bad in (["line"], ["line", 0], ["line", "thick"], ["blob", 2], "line"):
        with pytest.raises(ValueError):
            parse_legend_handle(bad)
    with pytest.raises(GraphJobValidationError, match="legend.handle"):
        validate_graph_job(_job(legend={"handle": ["blob", 2]}))
    assert validate_graph_job(_job())["legend"]["handle"] is None

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="C1", label="curve")
    bars = ax.barh([0], [1.0], color="C2", label="bar")
    drawn = [ax.get_lines()[0], bars]
    line = legend_handles({"legend": {"handle": ["line", 3.5]}}, drawn)
    assert [h.get_linewidth() for h in line] == [3.5, 3.5]
    assert matplotlib.colors.same_color(line[1].get_color(), "C2")
    marker = legend_handles({"legend": {"handle": ["o", 7]}}, drawn)
    assert [(h.get_marker(), h.get_markersize(), h.get_linestyle()) for h in marker] == [("o", 7.0, "None")] * 2
    assert legend_handles({"legend": {}}, drawn) == drawn
    plt.close(fig)


def test_catalog_lists_metrics_per_source_and_process(tmp_path):
    import json

    from services.graph_manager.metric_catalog import catalog

    def history(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    root = tmp_path / "metrics"
    rows = [{"epoch": 1, "accuracy": 0.1, "traces_to_ge_1": None, "accuracy_total": 5, "ge_threshold_status": "x"}]
    history(root / "attack_metrics" / "scalars" / "m_metrics_test_history.jsonl", rows)
    curve = root / "attack_metrics" / "arrays" / "test" / "mean_attack_rank_curve"
    curve.mkdir(parents=True)
    (curve / "ep0001.npy").write_bytes(b"")
    history(root / "cross_validation" / "fold_01" / "scalars" / "m_metrics_validation_history.jsonl", rows)
    fold_curve = root / "cross_validation" / "fold_01" / "arrays" / "validation" / "mean_attack_rank_curve"
    fold_curve.mkdir(parents=True)
    (fold_curve / "ep0001.npy").write_bytes(b"")
    (root / "key_recovery").mkdir()
    (root / "key_recovery" / "test.json").write_text(json.dumps({
        "epoch": None,
        "key": {"final_true_key_rank": 3, "true_key_byte": 224},
        "recovery": {"first_rank1_trace": None, "final_ge": 3.0},
        "diagnostics": {"top_key_candidates": [{"key_guess": 0, "score": -1.0}]},
    }))

    found = catalog(root)
    assert found[("attack_metrics", "test")] == ({"accuracy"}, {"mean_attack_rank_curve"}, {"traces_to_ge_1"})
    # fold_rank_curves is offered wherever the folds recorded rank curves
    assert "fold_rank_curves" in found[("cross_validation", "validation")][1]
    scalars, arrays, empty = found[("key_recovery", "test")]
    assert scalars == {"final_true_key_rank", "final_ge"} and arrays == {"key_candidate_scores"}
    assert empty == {"first_rank1_trace"}        # recorded, never reached



def _facet(arrangement=None, n=4):
    facet = {"id": 1, "panels": [
        {"file_id": k, "address": f"p{k}.pdf", "orientation": "portrait" if k == 2 else "landscape"}
        for k in range(1, n + 1)
    ]}
    if arrangement is not None:
        facet["arrangement"] = arrangement
    return facet



def test_facet_arrangements_follow_the_order_of_file_ids():
    from services.graph_manager.faceting import place_panels, validate_facet

    def cells(arrangement, n=4):
        return {p.file_id: (p.row, p.column) for p in place_panels(validate_facet(_facet(arrangement, n)))}

    assert cells(None) == {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (1, 4)}
    assert cells({"mode": "horizontal", "order": [2, 1]}) == {2: (1, 1), 1: (1, 2)}
    assert cells({"mode": "vertical", "order": [3, 1, 2]}) == {3: (1, 1), 1: (2, 1), 2: (3, 1)}
    assert cells({"mode": "grid", "columns": 2, "order": [4, 3, 2, 1]}) == {
        4: (1, 1), 3: (1, 2), 2: (2, 1), 1: (2, 2),
    }
    # Rows written out, of different lengths.
    assert cells({"mode": "grid", "order": [[1, 2], [3, 4, 5]]}, n=5) == {
        1: (1, 1), 2: (1, 2), 3: (2, 1), 4: (2, 2), 5: (2, 3),
    }


def test_facet_legend_entries_and_default_colours():
    from services.graph_manager.faceting import FacetValidationError, validate_facet

    facet = _facet()
    facet["legend"] = {"mode": "custom", "position": "bottom", "labels": ["batch size=50", "batch size=100"]}
    legend = validate_facet(facet)["legend"]
    assert [(e["label"], e["color"]) for e in legend["entries"]] == [
        ("batch size=50", "C0"), ("batch size=100", "C1"),
    ]
    assert legend["columns"] == 2 and legend["enabled"]
    assert legend["handle"] == {"shape": "line", "marker": None, "size": 1.5}

    facet["legend"] = {"mode": "custom", "position": "right", "handle": ["patch", 2],
                       "entries": [{"label": "a", "color": "red"}]}
    legend = validate_facet(facet)["legend"]
    assert legend["columns"] == 1 and legend["entries"][0] == {"label": "a", "color": "red"}
    assert legend["handle"]["shape"] == "patch"
    facet["legend"]["handle"] = ["o", 4]
    assert validate_facet(facet)["legend"]["handle"] == {"shape": "marker", "marker": "o", "size": 4.0}

    assert validate_facet(_facet())["legend"]["enabled"] is False
    for bad in ({"mode": "custom", "enabled": True}, {"mode": "custom", "position": "middle", "labels": ["a"]},
                {"style": "line", "mode": "custom", "labels": ["a"]},                      # moved to handle
                {"mode": "custom", "entries": [{"label": "a", "style": "patch"}]},
                {"mode": "auto", "labels": ["a"]},                                          # entries need custom
                {"mode": "custom", "handle": ["##", 1], "labels": ["a"]},
                {"mode": "auto"}):                                                       # p1.pdf is no job's
        facet["legend"] = bad
        with pytest.raises(FacetValidationError):
            validate_facet(facet)


def test_facet_legend_auto_takes_the_first_panels_job_and_its_labels():
    from services.graph_manager.faceting import FacetValidationError, validate_facet

    facet = {"id": 1, "panels": [{"file_id": 1, "address": "jobs/job_002_b/x.png"},
                                 {"file_id": 2, "address": "jobs/job_001_a/y.png"}],
             "arrangement": {"mode": "horizontal", "order": [1, 2]}, "legend": {"handle": ["s", 5]}}
    jobs = [{"id": 1, "models": ["m"]},
            {"id": 2, "models": ["adam_run", "sgd_run"], "legend": {"mode": "custom", "labels": {"adam_run": "Adam"}}}]
    legend = validate_facet(facet, jobs)["legend"]
    assert legend["mode"] == "auto" and legend["entries"] == [{"label": "Adam", "color": "C0"}, {"label": "sgd_run", "color": "C1"}]
    assert validate_facet(facet)["legend"]["entries"] == []          # without the jobs: checked, not filled
    with pytest.raises(FacetValidationError, match="no such job"):
        validate_facet(facet, [{"id": 1, "models": ["m"]}])


def test_facet_moved_keys_name_their_new_place():
    from services.graph_manager.faceting import FacetValidationError, validate_facet

    for key, value, where in (("style", {"text_rules": []}, "text_rules"),
                              ("export", {"dpi": 300}, "export.png.dpi")):
        facet = {**_facet(), key: value}
        with pytest.raises(FacetValidationError, match=where.replace(".", r"\.")):
            validate_facet(facet)
    out = validate_facet({**_facet(), "export": {"png": {"dpi": 150}}, "text_rules": [{"target": "title", "text": "T"}]})
    assert out["export"]["png"] == {"dpi": 150} and out["text_rules"][0]["text"] == "T"


def test_facet_panel_labels_sit_above_or_below_their_panel():
    from services.graph_manager.faceting import FacetValidationError, build_layout, place_panels, validate_facet

    facet = _facet(n=2)
    facet["panels"][0].update(label="(a) Adam", label_position="top-left")
    facet["panels"][1].update(label="{file_id}", label_position="bottom-right")
    facet["text_rules"] = [{"target": "panel_label", "text": "({letter})", "italic": True}]
    v = validate_facet(facet)
    panels = place_panels(v)
    for p in panels:
        p.aspect = 4 / 3 if p.orientation == "landscape" else 3 / 4
    layout = build_layout(v, panels)
    texts = {t[2]: t for t in layout.texts}
    a, b = texts["(a) Adam"], texts["2"]
    (x0, y0, w0, h0), (x1, y1, w1, h1) = layout.boxes
    assert a[3]["ha"] == "left" and a[1] > y0 + h0 and a[3]["fontstyle"] == "italic"     # above panel 1
    assert b[3]["ha"] == "right" and b[1] < y1                                             # below panel 2
    facet["panels"][0]["label_position"] = "middle"
    with pytest.raises(FacetValidationError):
        validate_facet(facet)


def test_facet_rejects_malformed_entries():
    from services.graph_manager.faceting import FacetValidationError, validate_facet

    bad_panels = (
        [{"file_id": 1, "address": "/abs/a.pdf"}],
        [{"file_id": 1, "address": "../a.pdf"}],
        [{"file_id": 1}],
        [{"address": "a.pdf"}],
        [{"file_id": 1, "address": "a.pdf", "orientation": "diagonal"}],
        [{"file_id": 1, "address": "a.pdf"}, {"file_id": 1, "address": "b.pdf"}],
        [],
    )
    for panels in bad_panels:
        with pytest.raises(FacetValidationError):
            validate_facet({"id": 1, "panels": panels})

    bad_arrangements = (
        {"mode": "diagonal"},
        {"mode": "horizontal", "order": [1, 9]},
        {"mode": "horizontal", "order": [1, 1]},
        {"mode": "horizontal", "order": []},
        {"mode": "vertical", "columns": 2},
        {"mode": "horizontal", "order": [[1, 2], [3]]},
        {"mode": "grid", "order": [[1, 2], 3]},
        {"mode": "grid", "columns": 2, "order": [[1, 2], [3]]},
    )
    for arrangement in bad_arrangements:
        with pytest.raises(FacetValidationError):
            validate_facet(_facet(arrangement))


def _job(**extra):
    job = {"id": 5, "name": "j", "type": "model_overlay", "enabled": True, "metric": "fit_time_s",
           "source": "cross_validation", "models": ["m"], "processes": ["train", "validation"],
           "output_file_name": "j"}
    job.update(extra)
    return job


def test_job_metric_spelling_and_processes_that_are_sources():
    from services.graph_manager.graph_job_validator import validate_graph_job

    assert validate_graph_job(_job(metric=["fit_time_{s}"]))["metric"] == "fit_time_s"
    assert validate_graph_job(_job(metric="fit_time_{per_epoch}"))["metric"] == "fit_time_per_epoch"
    with pytest.raises(GraphJobValidationError, match="one metric"):
        validate_graph_job(_job(type="solo", metric=["mean_rank", "accuracy"]))
    # a model_overlay may overlay several per-epoch scalars
    several = validate_graph_job(_job(metric=["training_objective", "cross_entropy"]))
    assert several["metrics"] == ["training_objective", "cross_entropy"] and several["metric"] == "training_objective"
    with pytest.raises(GraphJobValidationError, match="that is a source"):
        validate_graph_job(_job(processes=["cross_validation", "train"]))


def test_job_combine_is_an_item_of_processes():
    from services.graph_manager.graph_job_validator import validate_graph_job

    item = {"combine": {"processes": ["train", "validation"], "mode": "sum"}}
    job = validate_graph_job(_job(processes=["train", "validation", item]))
    assert job["processes"] == ["train", "validation", "train+validation"]
    assert job["combines"]["train+validation"]["mode"] == "sum"

    only = validate_graph_job(_job(processes=[item]))
    assert only["processes"] == ["train+validation"]
    assert validate_graph_job(_job())["combines"] == {}

    bad_items = (
        {"combine": {"processes": ["train"]}},
        {"combine": {"processes": ["train", "test"]}},
        {"combine": {"processes": ["train", "validation"], "mode": "product"}},
        {"merge": {"processes": ["train", "validation"]}},
    )
    for bad in bad_items:
        with pytest.raises(GraphJobValidationError):
            validate_graph_job(_job(processes=["train", bad]))
    with pytest.raises(GraphJobValidationError, match="belongs under processes"):
        validate_graph_job(_job(combine={"processes": ["train", "validation"]}))


def test_job_axis_limits():
    from services.graph_manager.graph_job_validator import validate_graph_job

    axes = validate_graph_job(_job(style={"axes": {"x_limits": [0, 600]}}))["style"]["axes"]
    assert axes["x_limits"] == [0.0, 600.0] and axes["y_limits"] is None
    for bad in ([600, 0], [0], "0-600"):
        with pytest.raises(GraphJobValidationError):
            validate_graph_job(_job(style={"axes": {"x_limits": bad}}))


def test_job_axis_steps():
    from services.graph_manager.graph_job_validator import validate_graph_job

    axes = validate_graph_job(_job(style={"axes": {"x_step": 20}}))["style"]["axes"]
    assert axes["x_step"] == 20.0 and axes["y_step"] is None
    for bad in (0, -5, "twenty"):
        with pytest.raises(GraphJobValidationError):
            validate_graph_job(_job(style={"axes": {"y_step": bad}}))
    with pytest.raises(GraphJobValidationError, match="linear axis"):
        validate_graph_job(_job(style={"axes": {"y_scale": "log", "y_step": 100}}))


def test_mean_accuracy_is_the_mean_of_accuracy_over_the_folds():
    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.side_channel_core.utils.cross_validation import (
        fold_mean_accuracy,
        mean_accuracy_requested,
    )

    folds = [
        {"trajectory": {"epochs": [1, 2], "training_accuracy": [0.10, 0.20], "validation_accuracy": [0.004, 0.006]}},
        {"trajectory": {"epochs": [1, 2], "training_accuracy": [0.30, 0.40], "validation_accuracy": [0.006, 0.010]}},
    ]
    means = fold_mean_accuracy(folds)
    last = means["validation"][-1]
    assert last["epoch"] == 2 and last["n_folds"] == 2
    assert last["mean_accuracy"] == pytest.approx(0.008)
    assert last["accuracy"] == [0.006, 0.010]
    assert last["mean_accuracy_std"] == pytest.approx(np.std([0.006, 0.010], ddof=1))
    assert means["train"][0]["mean_accuracy"] == pytest.approx(0.20)

    folds[1]["trajectory"]["training_accuracy"] = [None, None]
    assert set(fold_mean_accuracy(folds)) == {"validation"}

    assert mean_accuracy_requested({"metrics_to_log": ["accuracy", "mean_accuracy"]})
    assert not mean_accuracy_requested({"metrics_to_log": ["accuracy", {"traces_to_ge_k": {"k": 1}}]})

    for source, processes in (("cross_validation", ["validation"]), ("attack_metrics", ["train", "test"])):
        job = validate_graph_job(_job(metric="mean_accuracy", source=source, processes=processes))
        assert job["metric"] == "mean_accuracy"


def test_mean_fit_time_is_the_mean_over_the_folds(tmp_path):
    import json
    from types import SimpleNamespace

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.side_channel_core.utils.cross_validation import (
        fold_mean_fit_time,
        merge_fold_means,
        requested_fit_time_columns,
    )

    for fold, times in ((1, [10.0, 20.0]), (2, [14.0, 30.0])):
        scalars = tmp_path / f"fold_{fold:02d}" / "scalars"
        scalars.mkdir(parents=True)
        (scalars / "m_metrics_train_history.jsonl").write_text(
            "".join(json.dumps({"epoch": e, "fit_time_s": t}) + "\n" for e, t in enumerate(times, 1))
        )
    means = fold_mean_fit_time(
        SimpleNamespace(fold_metrics=tmp_path), model_name="m", n_folds=2, columns=["fit_time_s"]
    )
    assert set(means) == {"train"}
    assert [row["mean_fit_time_s"] for row in means["train"]] == [12.0, 25.0]
    assert [row["fit_time_s"] for row in means["train"]] == [[10.0, 14.0], [20.0, 30.0]]
    assert means["train"][1]["mean_fit_time_s_std"] == pytest.approx(np.std([20.0, 30.0], ddof=1))

    merged = merge_fold_means(means, {"train": [{"epoch": 2, "process": "train", "mean_accuracy": 0.5}]})
    assert merged["train"][1]["mean_accuracy"] == 0.5 and merged["train"][1]["mean_fit_time_s"] == 25.0

    # The mean is written only when it is asked for.
    assert requested_fit_time_columns({"metrics_to_log": ["mean_fit_time_{s}", "mean_attack_rank_curve"]}) == ["fit_time_s"]
    assert requested_fit_time_columns({"metrics_to_log": ["fit_time_{s}", "accuracy"]}) == []
    assert validate_graph_job(_job(metric="mean_fit_time_{s}"))["metric"] == "mean_fit_time_s"


def test_a_run_records_and_shows_what_metrics_to_log_asks_for():
    from services.side_channel_core.src.sca.metrics import fit_time_values, requested_fit_time_metrics
    from services.side_channel_core.utils.logger import format_epoch_table_lines

    assert requested_fit_time_metrics({"metrics_to_log": ["fit_time_{s}"]}) == ["fit_time_s"]
    columns = requested_fit_time_metrics({"metrics_to_log": ["mean_fit_time_{s}"]})
    assert columns == ["fit_time_s", "mean_fit_time_s"]
    assert fit_time_values(columns, seconds=7200.0, epoch=4) == {"fit_time_s": 7200.0, "mean_fit_time_s": 7200.0}

    results = {"accuracy": 0.5, "mean_accuracy": 0.5, "fit_time_s": 3.0, "mean_fit_time_s": 3.0, "guessing_entropy": 1.0}
    def shown(metrics):
        lines = format_epoch_table_lines(columns=[("train", "train", results, None)], elapsed=1.0, eta=0.0, epoch=1,
                                         epochs=1, cfg={"metrics_to_log": metrics})
        rows = [line.split(" | ")[0].strip() for line in lines[3:] if " | " in line]   # after the header and its rule
        return [name for name in rows if name not in ("time", "ETA")]
    assert shown(["mean_accuracy", "mean_fit_time_{s}"]) == ["mean_accuracy", "mean_fit_time_s"]
    assert shown(["accuracy", "fit_time_{s}"]) == ["accuracy", "fit_time_s"]


def test_rank_ties_prior_and_float64_softmax():
    import torch
    from services.side_channel_core.src.sca.leakage_models import build_label_table
    from services.side_channel_core.src.sca.metrics import _ensure_logits_and_probs
    from services.side_channel_core.src.sca.rank_curve import _fixed_key_attack_from_indices

    def attack(scores, key, model, n=40):
        plain = np.arange(n, dtype=np.uint8)
        return _fixed_key_attack_from_indices(
            scores, plain, np.full(n, key, np.uint8), np.arange(n), step=n, eps=1e-40,
            rank_base=1, leakage_model=model, scores_are_log=True,
        )

    # Ties: every key scores the same, so all 256 keys tie and the true key
    # takes a random place among them, never the best place by rule.
    flat = np.zeros((40, 256), dtype=np.float64)
    evidence = attack(flat, 224, "ID")
    assert evidence["true_key_tie_count"] == 256
    assert evidence["tie_break_policy"] == "uniform_random_among_tied_keys_seeded"
    ranks = [
        _fixed_key_attack_from_indices(
            flat, np.arange(40, dtype=np.uint8), np.full(40, 224, np.uint8), np.arange(40),
            step=40, eps=1e-40, rank_base=1, leakage_model="ID",
            scores_are_log=True, tie_break_seed=seed,
        )["final_true_key_rank"]
        for seed in range(2000)
    ]
    assert abs(np.mean(ranks) - 128.5) < 5          # expected rank of a random guess

    # HW prior: a model that outputs the class prior carries no information and
    # must not favour keys that map to the central (common) HW classes.
    table = build_label_table("HW")
    prior = np.bincount(table[0], minlength=9) / 256.0
    uninformative = np.log(np.tile(prior, (40, 1)))
    evidence = attack(uninformative, 224, "HW")
    assert evidence["true_key_tie_count"] == 256            # all tied after the prior
    assert evidence["score_rule"] == "log_likelihood_minus_log_class_prior"

    # Softmax in float64: a probability below the float32 range keeps its true log value.
    cfg = {"metrics_cfg": {"preds_are_probs": False}, "trainer": {"evaluation": {"output_transformation": {"type": "Softmax"}}}}
    logits = torch.tensor([[0.0, -120.0]], dtype=torch.float32)
    _logits, _probs, log_probs, _ = _ensure_logits_and_probs(logits, cfg)
    assert log_probs.dtype == torch.float64 and float(log_probs[0, 1]) == pytest.approx(-120.0, abs=1e-6)


def test_mean_dependencies_are_separated_from_the_requested_metrics():
    from services.side_channel_core.src.sca.metrics import split_mean_dependencies

    results = {"mean_accuracy": 0.5, "accuracy": 0.5, "accuracy_correct": 1, "accuracy_total": 2,
               "mean_fit_time_s": 3.0, "fit_time_s": 3.0, "guessing_entropy": 1.0}
    kept, deps = split_mean_dependencies(results, {"metrics_to_log": ["mean_accuracy", "mean_fit_time_{s}", "guessing_entropy"]})
    assert set(kept) == {"mean_accuracy", "mean_fit_time_s", "guessing_entropy"}
    assert set(deps) == {"accuracy", "accuracy_correct", "accuracy_total", "fit_time_s"}
    kept, deps = split_mean_dependencies(results, {"metrics_to_log": ["accuracy", "mean_accuracy", "fit_time_{s}"]})
    assert deps == {} and "accuracy" in kept and "fit_time_s" in kept


def test_uninformative_model_scores_chance_and_is_never_recovered():
    import torch
    from services.side_channel_core.src.sca.perform_attack import perform_attacks

    n, key = 2000, 0xE0
    rng = np.random.default_rng(0)
    plain = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    meta = [{"plaintext": plain[i], "key": np.full(16, key, np.uint8)} for i in range(n)]
    cfg = {"sca": {"attack_byte": 2, "leakage_model": "ID", "rank_base": 1, "variant": "fixed_key"},
           "perform_attack": {"enabled": True, "attack_step": 2, "ge_thresholds": 1,
                              "test": {"attack_times": 1000, "shuffle": False,
                                       "attack_distribution": "disjoint"}},
           "metrics_cfg": {"rank_log_eps": 1e-40}, "data": {"name": "ascad", "test_range": [0, n]}, "seed": 42}
    uniform = torch.full((n, 256), -np.log(256.0), dtype=torch.float64)
    result = perform_attacks(uniform, meta, cfg, process="test", preds_are_log_probs=True)
    assert abs(result["guessing_entropy"] - 128.5) < 5     # chance, not rank 1
    assert result["success_rate"] < 0.02 and result["sr_trials"] == 1000
    evidence = result["__key_recovery_evidence__"]
    assert evidence["true_key_tie_count"] == 256            # never a unique rank 1
    order = [row["key_guess"] for row in evidence["top_key_candidates"]]
    assert order[evidence["final_true_key_rank"] - 1] == key  # top hypothesis agrees with the rank


def _attack_setup(validation, test):
    n, key = 200, 0xE0
    rng = np.random.default_rng(1)
    plain = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    meta = [{"plaintext": plain[i], "key": np.full(16, key, np.uint8)} for i in range(n)]
    cfg = {"sca": {"attack_byte": 2, "leakage_model": "ID", "rank_base": 1, "variant": "fixed_key"},
           "perform_attack": {"enabled": True, "attack_step": 10, "ge_thresholds": 1,
                              "validation": validation, "test": test},
           "metrics_cfg": {"rank_log_eps": 1e-40},
           "data": {"name": "ascad", "test_range": [0, n], "validation_range": [0, n]}, "seed": 42}
    return rng, meta, cfg


def test_attack_settings_per_process_and_near_tie_endpoint():
    import pytest
    import torch
    from services.side_channel_core.src.sca.perform_attack import perform_attacks

    rng, meta, cfg = _attack_setup(
        {"attack_times": 0, "shuffle": False, "attack_distribution": "full_population"},
        {"attack_times": 20, "shuffle": True, "attack_distribution": "full_population"},
    )
    # Near-tied scores: permuted float64 sums may flip the last bit.
    near_tie = torch.as_tensor(-np.log(256.0) + 1e-15 * rng.standard_normal((200, 256)))
    result = perform_attacks(near_tie, meta, cfg, process="test", preds_are_log_probs=True)
    assert result["attack_repetitions"] == 20
    assert len(set(result["final_ranks"])) == 1
    with pytest.raises(ValueError, match="attack_times is 0"):
        perform_attacks(near_tie, meta, cfg, process="validation", preds_are_log_probs=True)
    cfg["perform_attack"]["attack_repetitions"] = 1
    with pytest.raises(ValueError, match="moved"):
        perform_attacks(near_tie, meta, cfg, process="test", preds_are_log_probs=True)


def test_disjoint_attacks_are_independent_subsets():
    import pytest
    import torch
    from services.side_channel_core.src.sca.perform_attack import perform_attacks

    plan = {"attack_times": 4, "shuffle": False, "attack_distribution": "disjoint"}
    rng, meta, cfg = _attack_setup(plan, plan)
    preds = torch.log_softmax(torch.as_tensor(rng.standard_normal((200, 256))), dim=1)
    result = perform_attacks(preds, meta, cfg, process="test", preds_are_log_probs=True)
    curves = result["per_attack_curves"]
    assert len(curves) == 4 and all(int(c[-1, 0]) == 50 for c in curves)
    assert result["guessing_entropy"] == np.mean(result["final_ranks"])
    assert result["attack_protocol"]["endpoint_is_independent_attack_estimate"] is True
    cfg["perform_attack"]["test"]["attack_times"] = 3
    with pytest.raises(ValueError, match="equal disjoint attacks"):
        perform_attacks(preds, meta, cfg, process="test", preds_are_log_probs=True)


def test_attacks_match_an_independent_rank_computation():
    """Ranks, GE and SR against a from-scratch ASCAD-style computation."""
    import torch
    from services.side_channel_core.src.sca.perform_attack import perform_attacks

    # AES S-box from its definition: inverse in GF(2^8), then the affine map.
    def gmul(a, b):
        r = 0
        while b:
            r ^= a if b & 1 else 0
            a = ((a << 1) ^ 0x11B) if a & 0x80 else a << 1
            b >>= 1
        return r
    rotl = lambda v, s: ((v << s) | (v >> (8 - s))) & 0xFF
    inverse = [0] + [next(y for y in range(1, 256) if gmul(x, y) == 1) for x in range(1, 256)]
    sbox = np.array([i ^ rotl(i, 1) ^ rotl(i, 2) ^ rotl(i, 3) ^ rotl(i, 4) ^ 0x63 for i in inverse])
    assert sbox[0] == 0x63 and sbox[0x53] == 0xED

    n, key, step = 2000, 0xE0, 10
    rng = np.random.default_rng(7)
    plain = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    meta = [{"plaintext": plain[i], "key": np.full(16, key, np.uint8)} for i in range(n)]
    logits = rng.standard_normal((n, 256))
    logits[np.arange(n), sbox[plain[:, 2] ^ key]] += 0.6        # weak leakage
    log_probs = torch.log_softmax(torch.as_tensor(logits), dim=1)

    def reference_curve(rows):
        labels = sbox[plain[rows, 2][:, None] ^ np.arange(256)[None, :]]
        per_key = np.take_along_axis(log_probs.numpy()[rows], labels, axis=1)   # [n, 256]
        scores = np.cumsum(per_key, axis=0)[step - 1::step]
        return 1 + np.count_nonzero(scores > scores[:, [key]], axis=1)

    def attack(times, shuffle, distribution):
        cfg = {"sca": {"attack_byte": 2, "leakage_model": "ID", "rank_base": 1, "variant": "fixed_key"},
               "perform_attack": {"enabled": True, "attack_step": step, "ge_thresholds": 1,
                                  "test": {"attack_times": times, "shuffle": shuffle,
                                           "attack_distribution": distribution}},
               "metrics_cfg": {"rank_log_eps": 1e-40}, "data": {"name": "ascad", "test_range": [0, n]}, "seed": 42}
        return perform_attacks(log_probs, meta, cfg, process="test", preds_are_log_probs=True)

    everything = reference_curve(np.arange(n))

    # Paper protocol: one attack in file order.
    ordered = attack(1, False, "full_population")
    assert np.array_equal(ordered["per_attack_curves"][0][:, 1], everything)

    # Disjoint blocks: each attack is the reference on its own traces.
    blocks = attack(20, False, "disjoint")
    expected = np.stack([reference_curve(np.arange(r * 100, (r + 1) * 100)) for r in range(20)])
    assert np.array_equal(np.stack([c[:, 1] for c in blocks["per_attack_curves"]]), expected)
    assert np.allclose(blocks["mean_attack_rank_curve"][:, 1], expected.mean(axis=0))
    assert blocks["guessing_entropy"] == expected[:, -1].mean()
    assert blocks["success_rate"] == np.mean(expected[:, -1] == 1)
    evidence = blocks["__key_recovery_evidence__"]
    assert evidence["final_true_key_rank"] == everything[-1]      # pooled traces = full attack

    # Shuffled full population: same endpoint, GE(n) agrees with random n-subsets.
    shuffled = attack(100, True, "full_population")
    assert set(shuffled["final_ranks"]) == {int(everything[-1])}
    monte_carlo = np.mean([reference_curve(rng.permutation(n)) for _ in range(100)], axis=0)
    spread = shuffled["std_attack_rank_curve"][:, 1] / 10 + monte_carlo.std() / 10
    assert np.all(np.abs(shuffled["mean_attack_rank_curve"][:, 1] - monte_carlo) <= 5 * spread + 1)
    # The first n traces of random orders and of disjoint random blocks estimate the same GE(n).
    random_blocks = attack(20, True, "disjoint")["mean_attack_rank_curve"][:, 1]
    assert np.all(np.abs(random_blocks - monte_carlo[:10]) <= 5 * shuffled["std_attack_rank_curve"][:10, 1] / np.sqrt(20) + 1)


def test_training_objective_is_read_from_the_fold_trajectory_and_averaged(tmp_path):
    from services.graph_manager.metric_loader import load_source_metric

    run = tmp_path / "m"
    for fold, losses in ((1, [5.6, 5.0]), (2, [5.4, 4.0])):
        (run / "metrics" / "cross_validation" / f"fold_{fold:02d}" / "scalars").mkdir(parents=True)
        folder = run / "cross_validation" / "folds" / f"fold_{fold:02d}"
        folder.mkdir(parents=True)
        (folder / "fold_result.yaml").write_text(yaml.safe_dump(
            {"trajectory": {"epochs": [1, 2], "training_objective": losses, "validation_objective": [6.0, 6.5]}}
        ))
    (run / "metrics" / "attack_metrics").mkdir(parents=True)

    loaded = load_source_metric(source="cross_validation", metrics_dir=run / "metrics" / "attack_metrics",
                                model="m", process="train", metric="training_objective")
    assert loaded.data["training_objective"].tolist() == pytest.approx([5.5, 4.5])
    assert loaded.data["training_objective_std"].tolist() == pytest.approx([np.std([5.6, 5.4], ddof=1), np.std([5.0, 4.0], ddof=1)])
    # the objective belongs to its process: no training objective on the held-out traces
    other = load_source_metric(source="cross_validation", metrics_dir=run / "metrics" / "attack_metrics",
                               model="m", process="validation", metric="training_objective")
    assert other.data is None


def test_line_styles_pick_a_series_by_its_key_metric_or_model():
    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plot_style import get_line_style

    job = validate_graph_job(_job(metric=["training_objective", "cross_entropy"],
                                  style={"line": {"styles": {"cross_entropy": "--", "m2": ":", "m3:cross_entropy": "-."}}}))
    assert get_line_style(job, "m1:cross_entropy") == "--"       # by metric
    assert get_line_style(job, "m1:training_objective") is None    # solid
    assert get_line_style(job, "m2:training_objective") == ":"     # by model
    assert get_line_style(job, "m3:cross_entropy") == "-."         # exact key first
    with pytest.raises(GraphJobValidationError, match="style.line.styles"):
        validate_graph_job(_job(style={"line": {"styles": {"m": "wavy"}}}))


def test_fold_training_time_is_read_and_made_relative_to_a_reference_run(tmp_path):
    from services.graph_manager.graph_manager import relative_series
    from services.graph_manager.metric_loader import load_source_metric

    def run(name, times):
        root = tmp_path / name
        (root / "metrics" / "attack_metrics").mkdir(parents=True)
        for fold, seconds in enumerate(times, 1):
            (root / "metrics" / "cross_validation" / f"fold_{fold:02d}" / "scalars").mkdir(parents=True)
            folder = root / "cross_validation" / "folds" / f"fold_{fold:02d}"
            folder.mkdir(parents=True)
            (folder / "fold_result.yaml").write_text(yaml.safe_dump({"training_time_s": seconds, "epochs_completed": 100}))
        return load_source_metric(source="cross_validation", metrics_dir=root / "metrics" / "attack_metrics",
                                  model=name, process="train", metric="training_time_s").data

    fast, slow = run("fast", [100.0, 110.0]), run("slow", [300.0, 330.0])
    assert fast["training_time_s"].tolist() == [105.0] and fast["epoch"].tolist() == [100]

    rel = relative_series({"fast": fast, "slow": slow}, {"": fast}, "training_time_s")
    assert rel["fast"]["training_time_s"].tolist() == [1.0]
    assert rel["slow"]["training_time_s"].tolist() == pytest.approx([3.0])
    assert rel["slow"]["training_time_s_std"].tolist() == pytest.approx([np.std([300.0, 330.0], ddof=1) / 105.0])

    # Several epochs: divided epoch by epoch; an epoch the reference lacks stops the job.
    ref = pd.DataFrame({"epoch": [1, 2], "a": [2.0, 4.0]})
    other = pd.DataFrame({"epoch": [1, 2], "a": [3.0, 8.0]})
    assert relative_series({"x": other}, {"": ref}, "a")["x"]["a"].tolist() == [1.5, 2.0]
    with pytest.raises(ValueError, match=r"no value at epoch\(s\) \[3\]"):
        relative_series({"x": pd.DataFrame({"epoch": [3], "a": [1.0]})}, {"": ref}, "a", label="r")


def test_relative_block_is_validated():
    from services.graph_manager.graph_job_validator import validate_graph_job

    for empty in (None, "", {}, []):
        assert "relative" not in validate_graph_job(_job(relative=empty))
    job = validate_graph_job(_job(relative={"rule": "run", "where": [{"path": "trainer.batch_size", "equals": 200}]}))
    assert job["relative"] == {"rule": "run", "where": [{"path": "trainer.batch_size", "equals": 200}],
                               "label": "batch_size = 200"}
    assert validate_graph_job(_job(relative={"rule": "probe"}))["relative"] == {
        "rule": "probe", "where": [], "label": "the device probe (probe steps)"}
    anchor = validate_graph_job(_job(relative={"rule": "anchor", "label": "CNNbase", "where": [
        {"path": "model_id", "equals": "ASCAD_CNNbase"}, {"path": "trainer.batch_size", "equals": "200|500"}]}))
    assert anchor["relative"]["label"] == "CNNbase"
    assert anchor["relative"]["where"][1] == {"path": "trainer.batch_size", "in": ["200", "500"]}
    for bad, message in (
        ({"rule": "probe", "where": [{"path": "a", "equals": 1}]}, "it takes no where"),
        ({"rule": "other", "where": [{"path": "a", "equals": 1}]}, "relative.rule must be one of"),
        ({"rule": "run"}, "relative.where needs at least one condition"),
        ({"rule": "run", "where": [{"path": "a"}]}, "needs exactly one operator"),
        ({"rule": "run", "where": [{"path": "a", "equals": 1}], "to": "x"}, "unknown key"),
        ("fast", "relative must be a mapping"),
    ):
        with pytest.raises(GraphJobValidationError, match=message):
            validate_graph_job(_job(relative=bad))
    with pytest.raises(GraphJobValidationError, match="use type: model_overlay"):
        validate_graph_job(_job(type="solo", plot_type="line", relative={"rule": "run", "where": [{"path": "a", "equals": 1}]}))
    for removed in ({"normalize_to": "m"}, {"paper_values": [1]}):
        with pytest.raises(GraphJobValidationError, match="was removed .* use relative"):
            validate_graph_job(_job(**removed))


def test_relative_reference_is_found_by_configuration(tmp_path):
    import json
    from types import SimpleNamespace
    from services.graph_manager.graph_manager import GraphManager

    root = tmp_path / "P"
    (root / "artifacts").mkdir(parents=True)

    def run(h, name, batch, status="DONE", model_id="ASCAD_CNNbase"):
        return {"model_hash": h, "model_name": name, "env": "dev", "status": status, "model_id": model_id,
                "path": f"artifacts/dev/generated_models/{name}", "sampled": {"trainer": {"batch_size": batch}}}

    runs = [run("h1", "b50", 50), run("h2", "b200", 200), run("h3", "b500", 500),
            run("h4", "other_b200", 200, model_id="ASCAD_CNNbest"), run("h5", "failed_b200", 200, status="FAILED")]
    (root / "artifacts" / "run_map.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    gm = SimpleNamespace(project_root=lambda: root, env=lambda: "dev")

    def reference(rule, where, models=("b50", "b200", "b500")):
        job = {"id": 9, "relative": {"rule": rule, "where": where, "label": "L"}}
        return GraphManager.relative_reference(gm, job, list(models))

    batch200 = [{"path": "trainer.batch_size", "equals": 200}]
    assert reference("run", batch200) == {"rule": "run", "model_name": "b200", "model_hash": "h2", "label": "L"}
    # anchor: any finished run of the ENV, listed or not; FAILED runs never anchor.
    with pytest.raises(ValueError, match="matches 2: b200 \\(h2\\), other_b200 \\(h4\\)"):
        reference("anchor", batch200)
    assert reference("anchor", batch200 + [{"path": "model_id", "equals": "ASCAD_CNNbest"}], models=())["model_name"] == "other_b200"
    with pytest.raises(ValueError, match="among the job's models; it matches 0: none"):
        reference("run", [{"path": "trainer.batch_size", "equals": 100}])


def test_relative_bars_mark_the_reference_at_one(tmp_path):
    """Hardware sets absolute speed; runs of one device are compared as ratios to a reference run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from services.graph_manager.graph_manager import relative_series
    from services.graph_manager.plotters import plot_scalar_overlay

    ours = {m: pd.DataFrame({"epoch": [100], "t": [v], "t_std": [10.0]}) for m, v in (("b50", 300.0), ("b200", 100.0))}
    rel = relative_series(ours, {"": ours["b200"]}, "t")
    job = {"plot_type": "barh", "relative": {"rule": "run", "where": [], "label": "batch size 200"},
           "legend": {"enabled": True, "position": "bottom-right"}, "export": {"formats": ["png"]}}
    captured = {}
    real_close = plt.close
    plt.close = lambda fig=None: captured.setdefault("fig", fig)
    try:
        plot_scalar_overlay(series=rel, job=job, out_base=tmp_path / "rel", metric="t")
    finally:
        plt.close = real_close
    ax = captured["fig"].axes[0]
    assert [p.get_width() for p in ax.patches] == pytest.approx([3.0, 1.0])
    assert any(line.get_linestyle() == "--" and list(line.get_xdata()) == [1.0, 1.0] for line in ax.lines)
    real_close(captured["fig"])


def test_axis_margin_pads_the_limits_and_cuts_the_curve_there():
    """style.axes.x_margin: the paper's padded origin; points beyond x_limits are not drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from services.graph_manager.plot_style import apply_axes_style, on_x_grid

    job = {"style": {"axes": {"x_limits": [0, 950], "x_margin": 0.05, "y_limits": [0, 180], "y_margin": 0.05, "x_grid": 50}}}
    x = np.arange(1, 5001, dtype=float)
    xs, ys = on_x_grid(job, x, x * 2)
    assert xs[0] == 50 and xs[-1] == 950 and len(xs) == 19          # every 50 traces, up to 950 only
    fig, ax = plt.subplots()
    apply_axes_style(ax, job)
    assert np.allclose(ax.get_xlim(), (-47.5, 997.5)) and np.allclose(ax.get_ylim(), (-9.0, 189.0))
    plt.close(fig)
    plain = {"style": {"axes": {"x_limits": [0, 1000], "x_grid": 50}}}        # no margin: as before
    xs, _ = on_x_grid(plain, x, x)
    assert xs[-1] == 5000                                                      # the axis clips, the data are not cut
    fig, ax = plt.subplots(); apply_axes_style(ax, plain)
    assert ax.get_xlim() == (0.0, 1000.0); plt.close(fig)


def test_facet_exports_svg_with_its_png_panels_embedded(tmp_path):
    from PIL import Image

    from services.graph_manager.faceting import FacetValidationError, render_facet, validate_facet

    for name, colour in (("a", "red"), ("b", "blue")):
        (tmp_path / "jobs" / f"job_00{1 if name == 'a' else 2}_{name}").mkdir(parents=True)
        Image.new("RGB", (400, 300), colour).save(tmp_path / "jobs" / f"job_00{1 if name == 'a' else 2}_{name}" / f"{name}.png")
    facet = {"id": 3, "name": "svg", "panels": [{"file_id": 1, "address": "jobs/job_001_a/a.png"},
                                                {"file_id": 2, "address": "jobs/job_002_b/b.png", "label": "(b) blue"}],
             "export": {"formats": ["png", "svg"], "png": {"dpi": 100}, "svg": {"on_error": "fail"}},
             "text_rules": [{"target": "title", "text": "Two panels"}]}
    v = validate_facet(facet)
    assert v["export"]["svg"] == {"sanitize": True, "validate_xml": True, "on_error": "fail"}
    written = render_facet(v, graph_root=tmp_path, project_root=tmp_path)
    assert sorted(p.suffix for p in written) == [".png", ".svg"]
    svg = next(p for p in written if p.suffix == ".svg").read_text()
    assert svg.count("<image") == 2 and "Two panels" in svg and "(b) blue" in svg     # panels raster, text vector
    with pytest.raises(FacetValidationError, match="on_error"):
        validate_facet({**facet, "export": {"formats": ["svg"], "svg": {"on_error": "ignore"}}})


def test_a_rank_axis_says_rank_1_is_the_key_recovered_whatever_the_job_writes():
    """TraceFlow ranks from 1, the paper from 0: every rank figure says which, on its axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from services.graph_manager.plot_style import apply_configured_text

    def ylabel(metric, job=None, default="mean rank"):
        fig, ax = plt.subplots()
        apply_configured_text(ax, job or {}, context={"metric": metric}, default_y_label=default)
        text = ax.get_ylabel()
        plt.close(fig)
        return text

    assert ylabel("mean_attack_rank_curve") == "mean rank (1 = key recovered)"
    written = {"style": {"text_rules": [{"target": "y_label", "text": "mean rank"}]}}
    assert ylabel("mean_attack_rank_curve", written) == "mean rank (1 = key recovered)"
    assert ylabel("guessing_entropy", default="Guessing Entropy") == "Guessing Entropy (1 = key recovered)"
    noted = {"style": {"text_rules": [{"target": "y_label", "text": "rank (1 = key recovered)"}]}}
    assert ylabel("key_rank", noted) == "rank (1 = key recovered)"                 # once
    assert ylabel("mean_accuracy", default="mean accuracy") == "mean accuracy"     # not a rank
    assert ylabel("mean_fit_time_s", default="fit time") == "fit time"


@pytest.mark.parametrize("enabled", [True, False])
def test_save_raw_plot_data_writes_what_the_figure_draws_beside_it(tmp_path, enabled):
    """TODO.md §6: export.save_raw_plot_data had no effect (raw_exporter.py was imported nowhere)."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from services.graph_manager.plotters import save_figure

    job = {"id": 3, "output_file_name": "acc", "export": {"formats": ["png"], "save_raw_plot_data": enabled}}
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [0.004, 0.005, 0.0061], label="RMSprop")
    ax.barh([0], [0.0043])
    ax.set_ylabel("mean accuracy")
    ax.legend()
    save_figure(fig, tmp_path / "job_003_acc" / "acc", job)
    raw = tmp_path / "job_003_acc" / "raw_plot_data" / "acc.json"
    if not enabled:
        assert not raw.exists()
        return
    data = json.loads(raw.read_text())["axes"][0]
    assert data["y_label"] == "mean accuracy" and data["legend"] == ["RMSprop"]
    assert data["lines"][0]["x"] == [1.0, 2.0, 3.0] and data["lines"][0]["y"] == [0.004, 0.005, 0.0061]
    assert [round(b["width"], 6) for b in data["bars"]] == [0.0043]


def test_overlay_keeps_a_dropped_models_slot_and_names_it_in_the_legend(tmp_path):
    """A model with NaN every epoch (or no record) is not dropped from the overlay: it keeps its
    colour slot, so the models after it keep their colours and a facet legend built from the job's
    order (faceting._job_entries: C0, C1, ...) names the right run; it stays in the legend with the reason."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from services.graph_manager.faceting import _job_entries
    from services.graph_manager.plotters import plot_curve_overlay, plot_scalar_overlay, slot_colour

    def draw(fn, **kwargs):
        captured = {}
        real_close = plt.close
        plt.close = lambda fig=None: captured.setdefault("fig", fig)
        try:
            fn(job={"legend": {"enabled": True, "position": "bottom-right"}, "export": {"formats": ["png"]}},
               out_base=tmp_path / "o", **kwargs)
        finally:
            plt.close = real_close
        fig = captured["fig"]
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        handles = ax.get_legend().legend_handles
        real_close(fig)
        return ax, texts, handles

    nan = float("nan")
    frames = {"m1": pd.DataFrame({"epoch": [1, 2], "ge": [3.0, 2.0]}),
              "m2": pd.DataFrame({"epoch": [1, 2], "ge": [nan, nan]}),      # CV fold mean NaN every epoch
              "m3": pd.DataFrame({"epoch": [1, 2], "ge": [5.0, 4.0]})}
    ax, texts, handles = draw(plot_scalar_overlay, series=frames, metric="ge")
    drawn = {line.get_label(): line.get_color() for line in ax.lines if len(line.get_xdata())}
    assert drawn == {"m1": "C0", "m3": "C2"}, "the third model keeps C2 although the second drew nothing"
    assert texts == ["m1", "m2 (no value: NaN every epoch)", "m3"]
    assert handles[1].get_linestyle() == "None"                             # a legend entry without a line

    # the order of the job (its models) sets the slots, also for a model never loaded (dropped)
    curves = {"m1": np.array([[1, 200.0], [2, 100.0]]), "m3": np.array([[1, 50.0], [2, 20.0]])}
    ax, texts, handles = draw(plot_curve_overlay, series=curves, metric="rank", x_label="traces", y_label="rank",
                              order=["m1", "m2", "m3"], dropped={"m2": "not recorded"})
    drawn = {line.get_label(): line.get_color() for line in ax.lines if len(line.get_xdata())}
    assert drawn == {"m1": "C0", "m3": "C2"}
    assert texts == ["m1", "m2 (no value: not recorded)", "m3"]

    # the facet legend of the same job names each model by the same slot colour
    entries = _job_entries({"models": ["m1", "m2", "m3"]})
    assert [(e["label"], e["color"]) for e in entries] == [("m1", "C0"), ("m2", "C1"), ("m3", "C2")]
    assert [slot_colour(i) for i in range(3)] == ["C0", "C1", "C2"]


def test_a_facet_legend_caption_is_written_under_the_legend(tmp_path):
    """Owner 2026-10-27 (Fig. 10): legend.caption, e.g. "Fig. a", a line under the facet's legend."""
    from PIL import Image

    from services.graph_manager.faceting import render_facet, validate_facet

    (tmp_path / "jobs" / "job_001_a").mkdir(parents=True)
    Image.new("RGB", (400, 300), "white").save(tmp_path / "jobs" / "job_001_a" / "a.png")
    facet = {"id": 4, "name": "cap", "panels": [{"file_id": 1, "address": "jobs/job_001_a/a.png"}],
             "legend": {"mode": "custom", "position": "right", "labels": ["ReLU", "tanh"], "caption": "Fig. a"},
             "export": {"formats": ["svg"], "png": {"dpi": 100}}}
    v = validate_facet(facet)
    assert v["legend"]["caption"] == "Fig. a"
    svg = render_facet(v, graph_root=tmp_path, project_root=tmp_path)[0].read_text()
    assert "Fig. a" in svg and "ReLU" in svg
    assert validate_facet({**facet, "legend": {**facet["legend"], "caption": None}})["legend"]["caption"] == ""


def test_a_facet_indexes_stacked_panels_on_the_left_in_roman_numerals():
    """Owner 2026-10-27: the figures stacked one per row, indexed I, II, III, ... on their left."""
    from services.graph_manager.faceting import _roman, build_layout, place_panels, validate_facet

    assert [_roman(n) for n in (1, 2, 3, 4, 6, 9, 14)] == ["I", "II", "III", "IV", "VI", "IX", "XIV"]
    facet = validate_facet({"id": 7, "panels": [
        {"file_id": k, "address": f"p{k}.png", "label": "{roman}", "label_position": "left"} for k in (1, 2, 3)],
        "arrangement": {"mode": "vertical"}})
    panels = place_panels(facet)
    for panel in panels:
        panel.aspect, panel.cell_width = 2.0, 2.0 * facet["panel_height"]
    layout = build_layout(facet, panels)
    labels = [(x, y, text) for x, y, text, kw in layout.texts if text in ("I", "II", "III")]
    assert [t for _, _, t in labels] == ["I", "II", "III"]
    for (x, y, _), (bx, by, bw, bh) in zip(labels, layout.boxes):
        assert x < bx and by < y < by + bh                  # left of its own panel, at its height


def test_an_svg_facet_nests_its_panels_svgs_with_their_ids_prefixed(tmp_path):
    """Owner 2026-10-27: SVG only; a facet of SVG panels is all vector, each panel's ids its own."""
    import re

    import matplotlib.pyplot as plt

    from services.graph_manager.faceting import render_facet, validate_facet

    for k in (1, 2):
        folder = tmp_path / "jobs" / f"job_00{k}_p{k}"
        folder.mkdir(parents=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [0, k], label=f"panel {k}")
        ax.legend()
        fig.savefig(folder / f"p{k}.svg", format="svg")
        plt.close(fig)
    facet = {"id": 5, "name": "nest", "panels": [{"file_id": k, "address": f"jobs/job_00{k}_p{k}/p{k}.svg"} for k in (1, 2)],
             "export": {"formats": ["svg"]}, "text_rules": [{"target": "title", "text": "Two"}]}
    v = validate_facet(facet)
    assert v["export"]["formats"] == ["svg"] and validate_facet({**facet, "export": {}})["export"]["formats"] == ["svg"]
    written = render_facet(v, graph_root=tmp_path, project_root=tmp_path)
    svg = written[0].read_text()
    assert [p.suffix for p in written] == [".svg"] and "<image" not in svg and svg.count("<svg") == 3    # page + 2 panels
    ids = re.findall(r'\bid="([^"]+)"', svg)
    assert len(ids) == len(set(ids))                                           # no clash between the panels
    assert any(i.startswith("panel1_") for i in ids) and any(i.startswith("panel2_") for i in ids)
    refs = set(re.findall(r'href="#([^"]+)"', svg)) | set(re.findall(r"url\(#([^)]+)\)", svg))
    assert refs <= set(ids)                                                    # every reference still resolves


def test_a_stack_fit_to_width_gives_every_row_the_widest_rows_width():
    """Owner 2026-10-27: the stacked figures I to VI meet both margins (arrangement.fit: width)."""
    from services.graph_manager.faceting import build_layout, place_panels, validate_facet

    facet = validate_facet({"id": 9, "panels": [{"file_id": k, "address": f"p{k}.svg"} for k in (1, 2, 3)],
                            "arrangement": {"mode": "vertical", "fit": "width"}})
    assert facet["arrangement"]["fit"] == "width"
    panels = place_panels(facet)
    for panel, aspect in zip(panels, (3.0, 2.0, 2.5)):
        panel.aspect, panel.cell_width = aspect, aspect * facet["panel_height"]
    boxes = build_layout(facet, panels).boxes
    lefts, rights = {round(x, 6) for x, _, _, _ in boxes}, {round(x + w, 6) for x, _, w, _ in boxes}
    assert len(lefts) == 1 and len(rights) == 1                               # one left and one right edge
    assert [round(w / h, 6) for _, _, w, h in boxes] == [3.0, 2.0, 2.5]          # each keeps its proportions
    with pytest.raises(Exception, match="arrangement.fit"):
        validate_facet({"id": 9, "panels": [{"file_id": 1, "address": "p1.svg"}], "arrangement": {"fit": "both"}})


def test_a_grid_fit_natural_draws_every_panel_at_its_own_size():
    """Owner 2026-10-27 (Fig. 5.ii): six heat maps of different heights, three rows of two, one text size in all."""
    from services.graph_manager.faceting import build_layout, place_panels, validate_facet

    facet = validate_facet({"id": 14, "panels": [{"file_id": k, "address": f"p{k}.svg"} for k in (1, 2, 3, 4)],
                            "arrangement": {"mode": "grid", "fit": "natural", "order": [[1, 2], [3, 4]]}})
    assert facet["arrangement"]["fit"] == "natural"
    panels = place_panels(facet)
    sizes = ((5.0, 3.0), (5.0, 4.0), (4.0, 3.5), (5.0, 2.5))
    for panel, (w, h) in zip(panels, sizes):
        panel.aspect, panel.size = w / h, (w, h)
    boxes = build_layout(facet, panels).boxes
    assert [(round(w, 6), round(h, 6)) for _, _, w, h in boxes] == [s for s in sizes]   # inches, unscaled
    tops = [round(y + h, 6) for _, y, _, h in boxes]
    assert tops[0] == tops[1] and tops[2] == tops[3]                           # a row's panels top-aligned
    panels[0].size = None
    with pytest.raises(Exception, match="SVG or PDF"):
        build_layout(facet, panels)


def test_a_facet_legend_colormap_draws_one_spectrum_above_its_entries():
    """Owner 2026-10-28 (Fig. 5.ii): the spectrum the six heat maps share, the grey cell's meaning under it."""
    import matplotlib.pyplot as plt

    from services.graph_manager.faceting import _figure, build_layout, place_panels, validate_facet

    raw = {"id": 14, "panels": [{"file_id": 1, "address": "p1.svg"}],
           "legend": {"enabled": True, "position": "top", "mode": "custom", "handle": ["patch", 8],
                      "colormap": {"name": "turbo", "limits": [0, 5000], "label": "traces"},
                      "entries": [{"label": "never", "color": "0.85"}]}}
    facet = validate_facet(raw)
    assert facet["legend"]["colormap"] == {"name": "turbo", "limits": [0.0, 5000.0], "label": "traces"}
    panels = place_panels(facet)
    panels[0].aspect, panels[0].cell_width = 1.5, 1.5 * facet["panel_height"]
    layout = build_layout(facet, panels)
    fig = _figure(layout)
    bars = [a for a in fig.axes if a.get_xlim() == (0.0, 5000.0)]
    assert len(bars) == 1 and bars[0].get_xlabel() == "traces"
    legend_top = fig.legends[0].get_window_extent(fig.canvas.get_renderer()).y1
    assert legend_top < bars[0].get_window_extent(fig.canvas.get_renderer()).y0      # the entries under the spectrum
    plt.close(fig)
    for bad, message in (({"name": "rgb", "limits": [0, 1]}, "colormap"), ({"limits": [1, 0]}, "low < high")):
        with pytest.raises(Exception, match=message):
            validate_facet({**raw, "legend": {**raw["legend"], "colormap": bad}})


def test_a_facet_legend_caption_takes_bold_italic_and_line_breaks():
    """Owner 2026-10-28 (Fig. 5.ii): "the number in a cell:" bold, "in colour" and "in grey" italic, on two lines."""
    import matplotlib.pyplot as plt

    from services.graph_manager.faceting import _figure, build_layout, place_panels, validate_facet

    facet = validate_facet({"id": 14, "panels": [{"file_id": 1, "address": "p1.svg"}],
                            "legend": {"enabled": True, "position": "top", "mode": "custom",
                                       "caption": "**the number:** *in colour*, traces;\n*in grey*, rank",
                                       "entries": [{"label": "never", "color": "0.85"}]}})
    panels = place_panels(facet)
    panels[0].aspect, panels[0].cell_width = 1.5, 1.5 * facet["panel_height"]
    fig = _figure(build_layout(facet, panels))
    looks = {t.get_text(): (t.get_fontweight(), t.get_fontstyle(), round(t.get_position()[1], 6)) for t in fig.texts}
    assert looks["the number:"][:2] == ("bold", "normal") and looks["in colour"][:2] == ("normal", "italic")
    assert looks["in grey"][1] == "italic" and looks["in grey"][2] < looks["in colour"][2]      # the second line under
    assert looks[", traces;"][2] == looks["in colour"][2]
    plt.close(fig)
