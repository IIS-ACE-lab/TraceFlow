"""GraphManager binds every job model (model_name or model_hash) to its run (SYS-08)."""

import json

import pytest

from services.graph_manager.graph_job_validator import (
    GraphJobValidationError,
    get_graph_job_by_id,
    resolve_model_refs,
)


REGISTRY = {
    "aaaa1111": {"model_name": "mesh_chicken", "env": "dev", "status": "DONE"},
    "bbbb2222": {"model_name": "ultimate_jewel", "env": "dev", "status": "RUNNING"},
    "cccc3333": {"model_name": "square_daffodil", "env": "prod", "status": "DONE"},
}
NAME_MAP = {h: e["model_name"] for h, e in REGISTRY.items()}


def _bind(refs, registry=REGISTRY, name_map=NAME_MAP, env="dev"):
    return resolve_model_refs(refs, registry=registry, name_map=name_map, env=env, job_id=7)


def test_a_model_is_listed_by_name_or_by_hash():
    bound = _bind(["mesh_chicken", "bbbb2222", "aaaa1111"])
    assert [(r["model_name"], r["model_hash"], r["status"]) for r in bound] == [
        ("mesh_chicken", "aaaa1111", "DONE"),
        ("ultimate_jewel", "bbbb2222", "RUNNING"),
    ]                                                  # the hash of a listed name adds nothing
    assert [r["ref"] for r in bound] == ["mesh_chicken", "bbbb2222"]


def test_unknown_entries_fail_together_with_close_matches():
    with pytest.raises(GraphJobValidationError) as err:
        _bind(["mesh_chiken", "aaaa1112", "mesh_chicken"])
    text = str(err.value)
    assert "graph job id=7" in text and "ENV=dev" in text
    assert "mesh_chiken: no model_name or model_hash" in text and "'mesh_chicken'" in text
    assert "aaaa1112: no model_name or model_hash" in text and "'aaaa1111'" in text
    assert "square_daffodil" not in text              # another env is not offered


def test_registries_that_disagree_fail():
    renamed = {**REGISTRY, "aaaa1111": {**REGISTRY["aaaa1111"], "model_name": "other_name"}}
    with pytest.raises(GraphJobValidationError, match="registries disagree"):
        _bind(["mesh_chicken"], registry=renamed)
    with pytest.raises(GraphJobValidationError, match="registries disagree"):
        _bind(["aaaa1111"], registry=renamed)

    # A deleted run: its name or hash may linger in the name map only.
    deleted = {h: e for h, e in REGISTRY.items() if h != "aaaa1111"}
    with pytest.raises(GraphJobValidationError, match="deleted"):
        _bind(["mesh_chicken"], registry=deleted)
    with pytest.raises(GraphJobValidationError, match="has no run of it"):
        _bind(["aaaa1111"], registry=deleted)

    # The same name on two hashes: only the hash says which run is meant.
    twice = {**NAME_MAP, "dddd4444": "mesh_chicken"}
    twice_runs = {**REGISTRY, "dddd4444": {"model_name": "mesh_chicken", "env": "dev", "status": "DONE"}}
    with pytest.raises(GraphJobValidationError, match="list the model_hash"):
        _bind(["mesh_chicken"], registry=twice_runs, name_map=twice)
    assert _bind(["dddd4444"], registry=twice_runs, name_map=twice)[0]["model_hash"] == "dddd4444"


def test_a_run_of_another_env_names_its_env():
    with pytest.raises(GraphJobValidationError, match="trained with ENV=prod; run the graph job with ENV=prod"):
        _bind(["square_daffodil"])
    assert _bind(["cccc3333"], env="prod")[0]["model_name"] == "square_daffodil"


def test_validator_accepts_names_and_hashes_and_rejects_numbers():
    job = {"id": 3, "name": "j", "type": "model_overlay", "enabled": True, "metric": "accuracy",
           "models": ["mesh_chicken", " aaaa1111 "], "processes": ["train"], "output_file_name": "j"}
    assert get_graph_job_by_id([job], 3)["models"] == ["mesh_chicken", "aaaa1111"]
    # YAML reads an unquoted 01234567 as the octal number 342391.
    for bad in ([342391], ["mesh_chicken", None]):
        with pytest.raises(GraphJobValidationError, match="quote a hash"):
            get_graph_job_by_id([{**job, "models": bad}], 3)
    job0 = {**job, "id": 0, "top_models": ["aaaa1111"]}
    assert get_graph_job_by_id([job0], 0)["models"] == ["aaaa1111"]


def _project(tmp_path, registry=REGISTRY, name_map=NAME_MAP):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "run_registry.json").write_text(json.dumps(registry))
    (artifacts / "model_name_map.json").write_text(json.dumps(name_map))
    for name in ("mesh_chicken", "ultimate_jewel"):
        scalars = artifacts / "dev" / "generated_models" / name / "metrics" / "attack_metrics" / "scalars"
        scalars.mkdir(parents=True)
        (scalars / f"{name}_metrics_train_history.jsonl").write_text(
            "".join(json.dumps({"epoch": e, "accuracy": 0.1 * e}) + "\n" for e in (1, 2, 3))
        )


def _manager(tmp_path, monkeypatch, jobs):
    from services.graph_manager.graph_manager import GraphManager

    monkeypatch.setenv("ENV", "dev")
    manager = GraphManager.__new__(GraphManager)
    manager.cfg = {"project": {"root": str(tmp_path)}, "env": "dev"}
    manager.graph_cfg = {"jobs": jobs}
    manager.jobs = jobs
    manager.ui_cfg, manager.should_open_ui = {}, False
    return manager


def _overlay_job(models, labels):
    return {"id": 3, "name": "acc", "type": "model_overlay", "enabled": True, "metric": "accuracy",
            "source": "attack_metrics", "models": models, "processes": ["train"],
            "output_file_name": "acc", "legend": {"mode": "custom", "labels": labels}}


def test_graph_job_listed_by_hash_and_name_plots_series_by_name(tmp_path, monkeypatch, capsys):
    import services.graph_manager.graph_manager as gm

    _project(tmp_path)
    job = _overlay_job(["aaaa1111", "ultimate_jewel"], {"aaaa1111": "batch 50", "ultimate_jewel": "batch 100"})
    manager = _manager(tmp_path, monkeypatch, [job])

    drawn = {}
    real = gm.plot_scalar_overlay

    def spy(**kwargs):
        drawn["series"] = list(kwargs["series"])
        drawn["labels"] = kwargs["job"]["legend"]["labels"]
        return real(**kwargs)

    monkeypatch.setattr(gm, "plot_scalar_overlay", spy)
    manager.execute(3)

    assert drawn["series"] == ["mesh_chicken", "ultimate_jewel"]       # readable legend keys
    assert drawn["labels"]["mesh_chicken"] == "batch 50"                # label given for the hash
    assert list((tmp_path / "graphs" / "jobs").rglob("*.svg"))             # SVG only, the default
    out = capsys.readouterr().out
    assert "model_name=mesh_chicken model_hash=aaaa1111 status=DONE (listed as aaaa1111)" in out
    assert "model_name=ultimate_jewel model_hash=bbbb2222 status=RUNNING" in out
    assert "[WARN] model=ultimate_jewel (bbbb2222) has training status RUNNING" in out


def test_relative_job_divides_every_model_by_the_reference_run(tmp_path, monkeypatch, capsys):
    """relative: the reference is found by configuration, read as the models are, and becomes 1.0."""
    import services.graph_manager.graph_manager as gm

    _project(tmp_path)
    scalars = tmp_path / "artifacts/dev/generated_models/ultimate_jewel/metrics/attack_metrics/scalars"
    (scalars / "ultimate_jewel_metrics_train_history.jsonl").write_text(
        "".join(json.dumps({"epoch": e, "accuracy": 0.3 * e}) + "\n" for e in (1, 2, 3))
    )
    (tmp_path / "artifacts" / "run_map.jsonl").write_text("".join(
        json.dumps({"model_hash": h, "model_name": n, "env": "dev", "status": s, "sampled": {"trainer": {"batch_size": b}}}) + "\n"
        for h, n, s, b in (("aaaa1111", "mesh_chicken", "DONE", 50), ("bbbb2222", "ultimate_jewel", "RUNNING", 100))
    ))
    job = _overlay_job(["mesh_chicken", "ultimate_jewel"], {})
    job["relative"] = {"rule": "run", "where": [{"path": "trainer.batch_size", "equals": 50}], "label": "batch 50"}
    job["style"] = {"text_rules": [{"target": "y_label", "text": "accuracy relative to {relative_to}"}]}
    manager = _manager(tmp_path, monkeypatch, [job])

    drawn = {}
    real = gm.plot_scalar_overlay

    def spy(**kwargs):
        drawn.update(series={k: v["accuracy"].tolist() for k, v in kwargs["series"].items()},
                     context=kwargs["job"]["active_context"])
        return real(**kwargs)

    monkeypatch.setattr(gm, "plot_scalar_overlay", spy)
    manager.execute(3)

    assert drawn["series"]["mesh_chicken"] == pytest.approx([1.0, 1.0, 1.0])
    assert drawn["series"]["ultimate_jewel"] == pytest.approx([3.0, 3.0, 3.0])
    assert drawn["context"]["relative_to"] == "batch 50"
    assert "relative (rule run): reference = mesh_chicken (aaaa1111), batch 50; its value is 1.0" in capsys.readouterr().out


def _fit_time_project(tmp_path, devices):
    """Two runs with mean_fit_time_s 100 s and 300 s and the given device records."""
    import yaml

    tmp_path.mkdir(parents=True, exist_ok=True)
    _project(tmp_path)
    for name, seconds in (("mesh_chicken", 100.0), ("ultimate_jewel", 300.0)):
        run = tmp_path / "artifacts/dev/generated_models" / name
        (run / "metrics/attack_metrics/scalars" / f"{name}_metrics_train_history.jsonl").write_text(
            json.dumps({"epoch": 1, "fit_time_s": seconds}) + "\n"
        )
        if devices.get(name):
            (run / "device.yaml").write_text(yaml.safe_dump({"run_device": devices[name]}))
    (tmp_path / "artifacts" / "run_map.jsonl").write_text("".join(
        json.dumps({"model_hash": h, "model_name": n, "env": "dev", "status": "DONE", "sampled": {"trainer": {"batch_size": b}}}) + "\n"
        for h, n, b in (("aaaa1111", "mesh_chicken", 50), ("bbbb2222", "ultimate_jewel", 100))
    ))
    job = _overlay_job(["mesh_chicken", "ultimate_jewel"], {})
    job["metric"] = "mean_fit_time_{s}"
    return job


def _device(name, probe=None, torch_version="2.9.1"):
    record = {"kind": "cuda", "name": name, "software": {"torch": torch_version, "cuda": "12.8", "cudnn": 91002},
              "precision": {"cudnn_allow_tf32": True, "matmul_allow_tf32": False}}
    if probe:
        from utils.device_record import PROBE_SPEC

        record["probe"] = {"version": PROBE_SPEC["version"], "seconds_per_step": probe}
    return record


def _drawn(manager, monkeypatch):
    import services.graph_manager.graph_manager as gm

    drawn = {}
    real = gm.plot_scalar_overlay

    def spy(**kwargs):
        drawn.update({k: v[kwargs["y_col"]].tolist() for k, v in kwargs["series"].items()})
        return real(**kwargs)

    monkeypatch.setattr(gm, "plot_scalar_overlay", spy)
    manager.execute(3)
    return drawn


def test_probe_rule_divides_each_run_by_its_own_device_probe(tmp_path, monkeypatch):
    """A slow and a fast device that did the same work read alike in probe steps."""
    job = _fit_time_project(tmp_path, {"mesh_chicken": _device("GTX 1080 Ti", probe=0.5),
                                       "ultimate_jewel": _device("RTX 5090", probe=1.5)})
    job["relative"] = {"rule": "probe"}
    drawn = _drawn(_manager(tmp_path, monkeypatch, [job]), monkeypatch)
    assert drawn == {"mesh_chicken": pytest.approx([200.0]), "ultimate_jewel": pytest.approx([200.0])}

    # A run without a probe stops the job and is named.
    job = _fit_time_project(tmp_path / "x", {"mesh_chicken": _device("RTX 5090", probe=0.5)})
    job["relative"] = {"rule": "probe"}
    with pytest.raises(ValueError, match=r"none for \['ultimate_jewel'\]"):
        _manager(tmp_path / "x", monkeypatch, [job]).execute(3)


def test_run_rule_refuses_times_of_two_devices(tmp_path, monkeypatch, capsys):
    where = [{"path": "trainer.batch_size", "equals": 50}]
    job = _fit_time_project(tmp_path, {"mesh_chicken": _device("RTX 5090"), "ultimate_jewel": _device("RTX 5090")})
    job["relative"] = {"rule": "run", "where": where}
    assert _drawn(_manager(tmp_path, monkeypatch, [job]), monkeypatch) == {
        "mesh_chicken": pytest.approx([1.0]), "ultimate_jewel": pytest.approx([3.0])}

    job = _fit_time_project(tmp_path / "x", {"mesh_chicken": _device("RTX 5090"),
                                             "ultimate_jewel": _device("RTX 5090", torch_version="2.10.0")})
    job["relative"] = {"rule": "run", "where": where}
    with pytest.raises(ValueError, match=r"ultimate_jewel \(software.torch\).*rule probe"):
        _manager(tmp_path / "x", monkeypatch, [job]).execute(3)

    # Without device records the ratio is drawn and the runs are named in a warning.
    job = _fit_time_project(tmp_path / "y", {})
    job["relative"] = {"rule": "run", "where": where}
    _drawn(_manager(tmp_path / "y", monkeypatch, [job]), monkeypatch)
    assert "no device record for ['mesh_chicken', 'ultimate_jewel']" in capsys.readouterr().out


def test_graph_job_with_an_unbound_model_or_damaged_registry_fails(tmp_path, monkeypatch):
    _project(tmp_path)
    manager = _manager(tmp_path, monkeypatch, [_overlay_job(["mesh_chiken"], {})])
    with pytest.raises(GraphJobValidationError, match="close matches: \\['mesh_chicken'\\]"):
        manager.execute(3)
    assert not (tmp_path / "artifacts" / "dev" / "generated_models" / "mesh_chiken").exists()

    (tmp_path / "artifacts" / "run_registry.json").write_text("{")
    manager = _manager(tmp_path, monkeypatch, [_overlay_job(["mesh_chicken"], {})])
    with pytest.raises(ValueError, match="not valid JSON"):
        manager.execute(3)


# ── style.axes.x_grid (2026-10-10): the paper's trace grid at drawing time ──
def test_x_grid_draws_only_the_grid_points_of_a_dense_curve():
    import numpy as np
    from services.graph_manager.plot_style import on_x_grid

    job = {"style": {"axes": {"x_grid": 50}}}
    n = np.arange(1, 1001, dtype=float)              # a curve recorded at every trace
    rank = np.sqrt(n)
    std = rank / 10
    x, y, s = on_x_grid(job, n, rank, std)
    assert x.tolist() == list(range(50, 1001, 50))   # what a rank_step 50 run records
    assert np.array_equal(y, np.sqrt(x)) and np.array_equal(s, np.sqrt(x) / 10)
    # no x_grid: unchanged
    x2, y2 = on_x_grid({"style": {"axes": {}}}, n, rank)
    assert np.array_equal(x2, n) and np.array_equal(y2, rank)


# ── style.axes.x_grid_first (2026-10-12): the paper's one-trace point at x = 0 ──
def test_x_grid_first_keeps_the_first_recorded_point():
    import numpy as np
    from services.graph_manager.plot_style import on_x_grid

    n = np.arange(1, 1001, dtype=float)              # recorded at every trace
    job = {"style": {"axes": {"x_grid": 50, "x_grid_first": True}}}
    x, y = on_x_grid(job, n, np.sqrt(n))
    assert x.tolist() == [1] + list(range(50, 1001, 50))
    assert np.array_equal(y, np.sqrt(x))
    # with a margin, the cut to x_limits still applies after the first point is kept
    padded = {"style": {"axes": {"x_grid": 50, "x_grid_first": True, "x_limits": [0, 950], "x_margin": 0.05}}}
    assert on_x_grid(padded, n, n)[0].tolist() == [1] + list(range(50, 951, 50))
    # a run recorded every 50 traces: its first point is on the grid, nothing added
    step50 = np.arange(50, 1001, 50, dtype=float)
    assert on_x_grid(job, step50, step50)[0].tolist() == step50.tolist()
    # false (the default): as before
    assert on_x_grid({"style": {"axes": {"x_grid": 50}}}, n, n)[0].tolist() == list(range(50, 1001, 50))


def test_x_grid_first_is_validated():
    from services.graph_manager.graph_job_validator import GraphJobValidationError, normalize_style_config

    axes = normalize_style_config({"id": 1, "style": {"axes": {"x_grid": 50, "x_grid_first": True}}})["axes"]
    assert axes["x_grid_first"] is True
    assert normalize_style_config({"id": 1, "style": {"axes": {}}})["axes"]["x_grid_first"] is False
    with pytest.raises(GraphJobValidationError, match="x_grid_first must be true or false"):
        normalize_style_config({"id": 1, "style": {"axes": {"x_grid": 50, "x_grid_first": "yes"}}})
    with pytest.raises(GraphJobValidationError, match="set style.axes.x_grid too"):
        normalize_style_config({"id": 1, "style": {"axes": {"x_grid_first": True}}})


def test_project_graph_config_styles_validate():
    import yaml
    from pathlib import Path
    from services.graph_manager.graph_job_validator import normalize_style_config

    root = Path(__file__).resolve().parents[4]
    jobs = yaml.safe_load((root / "projects/DLSCA_ASCAD/config/graph_config.yaml").read_text())["jobs"]
    styles = {job["id"]: normalize_style_config(job)["axes"] for job in jobs}     # every project job's style validates
    assert all("x_grid_first" in job["style"]["axes"] for job in jobs)
    assert all(isinstance(axes["x_grid_first"], bool) for axes in styles.values())
    # x_grid_first with x_grid, as a run recorded every 50 traces draws it (no project job is relied on:
    # the project's jobs are the owner's and change)
    axes = normalize_style_config({"id": 1, "style": {"axes": {"x_grid": 50, "x_grid_first": True}}})["axes"]
    assert axes["x_grid_first"] is True and axes["x_grid"] == 50.0


def test_x_grid_is_validated_like_x_step():
    from services.graph_manager.graph_job_validator import GraphJobValidationError, normalize_style_config

    assert normalize_style_config({"id": 1, "style": {"axes": {"x_grid": 50}}})["axes"]["x_grid"] == 50.0
    with pytest.raises(GraphJobValidationError, match="x_grid"):
        normalize_style_config({"id": 1, "style": {"axes": {"x_grid": 0}}})


def test_metric_process_reads_each_metric_from_its_own_process_in_one_figure(tmp_path, monkeypatch):
    """Owner 2026-10-27 (Fig. 10 side panel): the validation accuracy beside the training fit time, relative to one
    run, one bar per model and metric; a model keeps its colour, a metric its hatch."""
    import services.graph_manager.graph_manager as gm
    import services.graph_manager.plotters as plotters

    _project(tmp_path)
    root = tmp_path / "artifacts/dev/generated_models"
    for name, accuracy, seconds in (("mesh_chicken", 0.2, 100.0), ("ultimate_jewel", 0.1, 300.0)):
        scalars = root / name / "metrics" / "attack_metrics" / "scalars"
        (scalars / f"{name}_metrics_train_history.jsonl").write_text(
            json.dumps({"epoch": 1, "accuracy": 0.9, "fit_time_s": seconds}) + "\n")
        (scalars / f"{name}_metrics_validation_history.jsonl").write_text(
            json.dumps({"epoch": 1, "accuracy": accuracy, "fit_time_s": 1.0}) + "\n")
    (tmp_path / "artifacts" / "run_map.jsonl").write_text("".join(
        json.dumps({"model_hash": h, "model_name": n, "env": "dev", "status": "DONE", "sampled": {"model_id": m}}) + "\n"
        for h, n, m in (("aaaa1111", "mesh_chicken", "base"), ("bbbb2222", "ultimate_jewel", "other"))))
    job = {**_overlay_job(["mesh_chicken", "ultimate_jewel"], {}), "metric": ["accuracy", "fit_time_{s}"],
           "metric_process": {"fit_time_{s}": "train"}, "processes": ["validation"], "plot_type": "barh",
           "array_epoch": {"mode": "latest"},
           "relative": {"rule": "run", "where": [{"path": "model_id", "equals": "base"}], "label": "base"}}
    drawn, bars = {}, {}
    real, real_bars = gm.plot_scalar_overlay, plotters._draw_categorical_bars

    def spy(**kwargs):
        drawn.update({k: v["value"].tolist() for k, v in kwargs["series"].items()})
        return real(**kwargs)

    def spy_bars(ax, labels, values, std, **kwargs):
        bars.update(colours=kwargs["colours"], hatches=kwargs["hatches"])
        return real_bars(ax, labels, values, std, **kwargs)

    monkeypatch.setattr(gm, "plot_scalar_overlay", spy)
    monkeypatch.setattr(plotters, "_draw_categorical_bars", spy_bars)
    _manager(tmp_path, monkeypatch, [job]).execute(3)

    assert drawn == {"mesh_chicken:accuracy": [1.0], "mesh_chicken:fit_time_s": [1.0],
                     "ultimate_jewel:accuracy": [pytest.approx(0.5)],          # validation: 0.1 / 0.2
                     "ultimate_jewel:fit_time_s": [pytest.approx(3.0)]}        # train: 300 s / 100 s
    c0, c1 = plotters.slot_colour(0), plotters.slot_colour(1)
    assert bars == {"colours": [c0, c0, c1, c1], "hatches": ["", "////", "", "////"]}
    assert list((tmp_path / "graphs" / "jobs").rglob("validation_train/acc__validation_train__accuracy_fit_time_s.svg"))


@pytest.mark.parametrize("change, message", [
    ({"metric_process": {"loss": "train"}}, "is not a metric of the job"),
    ({"metric_process": {"fit_time_{s}": "refit"}}, "is not one of"),
    ({"processes": ["train", "validation"]}, "processes lists one process"),
    ({"type": "solo"}, "is for a model_overlay"),
])
def test_metric_process_refusals(change, message):
    from services.graph_manager.graph_job_validator import validate_graph_job

    job = {**_overlay_job(["mesh_chicken"], {}), "metric": ["accuracy", "fit_time_{s}"],
           "metric_process": {"fit_time_{s}": "train"}, "processes": ["validation"], **change}
    if job["type"] == "solo":
        job["metric"] = "accuracy"
    with pytest.raises(GraphJobValidationError, match=message):
        validate_graph_job(job)


def test_style_series_sets_a_runs_colour_and_hatch_and_legend_entries_replace_the_series(tmp_path, monkeypatch):
    """Owner 2026-10-27 (Fig. 10 c): both desynchronizations in one bar panel, each run its activation's colour
    and a hatch per N_max, the legend naming the hatches only."""
    import services.graph_manager.plotters as plotters

    _project(tmp_path)
    job = {**_overlay_job(["mesh_chicken", "ultimate_jewel"], {}), "plot_type": "barh", "array_epoch": {"mode": "latest"},
           "style": {"series": {"mesh_chicken": {"color": "C2", "hatch": ""}, "ultimate_jewel": {"color": "C2", "hatch": "////"}}},
           "legend": {"position": "bottom-right", "entries": [{"label": "N 0"}, {"label": "N 100", "hatch": "////"}]}}
    bars, legends = {}, []
    real_bars = plotters._draw_categorical_bars

    def spy_bars(ax, labels, values, std, **kwargs):
        bars.update(colours=kwargs["colours"], hatches=kwargs["hatches"])
        real_bars(ax, labels, values, std, **kwargs)
        real_legend = ax.legend
        ax.legend = lambda *a, **k: legends.append(a) or real_legend(*a, **k)

    monkeypatch.setattr(plotters, "_draw_categorical_bars", spy_bars)
    _manager(tmp_path, monkeypatch, [job]).execute(3)

    assert bars == {"colours": ["C2", "C2"], "hatches": ["", "////"]}
    handles, labels = legends[0]
    assert labels == ["N 0", "N 100"] and [h.get_hatch() or "" for h in handles] == ["", "////"]


@pytest.mark.parametrize("change, message", [
    ({"style": {"series": {"mesh_chicken": {"color": "not-a-colour"}}}}, "is not a Matplotlib colour"),
    ({"style": {"series": {"mesh_chicken": {"hatch": "ab"}}}}, "a hatch is made of"),
    ({"style": {"series": {"mesh_chicken": {"alpha": 0.5}}}}, "takes color, hatch, width and dashes"),
    ({"legend": {"entries": [{"color": "C0"}]}}, "with a label"),
])
def test_style_series_and_legend_entries_refusals(change, message):
    from services.graph_manager.graph_job_validator import validate_graph_job

    with pytest.raises(GraphJobValidationError, match=message):
        validate_graph_job({**_overlay_job(["mesh_chicken"], {}), **change})


def _kr_folds(tmp_path, name, folds):
    """Cross-validation key-recovery reports of one run: folds = [(stable_rank1_trace, final rank) per fold]."""
    for i, (stable, rank) in enumerate(folds, start=1):
        kr = tmp_path / "artifacts/dev/generated_models" / name / "metrics/cross_validation" / f"fold_{i:02d}" / "key_recovery"
        kr.mkdir(parents=True)
        rows = [{"epoch": 1, "key": {"final_true_key_rank": 200}, "recovery": {"stable_rank1_trace": None}},
                {"epoch": 2, "key": {"final_true_key_rank": rank}, "recovery": {"stable_rank1_trace": stable}}]
        (kr / "validation_history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_heatmap_draws_a_model_per_row_and_a_fold_per_column_at_the_final_epoch(tmp_path, monkeypatch):
    """Owner 2026-10-27 (Fig. 20 c): each fold's stable rank-1 trace; a fold never recovered shows its final rank."""
    import services.graph_manager.graph_manager as gm

    _project(tmp_path)
    _kr_folds(tmp_path, "mesh_chicken", [(201, 1), (1793, 1), (2248, 1)])
    _kr_folds(tmp_path, "ultimate_jewel", [(None, 78), (495, 1), (None, 200)])
    job = {**_overlay_job(["mesh_chicken", "ultimate_jewel"], {"mesh_chicken": "RMSprop", "ultimate_jewel": "Adam"}),
           "source": "key_recovery", "metric": "stable_rank1_trace", "plot_type": "heatmap", "processes": ["validation"]}
    drawn = {}
    real = gm.plot_fold_heatmap

    def spy(**kwargs):
        drawn.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(gm, "plot_fold_heatmap", spy)
    _manager(tmp_path, monkeypatch, [job]).execute(3)

    assert drawn["rows"] == ["RMSprop", "Adam"] and drawn["colours"] == ["C0", "C1"] and drawn["folds"] == [1, 2, 3]
    assert drawn["values"][0] == [201.0, 1793.0, 2248.0]
    assert drawn["values"][1][1] == 495.0 and all(v != v for v in (drawn["values"][1][0], drawn["values"][1][2]))  # NaN
    assert drawn["notes"][1][0] == "rank\n78" and drawn["notes"][1][2] == "rank\n200"
    assert list((tmp_path / "graphs" / "jobs").rglob("acc__validation__stable_rank1_trace.svg"))


@pytest.mark.parametrize("change, message", [
    ({"source": "cross_validation"}, "source key_recovery"),
    ({"processes": ["train"]}, "processes \\[validation\\]"),
    ({"metric": ["stable_rank1_trace", "final_true_key_rank"]}, "draws one metric"),
])
def test_heatmap_refusals(change, message):
    from services.graph_manager.graph_job_validator import validate_graph_job

    job = {**_overlay_job(["mesh_chicken"], {}), "source": "key_recovery", "metric": "stable_rank1_trace",
           "plot_type": "heatmap", "processes": ["validation"], **change}
    with pytest.raises(GraphJobValidationError, match=message):
        validate_graph_job(job)


def test_a_heatmap_takes_a_matplotlib_colormap():
    """Owner 2026-10-27: Fig. 20 c on turbo (the RGB spectrum); a fold never recovered stays light grey."""
    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import NEVER_COLOUR

    job = {**_overlay_job(["mesh_chicken"], {}), "source": "key_recovery", "metric": "stable_rank1_trace",
           "plot_type": "heatmap", "processes": ["validation"]}
    assert validate_graph_job({**job, "style": {"colormap": "turbo"}})["style"]["colormap"] == "turbo"
    assert validate_graph_job(job)["style"]["colormap"] == "viridis" and NEVER_COLOUR == "0.85"
    with pytest.raises(GraphJobValidationError, match="is not a Matplotlib colormap"):
        validate_graph_job({**job, "style": {"colormap": "rgb"}})


def test_bar_values_print_each_bars_value_at_its_end(tmp_path, monkeypatch):
    """Owner 2026-10-27 (Fig. 20 b): the mean accuracy of each optimizer written at its bar."""
    import matplotlib.pyplot as plt

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import _draw_categorical_bars

    fig, ax = plt.subplots()
    _draw_categorical_bars(ax, ["a", "b"], [0.0053, 0.0049], [None, None], horizontal=True, value_format=".4f")
    assert [t.get_text() for t in ax.texts] == [" 0.0053", " 0.0049"]
    plt.close(fig)
    job = _overlay_job(["mesh_chicken"], {})
    assert validate_graph_job({**job, "style": {"bar_values": ".4f"}})["style"]["bar_values"] == ".4f"
    assert validate_graph_job(job)["style"]["bar_values"] is None
    with pytest.raises(GraphJobValidationError, match="is not a number format"):
        validate_graph_job({**job, "style": {"bar_values": "four"}})


def test_a_legend_below_sits_under_the_axes():
    """Owner 2026-10-27 (Figs. 10 c, 14 c): the N_max legend under the x label, where it covers no bar or value."""
    import matplotlib.pyplot as plt

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import _finalize_axes

    job = validate_graph_job({**_overlay_job(["mesh_chicken"], {}), "legend": {"position": "below",
                              "entries": [{"label": "N 0"}, {"label": "N 100", "hatch": "////"}]}})
    fig, ax = plt.subplots()
    ax.barh(0, 0.005)
    _finalize_axes(ax, job=job, context={}, default_title="", default_x_label="", default_y_label="")
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["N 0", "N 100"]
    box = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    assert box.y0 < 0 and not legend.get_frame_on()                      # under the axes, no frame
    plt.close(fig)


def test_legend_columns_lay_the_entries_out_in_rows():
    """Owner 2026-10-27 (Fig. 24 a): eight activations, four per row, below the axes."""
    import matplotlib.pyplot as plt

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import _finalize_axes

    job = validate_graph_job({**_overlay_job(["mesh_chicken"], {}), "legend": {"position": "below", "columns": 4}})
    assert job["legend"]["columns"] == 4 and validate_graph_job(_overlay_job(["mesh_chicken"], {}))["legend"]["columns"] == 1
    fig, ax = plt.subplots()
    for i in range(8):
        ax.plot([0, 1], [i, i], label=f"a{i}")
    _finalize_axes(ax, job=job, context={}, default_title="", default_x_label="", default_y_label="")
    assert ax.get_legend()._ncols == 4
    plt.close(fig)


def test_legend_entries_draw_lines_and_curves_take_their_series_colour_and_style(tmp_path, monkeypatch):
    """Owner 2026-10-27 (Fig. 26 a, b): a colour per epoch count, HW dashed and ID solid, named by legend lines."""
    import matplotlib.pyplot as plt
    import numpy as np

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import _finalize_axes, plot_curve_overlay

    job = validate_graph_job({**_overlay_job(["hw", "id"], {}), "metric": "mean_attack_rank_curve",
                              "style": {"series": {"hw": {"color": "C3"}, "id": {"color": "C3"}}, "line": {"styles": {"hw": "--"}}},
                              "legend": {"entries": [{"label": "HW", "color": "black", "linestyle": "--"}, {"label": "200 epochs", "color": "C3"}]}})
    curve = np.column_stack([np.arange(1, 11), np.linspace(100, 1, 10)])
    drawn = []
    real_plot = plt.Axes.plot
    monkeypatch.setattr(plt.Axes, "plot", lambda self, *a, **k: drawn.append(k) or real_plot(self, *a, **k))
    plot_curve_overlay(series={"hw": curve, "id": curve}, job=job, out_base=tmp_path / "c", metric="mean_attack_rank_curve",
                       x_label="traces", y_label="rank", order=["hw", "id"])
    lines = [k for k in drawn if k.get("label") in ("hw", "id")]
    assert [(k["color"], k.get("linestyle", "-")) for k in lines] == [("C3", "--"), ("C3", "-")]
    fig, ax = plt.subplots()
    _finalize_axes(ax, job=job, context={}, default_title="", default_x_label="", default_y_label="")
    handles = ax.get_legend().legend_handles
    assert handles[0].get_linestyle() == "--" and type(handles[1]).__name__ == "Rectangle"
    plt.close("all")
    with pytest.raises(GraphJobValidationError, match="linestyle"):
        validate_graph_job({**_overlay_job(["hw"], {}), "legend": {"entries": [{"label": "x", "linestyle": "wavy"}]}})


def test_heatmap_rows_take_their_style_series_colour(tmp_path, monkeypatch):
    """Owner 2026-10-27 (Fig. 26 c): a row label in its run's epoch colour from panel a."""
    import services.graph_manager.graph_manager as gm

    _project(tmp_path)
    _kr_folds(tmp_path, "mesh_chicken", [(201, 1)])
    _kr_folds(tmp_path, "ultimate_jewel", [(None, 7)])
    job = {**_overlay_job(["mesh_chicken", "ultimate_jewel"], {}), "source": "key_recovery", "metric": "stable_rank1_trace",
           "plot_type": "heatmap", "processes": ["validation"], "style": {"series": {"ultimate_jewel": {"color": "C5"}}}}
    drawn = {}
    monkeypatch.setattr(gm, "plot_fold_heatmap", lambda **kw: drawn.update(kw) or [tmp_path / "drawn.png"])
    _manager(tmp_path, monkeypatch, [job]).execute(3)
    assert drawn["colours"] == ["C0", "C5"]


def test_style_series_sets_a_lines_width_and_dashes():
    """Owner 2026-10-27 (Fig. 26): HW curves in longer, thicker dashes, ID solid and thin; the legend lines alike."""
    import matplotlib.pyplot as plt
    import numpy as np

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import _draw_xy, _finalize_axes

    job = validate_graph_job({**_overlay_job(["hw", "id"], {}), "style": {"series": {
        "hw": {"color": "crimson", "width": 1.3, "dashes": [6, 3]}, "id": {"color": "navy", "width": 0.7}}},
        "legend": {"entries": [{"label": "HW", "color": "crimson", "dashes": [6, 3], "width": 1.3}]}})
    fig, ax = plt.subplots()
    _draw_xy(ax, np.arange(3), np.arange(3), job=job, label="hw", series="hw", color="crimson")
    _draw_xy(ax, np.arange(3), np.arange(3), job=job, label="id", series="id", color="navy")
    hw, id_ = ax.get_lines()
    assert (hw.get_linewidth(), id_.get_linewidth()) == (1.3, 0.7) and hw.get_linestyle() != "-" and id_.get_linestyle() == "-"
    _finalize_axes(ax, job=job, context={}, default_title="", default_x_label="", default_y_label="")
    handle = ax.get_legend().legend_handles[0]
    assert handle.get_linewidth() == 1.3 and handle.get_linestyle() != "-"
    plt.close(fig)
    for bad, message in (({"width": 0}, "line width"), ({"dashes": [6]}, "pairs of lengths"), ({"dashes": [6, -3]}, "pairs of lengths")):
        with pytest.raises(GraphJobValidationError, match=message):
            validate_graph_job({**_overlay_job(["hw"], {}), "style": {"series": {"hw": bad}}})


def test_figure_margins_fix_the_plot_box_and_the_page_and_a_legend_anchor_places_the_legend(tmp_path):
    """Owner 2026-10-27: panels of one row (same height, same top and bottom margins) align in a facet; a legend
    placed inside the box at legend.anchor."""
    import re

    import matplotlib.pyplot as plt

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import _finalize_axes, save_figure

    job = validate_graph_job({**_overlay_job(["mesh_chicken"], {}), "legend": {"position": "center", "anchor": [0.66, 0.4]},
                              "style": {"figure": {"width": 6.4, "height": 4.8,
                                                   "margins": {"top": 0.42, "bottom": 1.0, "left": 0.85, "right": 0.2}}}})
    assert job["legend"]["anchor"] == [0.66, 0.4]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot([0, 1], [0, 1], label="a")
    _finalize_axes(ax, job=job, context={}, default_title="", default_x_label="", default_y_label="")
    box = ax.get_position()
    assert (round(box.x0 * 6.4, 3), round(box.y0 * 4.8, 3), round((1 - box.y1) * 4.8, 3)) == (0.85, 1.0, 0.42)
    assert tuple(ax.get_legend().get_bbox_to_anchor().transformed(ax.transAxes.inverted()).p0.round(2)) == (0.66, 0.4)
    svg = save_figure(fig, tmp_path / "f", job)[0].read_text()
    plt.close(fig)
    assert re.search(r'width="460.8pt" height="345.6pt"', svg)                 # the page, not trimmed
    for bad, message in (({"top": -1}, "length in inches"), ({"middle": 1}, "top, bottom, left, right")):
        with pytest.raises(GraphJobValidationError, match=message):
            validate_graph_job({**_overlay_job(["mesh_chicken"], {}), "style": {"figure": {"margins": bad}}})
    with pytest.raises(GraphJobValidationError, match="legend.anchor"):
        validate_graph_job({**_overlay_job(["mesh_chicken"], {}), "legend": {"anchor": [1]}})


def test_a_heatmap_without_its_legend_when_the_row_shares_another_panels(tmp_path, monkeypatch):
    """Owner 2026-10-27: IV.c and V.c share VI.c's grey-cell legend (legend.enabled false leaves it out)."""
    import services.graph_manager.plotters as plotters

    real = plotters._finalize_axes
    for enabled, expected in ((False, None), (True, ["key never stays at rank 1: the fold's final key rank"])):
        seen = []

        def spy(ax, **kw):
            real(ax, **kw)
            seen.append(ax)

        monkeypatch.setattr(plotters, "_finalize_axes", spy)
        job = {**_overlay_job(["m"], {}), "legend": {"enabled": enabled}}
        plotters.plot_fold_heatmap(values=[[float("nan"), 5.0]], notes=[["rank\n7", ""]], rows=["m"], colours=["C0"],
                                   folds=[1, 2], job=job, out_base=tmp_path / f"h{enabled}", metric="stable_rank1_trace")
        legend = seen[0].get_legend()
        assert (None if legend is None else [t.get_text() for t in legend.get_texts()]) == expected


def test_a_legend_wider_than_its_box_is_made_again_at_a_size_that_fits():
    """Owner 2026-10-27 (Fig. 24 a): eight activations in four columns at the top of the box, inside it; 2026-10-28:
    shorter marks and narrower gaps first, the text as large as fits."""
    import matplotlib.pyplot as plt

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import _finalize_axes

    job = validate_graph_job({**_overlay_job(["mesh_chicken"], {}), "legend": {"position": "top", "columns": 4}})
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for i in range(8):
        ax.plot([0, 1], [i, i], label=f"softsign {i}")
    full = plt.rcParams["legend.fontsize"]
    _finalize_axes(ax, job=job, context={}, default_title="", default_x_label="", default_y_label="")
    renderer = fig.canvas.get_renderer()
    legend = ax.get_legend()
    assert legend._loc == 9 and legend._ncols == 4                                # "top": upper center
    assert legend.get_window_extent(renderer).width <= ax.get_window_extent(renderer).width
    size = legend.get_texts()[0].get_fontsize()
    full = plt.matplotlib.font_manager.FontProperties(size=full).get_size_in_points()
    assert len(legend.get_texts()) == 8 and 6.0 <= size <= full
    assert legend.handlelength == 1.2                                           # shorter marks before smaller text
    plt.close(fig)


def test_heatmap_grey_cells_print_the_rank_alone_and_no_coloured_cell_means_no_colour_bar(tmp_path, monkeypatch):
    """Owner 2026-10-27 (Fig. 5.ii): one look in all six heat maps; Fig. 10's heat map, where no fold recovers the
    key, has no colour bar."""
    import services.graph_manager.plotters as plotters

    real = plotters._finalize_axes
    for values, bars in (([[float("nan")] + [4999.0] * 9] * 3, 1), ([[float("nan")] * 10] * 3, 0)):
        seen = []

        def spy(ax, **kw):
            real(ax, **kw)
            seen.append(ax)

        monkeypatch.setattr(plotters, "_finalize_axes", spy)
        job = {**_overlay_job(["m"], {}), "style": {"font": {"tick_size": 20}}}
        notes = [["rank\n7"] * 10] * 3
        plotters.plot_fold_heatmap(values=values, notes=notes, rows=["a", "b", "c"], colours=["C0"] * 3,
                                   folds=list(range(1, 11)), job=job, out_base=tmp_path / f"h{bars}",
                                   metric="stable_rank1_trace")
        ax = seen[0]
        assert ax.texts[0].get_text() == "7" and (bars == 0 or ax.texts[1].get_text() == "4,999")
        assert len({t.get_fontsize() for t in ax.texts}) == 1                    # one size for every cell
        assert sum(a.get_label() == "<colorbar>" for a in ax.figure.axes) == bars


def test_a_colorbar_label_text_rule_sets_or_removes_the_label(tmp_path, monkeypatch):
    """Owner 2026-10-27 (Fig. 5.ii): text_rules target colorbar_label; "" leaves the label out, the title naming it."""
    from services.graph_manager.graph_job_validator import validate_graph_job
    import services.graph_manager.plotters as plotters

    real = plotters._finalize_axes
    seen = []
    monkeypatch.setattr(plotters, "_finalize_axes", lambda ax, **kw: real(ax, **kw) or seen.append(ax))
    for text in ("", "traces"):
        job = {**_overlay_job(["m"], {}), "style": {"text_rules": [{"target": "colorbar_label", "text": text}]}}
        assert validate_graph_job(job)["style"]["text_rules"][0]["target"] == "colorbar_label"
        plotters.plot_fold_heatmap(values=[[5.0, 6.0]], notes=[["", ""]], rows=["m"], colours=["C0"], folds=[1, 2],
                                   job=job, out_base=tmp_path / f"c{text}", metric="stable_rank1_trace")
        bar = [a for a in seen[-1].figure.axes if a.get_label() == "<colorbar>"][0]
        assert bar.yaxis.label.get_text() == text


def test_an_x_step_keeps_its_ticks_on_an_epoch_axis():
    """Owner 2026-10-27 (Fig. 26 b): a tick every 200 epochs, not Matplotlib's integer epoch ticks."""
    import matplotlib.pyplot as plt

    from services.graph_manager.plotters import _integer_epoch_ticks

    fig, ax = plt.subplots()
    ax.plot([0, 800], [0, 1])
    ax.set_xticks([0, 200, 400, 600, 800])
    _integer_epoch_ticks(ax, "epoch", {"style": {"axes": {"x_step": 200}}})
    assert list(ax.get_xticks()) == [0, 200, 400, 600, 800]
    _integer_epoch_ticks(ax, "epoch", {"style": {"axes": {}}})
    assert type(ax.xaxis.get_major_locator()).__name__ == "MaxNLocator"
    plt.close(fig)


def test_a_heatmaps_legend_sits_just_under_its_x_axis_and_a_long_colour_bar_label_breaks_in_two(tmp_path, monkeypatch):
    """Owner 2026-10-27 (Fig. 26 c): no space wasted between stacked rows; the colour-bar label no longer runs past
    the title."""
    import services.graph_manager.plotters as plotters

    real = plotters._finalize_axes
    seen = []

    def spy(ax, **kw):
        real(ax, **kw)
        seen.append(ax)

    monkeypatch.setattr(plotters, "_finalize_axes", spy)
    job = {**_overlay_job(["m"], {}), "style": {"figure": {"width": 6.0, "height": 2.5,
                                                          "margins": {"top": 0.3, "bottom": 1.0, "left": 1.0, "right": 1.5}}}}
    plotters.plot_fold_heatmap(values=[[float("nan"), 5.0]], notes=[["rank\n7", ""]], rows=["m"], colours=["C0"],
                               folds=[1, 2], job=job, out_base=tmp_path / "h", metric="stable_rank1_trace")
    ax = seen[0]
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    gap = (ax.xaxis.get_tightbbox(renderer).y0 - ax.get_legend().get_window_extent(renderer).y1) / fig.dpi
    assert 0.0 < gap < 0.2                                                        # inches under the fold label
    bar = [a for a in fig.axes if a.get_label() == "<colorbar>"][0]
    lines = bar.yaxis.label.get_text().split("\n")
    assert len(lines) > 1 and " ".join(lines) == "traces until the key stays at rank 1"   # words spread over lines
    assert bar.yaxis.label.get_window_extent(renderer).height <= bar.get_window_extent(renderer).height


def test_style_heatmap_shares_a_scale_drops_the_colour_bar_formats_cells_and_groups_rows(tmp_path, monkeypatch):
    """Owner 2026-10-28 (Fig. 5.ii): one scale for six heat maps, no colour bar of their own, 4999 not 4,999, VI's rows
    HW and ID over the epochs, the row names bold."""
    from services.graph_manager.graph_job_validator import validate_graph_job
    import services.graph_manager.plotters as plotters

    base = _overlay_job(["m"], {})
    job = validate_graph_job({**base, "source": "key_recovery", "metric": "stable_rank1_trace", "plot_type": "heatmap",
                              "processes": ["validation"],
                              "style": {"heatmap": {"limits": [0, 5000], "colorbar": False, "cell_values": ".0f",
                                                    "row_groups": True},
                                        "text_rules": [{"target": "y_label", "text": "", "bold": True}]}})
    assert job["style"]["heatmap"] == {"limits": [0.0, 5000.0], "colorbar": False, "cell_values": ".0f", "row_groups": True}
    assert validate_graph_job(base)["style"]["heatmap"] == {"limits": None, "colorbar": True, "cell_values": ",.0f",
                                                            "row_groups": False}
    for bad, message in (({"limits": [5, 1]}, "low < high"), ({"cell_values": "x"}, "number format"), ({"size": 1}, "limits")):
        with pytest.raises(GraphJobValidationError, match=message):
            validate_graph_job({**base, "style": {"heatmap": bad}})
    real = plotters._finalize_axes
    seen = []
    monkeypatch.setattr(plotters, "_finalize_axes", lambda ax, **kw: real(ax, **kw) or seen.append(ax))
    plotters.plot_fold_heatmap(values=[[4999.0, 10.0], [20.0, 30.0], [40.0, 50.0]], notes=[["", ""]] * 3,
                               rows=["HW, 20", "HW, 50", "ID, 20"], colours=["C0", "C1", "C0"], folds=[1, 2], job=job,
                               out_base=tmp_path / "h", metric="stable_rank1_trace")
    ax = seen[0]
    image = ax.get_images()[0]
    assert image.get_clim() == (0.0, 5000.0)                                    # the shared scale
    assert not any(a.get_label() == "<colorbar>" for a in ax.figure.axes)
    assert ax.texts[0].get_text() == "4999"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["20", "50", "20"]
    assert all(t.get_fontweight() == "bold" for t in ax.get_yticklabels())
    assert {"HW", "ID"} <= {t.get_text() for t in ax.texts}                     # the groups beside their rows


def test_a_legend_bottom_sits_inside_the_box_at_its_bottom():
    """Owner 2026-10-28 (Fig. 4 II): under the curves, which run high."""
    import matplotlib.pyplot as plt

    from services.graph_manager.graph_job_validator import validate_graph_job
    from services.graph_manager.plotters import _finalize_axes

    job = validate_graph_job({**_overlay_job(["mesh_chicken"], {}), "legend": {"position": "bottom", "columns": 3}})
    fig, ax = plt.subplots()
    for i in range(5):
        ax.plot([0, 1], [i, i], label=f"o{i}")
    _finalize_axes(ax, job=job, context={}, default_title="", default_x_label="", default_y_label="")
    assert ax.get_legend()._loc == 8                                            # lower center
    plt.close(fig)


def test_a_row_groups_name_takes_its_rows_colour_when_they_share_one(tmp_path, monkeypatch):
    """Owner 2026-10-28 (Fig. 5.ii): I to III's model names in the model's colour; VI's HW and ID black (a colour per
    epoch)."""
    import matplotlib.colors as mcolors

    import services.graph_manager.plotters as plotters

    real = plotters._finalize_axes
    seen = []
    monkeypatch.setattr(plotters, "_finalize_axes", lambda ax, **kw: real(ax, **kw) or seen.append(ax))
    job = {**_overlay_job(["m"], {}), "style": {"heatmap": {"row_groups": True}}}
    plotters.plot_fold_heatmap(values=[[1.0, 2.0]] * 4, notes=[["", ""]] * 4,
                               rows=["ReLU, 0", "ReLU, 100", "HW, 20", "HW, 50"], colours=["C0", "C0", "C1", "C2"],
                               folds=[1, 2], job=job, out_base=tmp_path / "h", metric="stable_rank1_trace")
    names = {t.get_text(): t.get_color() for t in seen[0].texts if t.get_text() in ("ReLU", "HW")}
    assert mcolors.same_color(names["ReLU"], "C0") and mcolors.same_color(names["HW"], "black")
