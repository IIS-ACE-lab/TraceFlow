# services/side_channel_core/tests/unit/test_graph_export.py
"""Finished runs into graph jobs (services/graph_manager/graph_export.py), on a small project."""

import json

import pytest
import yaml

from services.graph_manager import graph_export as gx

CONFIG = """\
# my graph config: these comments must survive
jobs:

  # ================================================================
  # JOB 0: default
  # ================================================================
  - id: 0
    name: "default"
    type: "model_overlay"
    mode: "auto"
    enabled: true
    source: "attack_metrics"
    metric: "mean_attack_rank_curve"
    plot_type: "line"
    top_models:
      - "adam_run"
    output_file_name: "default"

  # ================================================================
  # JOB 1: hand-written rank figure
  # Run: make graph ENV=dev PROJECT=P GRAPH_ID=1
  # ================================================================
  - id: 1
    name: "rank"
    type: "model_overlay"
    mode: "auto"
    enabled: true
    source: "cross_validation"
    metric: "mean_attack_rank_curve"
    plot_type: "line"
    models:
      - adam_run      # Adam, written by hand
    processes:
      - "validation"
    output_file_name: "rank"
    legend:
      mode: "custom"
      labels:
        adam_run: "Adam"
    style:
      text_rules:
        - target: "title"
          text: "Desynchronization Nmax = 0"

  # ================================================================
  # JOB 2: another figure
  # Run: make graph ENV=dev PROJECT=P GRAPH_ID=2
  # ================================================================
  - id: 2
    name: "accuracy"
    type: "model_overlay"
    mode: "auto"
    enabled: true
    source: "cross_validation"
    metric: "mean_accuracy"
    plot_type: "line"
    models:
      - adam_run
    processes:
      - "validation"
    output_file_name: "accuracy"

# ======================================================================
# FACETING
# ======================================================================
faceting:
  - id: 1
    panels:
      - file_id: 1
        address: "artifacts/dev/generated_models/adam_run/graphs/job_002_accuracy/model_overlay/a.pdf"
"""

RUNS = {"adam_run": "Adam", "sgd_run": "SGD", "rms_run": "RMSprop"}


class _Project(gx.Project):
    def __init__(self, root):            # noqa: D107 - no project config: the paths are given
        self.name, self.env, self.root = "P", "dev", root
        self.graph_config = root / "config" / "graph_config.yaml"
        self.graphs = root / "graphs"

    def registry(self):
        return {f"h{i}": {"model_name": n, "batches": ["fig10"], "status": "DONE"} for i, n in enumerate(RUNS)}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "graph_config.yaml").write_text(CONFIG)
    for name, optimizer in RUNS.items():
        run = tmp_path / "artifacts" / "dev" / "generated_models" / name
        (run / "config").mkdir(parents=True)
        args = {"Adam": {"beta_1": 0.9}, "SGD": {"momentum": 0.0}, "RMSprop": {"rho": 0.9}}[optimizer]
        snapshot = {"model_id": "CNN", "data": {"name": "ascad"}, "model_name": name,
                    "trainer": {"epochs": 100, "batch_size": 200, "optimizer": {"type": optimizer, "learning_rate": 1e-5}},
                    "sampled": {"trainer": {"optimizer": {"type": optimizer, "args": args}}}}
        (run / "config" / f"{name}_params.yaml").write_text(yaml.safe_dump(snapshot))
        fold = run / "metrics" / "cross_validation" / "fold_01" / "arrays" / "validation" / "mean_attack_rank_curve"
        fold.mkdir(parents=True)
        (fold / "ep0100.npy").write_bytes(b"x")
    return _Project(tmp_path)


def _data(project):
    return yaml.safe_load(project.graph_config.read_text())


def test_what_differs_is_the_optimizer_type_not_the_arguments_that_follow_it(project):
    varying, values = gx.differences({n: project.snapshot(n) for n in RUNS})
    assert varying == ["trainer.optimizer.type"]
    assert {n: gx.label(v) for n, v in values.items()} == RUNS
    assert gx.shared_line(project.snapshot("adam_run"), varying, ["fig10"]) == \
        "CNN on ascad, 100 epochs, batch 200, learning rate 1e-05 (batch fig10)"


def test_a_new_figure_is_a_job_written_as_by_hand_and_nothing_else_changes(project, monkeypatch):
    monkeypatch.setattr(gx, "choices", lambda p, models: {m: [{"source": "cross_validation", "process": "validation",
                                                                "metric": "mean_fit_time_s", "kind": "per_epoch"}] for m in models})
    before = project.graph_config.read_text()
    out = gx.export(project, list(RUNS), source="cross_validation", metric="mean_fit_time_s")
    assert out["created"] == 3
    text = project.graph_config.read_text()
    assert text.startswith(before[:before.index("# ======================================================================\n# FACETING")])
    assert "  # JOB 3: fig10: mean_fit_time_s (cross_validation, validation), dataset ascad" in text
    assert "GRAPH_ID=3" in text and "      - sgd_run  # SGD" in text
    assert "# CNN on ascad, 100 epochs, batch 200, learning rate 1e-05 (batch fig10)" in text
    job = _data(project)["jobs"][-1]
    assert job["models"] == list(RUNS) and job["legend"]["labels"] == RUNS and job["processes"] == ["validation"]
    assert "top_models" not in job
    assert _data(project)["faceting"] == yaml.safe_load(before)["faceting"]


def test_runs_of_an_existing_figure_join_its_job_with_their_comment_and_label(project, monkeypatch):
    monkeypatch.setattr(gx, "choices", lambda p, models: {m: [{"source": "cross_validation", "process": "validation",
                                                                "metric": "mean_attack_rank_curve", "kind": "curve"}] for m in models})
    out = gx.export(project, ["adam_run", "sgd_run"], source="cross_validation", metric="mean_attack_rank_curve")
    assert out == {"ok": True, "updated": 1, "added": ["sgd_run"], "labels": {"adam_run": "Adam", "sgd_run": "SGD"}}
    text = project.graph_config.read_text()
    assert "      - adam_run      # Adam, written by hand" in text and "- sgd_run  # SGD" in text
    assert _data(project)["jobs"][1]["legend"]["labels"] == {"adam_run": "Adam", "sgd_run": "SGD"}
    with pytest.raises(gx.ExportError, match="is not recorded by"):
        gx.export(project, ["adam_run"], source="cross_validation", metric="mean_accuracy")


def test_a_deleted_run_leaves_every_job(project):
    assert gx.prune(project, ["adam_run"]) == {"ok": True, "jobs": [0, 1, 2]}
    jobs = _data(project)["jobs"]
    assert jobs[0]["top_models"] == [] and jobs[1]["models"] == [] and jobs[1]["legend"]["labels"] == {}
    assert "# my graph config: these comments must survive" in project.graph_config.read_text()


def test_deleting_a_job_moves_the_later_ones_down_with_their_folders_facets_and_headers(project):
    # jobs 1 and 2 list one run (adam_run): their figures are in its graphs folder (job_paths.py)
    figures = project.root / "artifacts" / "dev" / "generated_models" / "adam_run" / "graphs"
    (figures / "job_001_rank").mkdir(parents=True)
    (figures / "job_002_accuracy" / "model_overlay").mkdir(parents=True)
    out = gx.delete_job(project, 1)
    assert out["moved"] == {"2": 1} and out["figures_moved_aside"][0].endswith("job_001_rank_" + gx.date.today().isoformat())
    assert sorted(p.name for p in figures.iterdir()) == ["job_001_accuracy"]
    assert [p.name for p in (project.graphs / "jobs" / "_deleted").iterdir()] == [f"job_001_rank_{gx.date.today().isoformat()}"]
    text = project.graph_config.read_text()
    assert "JOB 1: hand-written" not in text and "  # JOB 1: another figure" in text and "GRAPH_ID=1\n" in text
    assert 'address: "artifacts/dev/generated_models/adam_run/graphs/job_001_accuracy/model_overlay/a.pdf"' in text
    data = _data(project)
    assert [j["id"] for j in data["jobs"]] == [0, 1] and data["jobs"][1]["name"] == "accuracy"
    with pytest.raises(gx.ExportError, match="never deleted"):
        gx.delete_job(project, 0)


def test_renumber_closes_every_gap(project):
    project.graph_config.write_text(project.graph_config.read_text().replace("  - id: 2\n", "  - id: 7\n").replace("GRAPH_ID=2", "GRAPH_ID=7"))
    out = gx.renumber(project)
    assert out["moved"] == {"7": 2} and [j["id"] for j in _data(project)["jobs"]] == [0, 1, 2]
    assert json.loads(json.dumps(gx.renumber(project))) == {"ok": True, "moved": {}}


def test_a_job_removed_by_hand_closes_its_gap_when_asked(project):
    figures = project.root / "artifacts" / "dev" / "generated_models" / "adam_run" / "graphs"     # one-run jobs
    (figures / "job_001_rank").mkdir(parents=True)
    (figures / "job_002_accuracy").mkdir(parents=True)
    text = project.graph_config.read_text()
    start = text.index("  # ================================================================\n  # JOB 1:")
    end = text.index("  # ================================================================\n  # JOB 2:")
    project.graph_config.write_text(text[:start] + text[end:])          # deleted in the editor, header and all
    with pytest.raises(gx.ExportError, match="still in graph_config.yaml"):
        gx.close_gap(project, 2)
    out = gx.close_gap(project, 1)
    assert out["moved"] == {"2": 1} and len(out["figures_moved_aside"]) == 1      # found in the run's folder
    assert sorted(p.name for p in figures.iterdir()) == ["job_001_accuracy"]
    assert [j["id"] for j in _data(project)["jobs"]] == [0, 1]


def test_a_run_exported_alone_is_labelled_by_what_differs_in_its_batch(project, monkeypatch):
    monkeypatch.setattr(gx, "choices", lambda p, models: {m: [{"source": "cross_validation", "process": "validation",
                                                                "metric": "mean_fit_time_s", "kind": "per_epoch"}] for m in models})
    out = gx.export(project, ["sgd_run"], source="cross_validation", metric="mean_fit_time_s")
    assert out["labels"] == {"sgd_run": "SGD"} and out["differ"] == ["trainer.optimizer.type"]


def _draw(project, *folders):
    for folder in folders:
        (project.graphs / "jobs" / folder).mkdir(parents=True, exist_ok=True)
        (project.graphs / "jobs" / folder / "f.pdf").write_bytes(b"%PDF")


def test_drawn_figures_become_a_facet_written_with_its_header(project):
    _draw(project, "job_001_rank", "job_002_accuracy")
    with pytest.raises(gx.ExportError, match="not drawn yet"):
        gx.export_facet(project, ["jobs/job_001_rank/f.pdf", "jobs/job_009_none/f.pdf"])
    out = gx.export_facet(project, ["jobs/job_001_rank/f.pdf", "jobs/job_002_accuracy/f.pdf"])
    assert out["created"] == 2 and out["panels"] == 2
    facet = _data(project)["faceting"][-1]
    assert [p["address"] for p in facet["panels"]] == ["jobs/job_001_rank/f.pdf", "jobs/job_002_accuracy/f.pdf"]
    assert facet["arrangement"] == {"mode": "horizontal", "order": [1, 2]}
    text = project.graph_config.read_text()
    assert "  # FACET 2: rank | accuracy" in text and "#   panel 1: job 1 (rank)" in text and "FACE_ID=2 graph" in text
    from services.graph_manager.faceting import validate_facet
    validate_facet(facet)


def test_a_deleted_job_leaves_its_facets_and_a_facet_too_small_goes(project):
    _draw(project, "job_001_rank", "job_002_accuracy")
    gx.export_facet(project, ["jobs/job_001_rank/f.pdf", "jobs/job_002_accuracy/f.pdf"])     # facet 2
    out = gx.delete_job(project, 2)
    assert out["facets_removed"] == [1, 2]          # facet 1 had one panel of job 2 left; facet 2 one of job 1
    assert _data(project)["faceting"] == [] or _data(project)["faceting"] is None


def test_a_facet_keeping_two_panels_loses_only_the_deleted_jobs(project):
    _draw(project, "job_001_rank", "job_002_accuracy")
    project.graph_config.write_text(project.graph_config.read_text().replace(
        '''        address: "artifacts/dev/generated_models/adam_run/graphs/job_002_accuracy/model_overlay/a.pdf"
''', '''        address: "artifacts/dev/generated_models/adam_run/graphs/job_002_accuracy/model_overlay/a.pdf"
      - file_id: 2
        address: "jobs/job_001_rank/f.pdf"
      - file_id: 3
        address: "jobs/job_001_rank/g.pdf"
    arrangement:
      mode: "grid"
      order: [[1, 2], [3]]
'''))
    out = gx.delete_job(project, 2)
    assert out["facets_removed"] == [] and out["facet_panels_removed"] == 1
    facet = _data(project)["faceting"][0]
    assert [p["file_id"] for p in facet["panels"]] == [2, 3] and facet["arrangement"]["order"] == [[2], [3]]


def test_facets_after_a_removed_one_move_down(project):
    _draw(project, "job_001_rank", "job_002_accuracy")
    gx.export_facet(project, ["jobs/job_001_rank/f.pdf", "jobs/job_002_accuracy/f.pdf"])     # facet 2
    (project.graphs / "facets" / "facet_002_job_1_and_job_2").mkdir(parents=True)
    assert gx.delete_facet(project, 1) == {"ok": True, "deleted": 1}
    assert [f["id"] for f in _data(project)["faceting"]] == [1]
    text = project.graph_config.read_text()
    assert "# FACET 1: rank | accuracy" in text and "FACE_ID=1 graph" in text
    assert (project.graphs / "facets" / "facet_001_job_1_and_job_2").is_dir()


def test_delete_from_graph_config_removes_the_run_and_a_job_left_empty(project):
    out = gx.remove_from_jobs(project, "adam_run", source="cross_validation", metric="mean_accuracy")
    assert out["deleted"] == [2] and out["updated"] == []
    assert [j["id"] for j in _data(project)["jobs"]] == [0, 1]                      # job 2 had only adam_run
    with pytest.raises(gx.ExportError, match="in no graph job drawing"):
        gx.remove_from_jobs(project, "sgd_run", source="cross_validation", metric="mean_attack_rank_curve")


def test_an_exported_facet_writes_its_order_on_one_line(project):
    _draw(project, "job_001_rank", "job_002_accuracy")
    gx.export_facet(project, ["jobs/job_001_rank/f.pdf", "jobs/job_002_accuracy/f.pdf"])
    assert "      order: [1, 2]   # the panels' file_ids in place order" in project.graph_config.read_text()


def test_a_job_of_one_run_draws_in_its_run_folder_and_moves_when_a_second_run_joins(project, monkeypatch):
    from services.graph_manager import job_paths

    run_dir = project.root / "artifacts" / "dev" / "generated_models" / "adam_run" / "graphs" / "job_001_rank"
    job = _data(project)["jobs"][1]
    assert gx._job_dir(project, job) == run_dir                                    # one run: its own folder
    assert gx._job_dir(project, {**job, "models": ["adam_run", "sgd_run"]}) == project.graphs / "jobs" / "job_001_rank"
    assert job_paths.job_dir({"id": 0, "name": "d", "top_models": []}, project_root=project.root, graphs_root=project.graphs,
                             env="dev") == project.graphs / "jobs" / "job_000_d"     # job 0 with no run: the project
    (run_dir / "model_overlay").mkdir(parents=True)
    (run_dir / "model_overlay" / "a.pdf").write_bytes(b"%PDF")
    facet_before = _data(project)["faceting"]
    monkeypatch.setattr(gx, "choices", lambda p, models: {m: [{"source": "cross_validation", "process": "validation",
                                                                "metric": "mean_attack_rank_curve", "kind": "curve"}] for m in models})
    out = gx.export(project, ["sgd_run"], source="cross_validation", metric="mean_attack_rank_curve")
    moved = project.graphs / "jobs" / "job_001_rank"
    assert out["updated"] == 1 and out["relocated"] == [[str(run_dir), str(moved)]]
    assert (moved / "model_overlay" / "a.pdf").is_file() and not run_dir.exists()
    assert _data(project)["faceting"] == facet_before                              # the facet into job 2: untouched
    gx.remove_from_jobs(project, "sgd_run", source="cross_validation", metric="mean_attack_rank_curve")   # one run again
    assert (run_dir / "model_overlay" / "a.pdf").is_file() and not moved.exists()


def test_a_figure_folder_moved_by_traceflow_keeps_its_seal_and_a_broken_one_stays_broken(tmp_path, monkeypatch):
    from services.side_channel_core.utils import seal
    from services.side_channel_core.utils.atomic_io import atomic_write_text

    monkeypatch.setenv("TRACEFLOW_SEAL_DIR", str(tmp_path / "keys"))
    seal._signer_cache.clear()
    seal._verified.clear()
    src = tmp_path / "P" / "graphs" / "jobs" / "job_003_a"
    src.mkdir(parents=True)
    atomic_write_text(src / "sources.json", json.dumps({"job": 3}) + "\n")            # sealed at its path
    atomic_write_text(src / "broken.json", json.dumps({"x": 1}) + "\n")
    (src / "broken.json").write_text((src / "broken.json").read_text().replace('"x": 1', '"x": 2'))
    assert seal.verify(src / "sources.json").state == "ok" and seal.verify(src / "broken.json").state == "tampered"
    dst = tmp_path / "P" / "graphs" / "jobs" / "job_002_a"
    gx._move_folder(src, dst)
    assert seal.verify(dst / "sources.json").state == "ok"                           # signed again at its new path
    assert json.loads((dst / "sources.json").read_text())["job"] == 3                  # content kept
    assert seal.verify(dst / "broken.json").state == "tampered"                        # never re-signed
    seal._signer_cache.clear()
    seal._verified.clear()


def test_a_new_job_meeting_a_deleted_jobs_folder_takes_the_next_name_and_the_folder_keeps_its_own(project, monkeypatch):
    monkeypatch.setattr(gx, "choices", lambda p, models: {m: [{"source": "cross_validation", "process": "validation",
                                                                "metric": "mean_fit_time_s", "kind": "per_epoch"}] for m in models})
    name = gx._job_name(["fig10"], "ascad", "mean_fit_time_s")
    for folder in (f"job_003_{name}", f"job_003_{name}_1"):                   # left by deleted jobs 3
        (project.graphs / "jobs" / folder).mkdir(parents=True)
        (project.graphs / "jobs" / folder / "old_accuracy.pdf").write_bytes(b"%PDF")
    out = gx.export(project, list(RUNS), source="cross_validation", metric="mean_fit_time_s")
    job = _data(project)["jobs"][-1]
    assert out["created"] == 3 and out["name"] == f"{name}_2"
    assert (job["name"], job["output_file_name"]) == (f"{name}_2", f"{name}_2")
    assert gx._job_dir(project, job) == project.graphs / "jobs" / f"job_003_{name}_2"
    assert all((project.graphs / "jobs" / f / "old_accuracy.pdf").is_file() for f in (f"job_003_{name}", f"job_003_{name}_1"))


def test_a_job_moving_down_onto_a_leftover_folder_takes_the_next_name_and_a_second_deletion_is_indexed(project):
    figures = project.root / "artifacts" / "dev" / "generated_models" / "adam_run" / "graphs"     # one-run jobs
    (figures / "job_001_rank").mkdir(parents=True)
    (figures / "job_002_accuracy" / "model_overlay").mkdir(parents=True)
    (figures / "job_001_accuracy").mkdir(parents=True)                       # left by a job 1 deleted by hand
    (figures / "job_001_accuracy" / "old_accuracy.pdf").write_bytes(b"%PDF")
    aside = project.graphs / "jobs" / "_deleted" / f"job_001_rank_{gx.date.today().isoformat()}"
    aside.mkdir(parents=True)                                                  # job 1 rank was set aside once today
    out = gx.delete_job(project, 1)
    assert out["figures_moved_aside"] == [str(aside) + "_1"] and aside.is_dir()
    assert out["folders"] == [["job_002_accuracy", "job_001_accuracy_1"]]
    assert sorted(p.name for p in figures.iterdir()) == ["job_001_accuracy", "job_001_accuracy_1"]
    assert (figures / "job_001_accuracy" / "old_accuracy.pdf").is_file()
    job = _data(project)["jobs"][1]
    assert (job["id"], job["name"], job["output_file_name"]) == (1, "accuracy_1", "accuracy_1")
    assert 'address: "artifacts/dev/generated_models/adam_run/graphs/job_001_accuracy_1/model_overlay/a.pdf"' in \
        project.graph_config.read_text()


def test_indexed_names_count_like_a_file_manager():
    from services.graph_manager import job_paths

    taken = {"acc", "acc_1", "acc_3"}
    assert [job_paths.indexed(n, taken.__contains__) for n in ("rank", "acc")] == ["rank", "acc_2"]


def _mlp(out_features):
    return [{"model_id": "MLP", "architecture_args": {"layers": [{"Flatten": {}}, {"Linear": {"out_features": out_features}}]}}]


def test_the_axes_of_a_study_leave_out_what_follows_from_an_axis_and_a_key_some_runs_never_set():
    runs = {}
    for leakage, out in (("ID", 256), ("HW", 9)):
        for epochs in (20, 50):
            runs[f"{leakage}{epochs}"] = {"model_id": "MLP", "data": {"name": "ascad"}, "sampled": {
                "model": _mlp(out), "sca": {"leakage_model": leakage, "variant": "fixed_key" if (leakage, epochs) == ("ID", 20) else None},
                "trainer": {"epochs": epochs}}}
    axes, values = gx.differences(runs)
    assert axes == ["sca.leakage_model", "trainer.epochs"]        # out_features follows; variant unset elsewhere
    assert gx.label(values["HW50"]) == "leakage_model=HW, epochs=50"


def _optimizer_run(optimizer, learning_rate):
    args = {"Adam": {"beta_1": 0.9}, "SGD": {"momentum": 0.0}, "RMSprop": {"rho": 0.9}}[optimizer]
    block = {"type": optimizer, "learning_rate": learning_rate, "args": args}
    return {"model_id": "CNN", "data": {"name": "ascad"},
            "trainer": {"epochs": 100, "batch_size": 200, "optimizer": dict(block)},
            "sampled": {"trainer": {"optimizer": block}}}


def test_a_learning_rate_differing_between_runs_of_one_optimizer_stays_an_axis_beside_the_type():
    runs = {"adam": _optimizer_run("Adam", 1e-5), "sgd": _optimizer_run("SGD", 1e-5),
            "rms5": _optimizer_run("RMSprop", 1e-5), "rms4": _optimizer_run("RMSprop", 1e-4)}
    axes, values = gx.differences(runs)
    assert axes == ["trainer.optimizer.learning_rate", "trainer.optimizer.type"]   # rho, beta_1, momentum follow the type
    labels = {n: gx.label(v) for n, v in values.items()}
    assert labels["rms5"] == "learning_rate=1e-05, type=RMSprop" and labels["rms4"] == "learning_rate=0.0001, type=RMSprop"
    assert len(set(labels.values())) == len(runs)
    assert gx.shared_line(runs["rms5"], [*axes, *gx.differing(runs)], ["fig10"]) == \
        "CNN on ascad, 100 epochs, batch 200 (batch fig10)"


def test_a_learning_rate_changing_only_with_the_optimizer_is_no_axis_and_not_in_the_header():
    runs = {"adam": _optimizer_run("Adam", 1e-3), "sgd": _optimizer_run("SGD", 1e-2), "rms": _optimizer_run("RMSprop", 1e-5)}
    axes, _ = gx.differences(runs)
    assert axes == ["trainer.optimizer.type"]
    assert gx.shared_line(runs["rms"], [*axes, *gx.differing(runs)], ["fig10"]) == \
        "CNN on ascad, 100 epochs, batch 200 (batch fig10)"


def test_export_refuses_runs_that_would_share_one_legend_label(project, monkeypatch):
    runs = project.root / "artifacts" / "dev" / "generated_models"
    twin = yaml.safe_load((runs / "adam_run" / "config" / "adam_run_params.yaml").read_text())
    twin["model_name"] = "sgd_run"
    (runs / "sgd_run" / "config" / "sgd_run_params.yaml").write_text(yaml.safe_dump(twin))
    monkeypatch.setattr(gx, "choices", lambda p, models: {m: [{"source": "cross_validation", "process": "validation",
                                                                "metric": "mean_fit_time_s", "kind": "per_epoch"}] for m in models})
    before = project.graph_config.read_text()
    with pytest.raises(gx.ExportError, match="adam_run, sgd_run would share one legend label"):
        gx.export(project, ["adam_run", "sgd_run"], source="cross_validation", metric="mean_fit_time_s")
    assert project.graph_config.read_text() == before


def _queues(project, queues, status=None):
    (project.root / "config" / "experiment_queue.yaml").write_text(yaml.safe_dump(queues))
    if status is not None:
        project.registry = lambda: {f"h{i}": {"model_name": n, "batches": ["fig10"], "status": status.get(n, "DONE")}
                                    for i, n in enumerate(RUNS)}


def _rank_choices(monkeypatch):
    monkeypatch.setattr(gx, "choices", lambda p, models: {m: [{"source": "cross_validation", "process": "validation",
                                                                "metric": "mean_attack_rank_curve", "kind": "curve"}] for m in models})


def test_a_study_is_named_by_its_queues_and_exists_only_once_every_queue_of_it_is_done(project, monkeypatch):
    queues = ["fig10", "fig10_optimizers", "fig14_cnnbest", "fig14_vgg16"]
    assert [gx.study_name(q, queues) for q in queues] == ["fig10", "fig10_optimizers", "fig14", "fig14"]
    _rank_choices(monkeypatch)
    _queues(project, {"fig10": ["make orchestrate"]}, status={"sgd_run": "RUNNING"})
    with pytest.raises(gx.ExportError, match="not DONE"):
        gx.study_plan(project, "fig10")
    _queues(project, {"fig10": ["make orchestrate"]}, status={})
    plan = gx.study_plan(project, "fig10")
    assert plan["runs"] == list(RUNS) and plan["legend"] == ["trainer.optimizer.type"]
    assert plan["axes"] == [{"key": "trainer.optimizer.type", "values": ["Adam", "RMSprop", "SGD"]}]


def test_a_study_writes_one_job_per_figure_and_metric_reading_the_last_epoch_and_never_doubles_it(project, monkeypatch):
    _rank_choices(monkeypatch)
    _queues(project, {"fig10": ["make orchestrate"]})
    metric = [("cross_validation", "validation", "mean_attack_rank_curve")]
    out = gx.study(project, "fig10", legend=["trainer.optimizer.type"], metrics=metric)
    assert (out["figures"], [c["id"] for c in out["created"]], out["kept"]) == (1, [3], [])
    job = _data(project)["jobs"][-1]
    assert job["models"] == ["adam_run", "rms_run", "sgd_run"]                     # in legend order: Adam, RMSprop, SGD
    assert job["array_epoch"] == {"mode": "latest"} and job["legend"]["labels"] == RUNS
    again = gx.study(project, "fig10", legend=["trainer.optimizer.type"], metrics=metric)
    assert (again["created"], again["kept"]) == ([], [3])
    split = gx.study(project, "fig10", legend=[], metrics=metric)            # every optimizer a figure of its own
    assert (split["figures"], len(split["created"]), split["kept"]) == (3, 2, [1])   # job 1 already draws adam_run alone


def test_a_study_traceflow_cannot_make_fails_with_the_one_reason_and_writes_nothing(project, monkeypatch):
    _rank_choices(monkeypatch)
    _queues(project, {"fig10": ["make orchestrate"]})
    same = project.snapshot("adam_run")
    monkeypatch.setattr(project, "snapshot", lambda name: same)                # nothing tells the runs apart
    before = project.graph_config.read_text()
    with pytest.raises(gx.StudyError) as failed:
        gx.study(project, "fig10", legend=[], metrics=[("cross_validation", "validation", "mean_attack_rank_curve")])
    assert str(failed.value) == gx.STUDY_FAILED and "share one legend label" in failed.value.detail
    with pytest.raises(gx.StudyError, match="studied automatically"):
        gx.study(project, "fig10", legend=[], metrics=[("cross_validation", "validation", "mean_accuracy")])
    assert project.graph_config.read_text() == before


def test_a_relative_fit_time_divides_each_figure_by_the_reference_line_chosen(project, monkeypatch):
    from services.graph_manager.graph_job_validator import validate_graph_job

    monkeypatch.setattr(gx, "choices", lambda p, models: {m: [{"source": "cross_validation", "process": "train",
                                                                "metric": "mean_fit_time_s", "kind": "per_epoch"}] for m in models})
    _queues(project, {"fig10": ["make orchestrate"]})
    snapshot = project.snapshot
    monkeypatch.setattr(project, "snapshot", lambda name: {**snapshot(name), "metrics_to_log": ["mean_fit_time_{s}"]})
    plan = gx.study_plan(project, "fig10")
    assert plan["metrics"][0]["logged"] and plan["metrics"][0]["relative"] and plan["values"]["sgd_run"] == {"trainer.optimizer.type": "SGD"}
    fit = [("cross_validation", "train", "mean_fit_time_s")]
    with pytest.raises(gx.ExportError, match="needs the reference line"):
        gx.study(project, "fig10", legend=["trainer.optimizer.type"], metrics=[], relative=fit)
    out = gx.study(project, "fig10", legend=["trainer.optimizer.type"], metrics=fit, relative=fit,
                   reference={"trainer.optimizer.type": "RMSprop"})
    assert [c["relative"] for c in out["created"]] == [False, True]
    job = _data(project)["jobs"][-1]
    assert job["relative"] == {"rule": "run", "where": [{"path": "trainer.optimizer.type", "equals": "RMSprop"}], "label": "RMSprop"}
    assert job["plot_type"] == "barh" and "relative" in job["name"] and "relative" not in _data(project)["jobs"][-2]
    assert validate_graph_job(job)["relative"]["rule"] == "run"
    again = gx.study(project, "fig10", legend=["trainer.optimizer.type"], metrics=fit, relative=fit,
                     reference={"trainer.optimizer.type": "RMSprop"})
    assert again["created"] == [] and len(again["kept"]) == 2
    with pytest.raises(gx.StudyError, match="studied automatically") as failed:
        gx.study(project, "fig10", legend=["trainer.optimizer.type"], metrics=[], relative=fit,
                 reference={"trainer.optimizer.type": "Adagrad"})
    assert "not a line" in failed.value.detail


def test_add_a_job_writes_one_at_the_next_id_without_models_and_disabled_until_they_are_listed(project):
    from services.graph_manager.graph_job_validator import validate_graph_job

    out = gx.new_job(project, "my figure")
    job = _data(project)["jobs"][-1]
    assert out == {"ok": True, "created": 3, "name": "my_figure"}
    assert (job["id"], job["name"], job["enabled"], job["models"], job["array_epoch"]) == (3, "my_figure", False, [], {"mode": "latest"})
    assert "top_models" not in job and job["legend"]["mode"] == "auto"
    assert "  # JOB 3: my_figure" in project.graph_config.read_text()
    assert validate_graph_job(job, require_models=False)["array_epoch"]["mode"] == "latest"


def test_add_a_facet_writes_one_at_the_next_id_without_panels_and_its_panels_are_added_after(project):
    from services.graph_manager.faceting import validate_facet
    from services.visualization.vscode_monitor import yamledit as Y

    assert gx.new_facet(project, "two runs") == {"ok": True, "created": 2, "name": "two_runs"}
    facet = _data(project)["faceting"][-1]
    assert (facet["id"], facet["enabled"], facet["panels"], facet["arrangement"]) == (2, False, [], {"mode": "horizontal"})
    assert "  # FACET 2: two_runs" in project.graph_config.read_text()
    text = project.graph_config.read_text()
    for k in (1, 2):                                           # + add a panel, twice; Find Figure fills them
        text = Y.edit_text(text, "add", ["faceting", 1, "panels"], f'{{"file_id": {k}, "address": "jobs/job_00{k}_x/{k}.png"}}')
    facet = yaml.safe_load(text)["faceting"][1]
    assert validate_facet({**facet, "enabled": True})["arrangement"]["order"] == [1, 2]   # drawn as listed


def test_a_done_run_skipped_from_another_sweep_does_not_inflate_a_queue():
    """fig10: ketchup_computer (DONE, trained in fig10_optimizers' 5-optimizer sweep) joined fig10's execution as
    skipped; fig10's own runs are a 1-combination sweep: 6 lines plan 6, not 30 (state.js queueStates agrees)."""
    from services.graph_manager import graph_export as ge

    batch = "fig10-20261019T231500Z"
    skipped = {"model_name": "ketchup_computer", "status": "DONE", "total_trials": 5,
               "started_at": "2026-10-15T20:16:17+08:00", "batch_ids": ["fig10_optimizers-20261015T121500Z", batch]}
    own = [{"model_name": f"r{i}", "status": "DONE", "total_trials": 1, "started_at": f"2026-10-20T{7 + i:02d}:30:00+08:00",
            "batch_ids": [batch]} for i in range(1, 6)]
    runs = ge._latest_batch([skipped, *own], "fig10")
    assert len(runs) == 6 and ge._latest_batch_id([skipped, *own], "fig10") == batch
    assert ge._queue_done(runs, 6, batch)                         # 6 lines x 1: done
    assert not ge._queue_done(runs[:5], 6, batch)                 # one line still to run
    assert ge._queue_done(runs, 6, "")                            # no batch start known: never ketchup's 5; one run per line


def test_a_retried_queue_that_trained_nothing_yet_is_sized_by_its_own_runs():
    """Retry Start opened a new fig10 batch before training anything: foamy_crackle (fig10's own, interrupted) and
    ketchup_computer (fig10_optimizers', 5 trials) both carry it; fig10 plans 6 lines x 1, not 30."""
    from services.graph_manager import graph_export as ge

    new = "fig10-20261019T232656Z"
    ketchup = {"model_name": "ketchup_computer", "status": "DONE", "total_trials": 5, "started_at": "2026-10-15T20:16:17+08:00",
               "batch_ids": ["fig10_optimizers-20261015T120942Z", "fig10-20261019T230835Z", new]}
    own = [{"model_name": f"r{i}", "status": "DONE", "total_trials": 1, "started_at": "2026-10-20T07:16:18+08:00",
            "batch_ids": ["fig10-20261019T230835Z", new]} for i in range(5)]
    runs = ge._latest_batch([ketchup, *own], "fig10")
    assert ge._queue_done(runs, 6, new, "fig10")                    # 6 runs, 6 lines x 1: done
    assert ge._queue_done(runs, 6, new, "")                         # without the queue: never ketchup's 5; one run per line


def test_a_queue_whose_only_run_another_queue_trained_plans_one_run_per_line():
    """fig10 after foamy_crackle was deleted: its latest batch holds only ketchup_computer (fig10_optimizers', 5 trials).
    6 lines plan one run each, never 5 x 6 and never what params.yaml plans now."""
    from services.graph_manager import graph_export as ge

    new = "fig10-20261019T232656Z"
    ketchup = {"model_name": "ketchup_computer", "status": "DONE", "total_trials": 5, "started_at": "2026-10-15T20:16:17+08:00",
               "batch_ids": ["fig10_optimizers-20261015T120942Z", new]}
    runs = ge._latest_batch([ketchup], "fig10")
    assert not ge._queue_done(runs, 6, new, "fig10")                # 1 of 6
    assert ge._queue_done(runs, 1, new, "fig10")                    # a one-line queue it fills


def test_a_run_another_queue_retrained_later_does_not_size_the_queue_it_left():
    """fig24 (8 lines x 1) trained haberdashery_flock; fig20's FORCE trained it again later (total_trials 5, DONE
    again): fig24 plans 8, not 40, so its 8 DONE runs make it done (state.js queueStates agrees)."""
    from services.graph_manager import graph_export as ge

    batch = "fig24-20261019T151315Z"
    hab = {"model_name": "haberdashery_flock", "status": "DONE", "total_trials": 5, "started_at": "2026-10-20T18:17:16+08:00",
           "batch_ids": ["fig24-20261019T081932Z", batch, "fig20-20261020T085403Z"]}
    own = [{"model_name": f"a{i}", "status": "DONE", "total_trials": 1, "started_at": f"2026-10-19T2{i}:00:00+08:00",
            "batch_ids": [batch]} for i in range(7)]
    runs = ge._latest_batch([hab, *own], "fig24")
    assert len(runs) == 8 and ge._queue_done(runs, 8, batch, "fig24")
    assert not ge._queue_done([{**hab, "status": "INTERRUPTED"}, *own], 8, batch, "fig24")   # fig20 stopped it: not done


def test_a_job_without_an_id_neither_stops_a_new_job_nor_is_counted_or_moved(project):
    """A template's job (`- id:` empty) or one written by hand with no id yet: new-job takes the next id
    after the integer ids (no TypeError), the entry is never matched by id, and the id shift leaves it."""
    text = project.graph_config.read_text()
    block = "  - id:\n    name: \"typed later\"\n    type: \"solo\"\n    models: [adam_run]\n"
    at = text.find("\nfaceting:")                       # into the jobs list: before the facets, else at the end
    project.graph_config.write_text(text[:at + 1] + block + text[at + 1:] if at >= 0 else text + block)
    out = gx.new_job(project, "my figure")
    ids = [j.get("id") for j in _data(project)["jobs"]]
    assert out["created"] == max(i for i in ids if isinstance(i, int)) and ids.count(None) == 1
    assert gx._int_id({"id": None}) is None and gx._int_id({"id": ""}) is None and gx._int_id({"id": "3"}) == 3
    assert gx._ids([{"id": 2}, {"id": None}, {"id": "x"}, {}]) == [2]


# ── a study whose lines repeat on splits (profiling_split.random_state, a run each: Wouters' ten trainings) ──

SPLIT_RUNS = {f"{m}_{i}": (m, i) for m in ("Zaid_N0", "NoConv1_N0") for i in range(3)}


class _SplitProject(_Project):
    def registry(self):
        return {f"h{i}": {"model_name": n, "batches": ["w_fig1"], "status": "DONE"} for i, n in enumerate(SPLIT_RUNS)}


@pytest.fixture
def split_project(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "graph_config.yaml").write_text(CONFIG)
    (tmp_path / "config" / "experiment_queue.yaml").write_text(yaml.safe_dump({"w_fig1": ["make orchestrate"]}))
    for name, (model, split) in SPLIT_RUNS.items():
        run = tmp_path / "artifacts" / "dev" / "generated_models" / name
        (run / "config").mkdir(parents=True)
        snapshot = {"model_id": model, "model_name": name, "trainer": {"epochs": 50, "batch_size": 50},
                    "data": {"name": "ascad_n0_fs01_w", "profiling_split": {"enabled": True, "test_size": 0.1,
                                                                             "shuffle": True, "random_state": split}},
                    "sampled": {"model_id": model}}
        (run / "config" / f"{name}_params.yaml").write_text(yaml.safe_dump(snapshot))
    return _SplitProject(tmp_path)


def test_a_study_pools_the_splits_of_a_line_into_a_group_with_the_band_asked(split_project, monkeypatch):
    kinds = {"mean_attack_rank_curve": "curve", "mean_fit_time_s": "per_epoch"}
    monkeypatch.setattr(gx, "choices", lambda p, models: {m: [{"source": "attack_metrics", "process": "test", "metric": k,
                                                                "kind": v} for k, v in kinds.items()] for m in models})
    plan = gx.study_plan(split_project, "w_fig1")
    assert plan["repeats"] == 3 and [a["key"] for a in plan["axes"]] == ["model_id"]   # the split is no axis
    assert gx.split_of(split_project.snapshot("Zaid_N0_2")) == 2
    curve = [("attack_metrics", "test", "mean_attack_rank_curve")]
    with pytest.raises(gx.ExportError, match="repeat on 3 splits each: pick the band"):
        gx.study(split_project, "w_fig1", legend=["model_id"], metrics=curve)
    out = gx.study(split_project, "w_fig1", legend=["model_id"], metrics=curve, band="min_max")
    job = _data(split_project)["jobs"][-1]
    assert job["type"] == "model_overlay" and out["figures"] == 1
    assert job["groups"] == [{"label": "NoConv1_N0", "models": ["NoConv1_N0_0", "NoConv1_N0_1", "NoConv1_N0_2"], "band": "min_max"},
                             {"label": "Zaid_N0", "models": ["Zaid_N0_0", "Zaid_N0_1", "Zaid_N0_2"], "band": "min_max"}]
    from services.graph_manager.graph_job_validator import validate_graph_job
    assert validate_graph_job(job)["groups"] == job["groups"]                      # the job validates as written
    gx.study(split_project, "w_fig1", legend=["model_id"], metrics=[("attack_metrics", "test", "mean_fit_time_s")])
    bars = _data(split_project)["jobs"][-1]
    assert "groups" not in bars and bars["legend"]["labels"]["Zaid_N0_1"] == "Zaid_N0, split 1"   # a scalar: a bar per run
