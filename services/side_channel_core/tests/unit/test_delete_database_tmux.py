"""make delete_database keeps every batch's tmux log (artifacts/<env>/tmux_logs/<batch>.txt): deleting runs (Housekeeping),
a whole environment or the whole project; the dry run names the logs it keeps."""
from utils.delete_database import _wipe_keeping_tmux_logs


def test_full_wipe_keeps_the_batch_logs(tmp_path):
    art = tmp_path / "artifacts"
    (art / "dev" / "tmux_logs").mkdir(parents=True)
    (art / "dev" / "tmux_logs" / "fig26.txt").write_text("run 1\nrun 2\n")
    (art / "dev" / "generated_models" / "x").mkdir(parents=True)
    (art / "dev" / "heartbeat.json").write_text("{}")
    (art / "staging" / "generated_models").mkdir(parents=True)
    (art / "run_registry.json").write_text("{}")
    _wipe_keeping_tmux_logs(art)
    assert (art / "dev" / "tmux_logs" / "fig26.txt").read_text() == "run 1\nrun 2\n"
    assert sorted(p.relative_to(art).as_posix() for p in art.rglob("*")) == ["dev", "dev/tmux_logs", "dev/tmux_logs/fig26.txt"]


def test_full_wipe_without_batch_logs_removes_artifacts(tmp_path):
    art = tmp_path / "artifacts"
    (art / "dev" / "generated_models" / "x").mkdir(parents=True)
    _wipe_keeping_tmux_logs(art)
    assert not art.exists()


def test_deleting_runs_keeps_their_batchs_tmux_log_even_with_its_last_run(tmp_path):
    """A batch's tmux log holds every run of it, appended run after run, the record of how the batch ran: deleting
    runs (Housekeeping → DELETE) keeps it, with the batch's last run too, and the dry run names it."""
    import json

    from services.side_channel_core.utils.delete_database import DeletionPlan, main

    from tests.unit.test_delete_database import make_project

    root = make_project(tmp_path / "P", runs=[("h1", "alpha", "dev", "ascad"), ("h2", "beta", "dev", "ascad"),
                                              ("h3", "gamma", "dev", "ascad")])
    art = root / "artifacts"
    registry = json.loads((art / "run_registry.json").read_text())
    registry["h1"]["batches"] = registry["h2"]["batches"] = ["q1"]
    registry["h3"]["batches"] = ["q2"]
    for name in ("run_registry.json", "test_registry.json"):
        (art / name).write_text(json.dumps(registry))
    logs = art / "dev" / "tmux_logs"
    logs.mkdir()
    (logs / "q1.txt").write_text("alpha's run\nbeta's run\n")
    (logs / "q2.txt").write_text("gamma's run\n")

    plan = DeletionPlan(root, "dev", ["alpha"])
    assert any(line.startswith("kept         : ") and "q1.txt" in line and "1 other run(s)" in line for line in plan.describe())
    assert main(["--project-root", str(root), "--env", "dev", "--model", "alpha", "--yes"]) == 0
    assert (logs / "q1.txt").is_file()                         # beta's output is in it too

    plan = DeletionPlan(root, "dev", ["beta"])
    assert any(line.startswith("kept         : ") and "q1.txt" in line and "no run of it stays" in line for line in plan.describe())
    assert main(["--project-root", str(root), "--env", "dev", "--model", "beta", "--yes"]) == 0
    assert (logs / "q1.txt").read_text() == "alpha's run\nbeta's run\n"   # its last run is gone, its log stays
    assert (logs / "q2.txt").read_text() == "gamma's run\n"


def test_deleting_a_whole_environment_keeps_its_tmux_logs(tmp_path):
    from services.side_channel_core.utils.delete_database import main

    from tests.unit.test_delete_database import make_project

    root = make_project(tmp_path / "P")                        # dev: alpha, beta; staging: gamma (the project is not wiped)
    logs = root / "artifacts" / "dev" / "tmux_logs"
    logs.mkdir()
    (logs / "q1.txt").write_text("both runs\n")
    assert main(["--project-root", str(root), "--env", "dev", "--yes"]) == 0
    assert (logs / "q1.txt").read_text() == "both runs\n"
    assert sorted(p.name for p in (root / "artifacts" / "dev").iterdir()) == ["tmux_logs"]   # all else of dev is gone
