# services/side_channel_core/tests/unit/test_graph_trace_budget.py
"""Graph jobs' trace_budget and test keys (graph_manager/trace_budget.py, owner 2026-10-26): the attack curves read
at B traces per attack, final_ranks at B, and the test process drawn from an external test."""

import numpy as np
import pytest


def _job(**extra):
    job = {"id": 9, "name": "ranks", "type": "process_overlay", "enabled": True, "metric": "mean_attack_rank_curve",
           "source": "attack_metrics", "models": ["mesh_chicken"], "processes": ["validation", "test"],
           "output_file_name": "r"}
    job.update(extra)
    return job


def _per_attack(n=300, attacks=5, step=10, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(step, n + 1, step, dtype=float)
    return np.stack([np.column_stack([x, rng.integers(1, 257, x.size)]) for _ in range(attacks)]).astype(float)


# ------------------------------------------------------------------ validation of the job
def test_the_keys_are_normalised_and_empty_means_the_latest_test():
    from services.graph_manager.graph_job_validator import validate_graph_job

    job = validate_graph_job(_job(trace_budget="150", test=2))
    assert job["trace_budget"] == 150 and job["test"] == 2
    plain = validate_graph_job(_job(trace_budget=None, test="latest"))
    assert "trace_budget" not in plain and "test" not in plain             # latest: the default
    assert validate_graph_job(_job(test="Budget"))["test"] == "budget"
    assert validate_graph_job(_job(test="recorded"))["test"] == "recorded"
    assert validate_graph_job(_job(metric="final_ranks", type="solo", trace_budget=40))["trace_budget"] == 40


@pytest.mark.parametrize("extra, message", [
    ({"trace_budget": 0}, "a trace count >= 1"),
    ({"trace_budget": True}, "a trace count >= 1"),
    ({"trace_budget": "many"}, "a trace count >= 1"),
    ({"trace_budget": 100, "source": "key_recovery"}, "use source: attack_metrics"),
    ({"trace_budget": 100, "metric": "guessing_entropy", "type": "model_overlay"}, "applies to the attack curves"),
    ({"test": 0}, "a test's index"),
    ({"test": "newest"}, "a test's index"),
    ({"test": 1, "processes": ["validation"]}, "add test to processes"),
])
def test_keys_the_job_cannot_use_are_refused(extra, message):
    from services.graph_manager.graph_job_validator import GraphJobValidationError, validate_graph_job

    with pytest.raises(GraphJobValidationError, match=message):
        validate_graph_job(_job(**extra))


# ------------------------------------------------------------------ the cut
def test_a_curve_is_cut_at_a_recorded_point_only():
    from services.graph_manager.trace_budget import TraceBudgetError, cut_curve, final_ranks_at

    curves = _per_attack()
    mean = np.column_stack([curves[0, :, 0], curves[:, :, 1].mean(axis=0)])
    cut = cut_curve(mean, 150, "m")
    assert cut[-1, 0] == 150 and cut.shape == (15, 2)
    np.testing.assert_array_equal(cut_curve(curves, 150, "m")[:, :, 1].mean(axis=0), cut[:, 1])   # mean of the cut attacks
    np.testing.assert_array_equal(final_ranks_at(curves, 150, "m"), curves[:, 14, 1].astype(int))
    assert set(cut_curve({"a": mean}, 150, "m")) == {"a"}
    with pytest.raises(TraceBudgetError, match="not a recorded point"):
        cut_curve(mean, 155, "m")
    with pytest.raises(TraceBudgetError, match="beyond the run's 300 recorded traces"):
        cut_curve(mean, 310, "m")


def test_an_external_test_is_found_by_its_index(tmp_path):
    from services.graph_manager.trace_budget import external_test_dir

    with pytest.raises(FileNotFoundError, match="not run"):
        external_test_dir(tmp_path, 1)
    (tmp_path / "test" / "abcd1234_et_1").mkdir(parents=True)
    assert external_test_dir(tmp_path, 1) == tmp_path / "test" / "abcd1234_et_1" / "metrics" / "attack_metrics"


# ------------------------------------------------------------------ GraphManager end to end
def _store(metrics_dir, process, metric, array, epoch=1):
    folder = metrics_dir / "arrays" / process / metric
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"ep{epoch:04d}.npy", array)


def _spy(monkeypatch):
    import services.graph_manager.graph_manager as gm

    seen = {}
    real = gm.plot_curve_overlay

    def spy(**kwargs):
        seen.update({k: np.asarray(v) for k, v in kwargs["series"].items()})
        return real(**kwargs)

    monkeypatch.setattr(gm, "plot_curve_overlay", spy)
    return seen


def test_a_job_draws_both_processes_at_the_budget(tmp_path, monkeypatch):
    from test_final_graph import _manager, _project

    _project(tmp_path)
    run = tmp_path / "artifacts" / "dev" / "generated_models" / "mesh_chicken" / "metrics" / "attack_metrics"
    test, validation = _per_attack(300)[0], _per_attack(200, seed=1)[0]
    _store(run, "test", "mean_attack_rank_curve", test)
    _store(run, "validation", "mean_attack_rank_curve", validation, epoch=20)
    seen = _spy(monkeypatch)
    _manager(tmp_path, monkeypatch, [_job(trace_budget=100)]).execute(9)
    assert seen["test"][-1, 0] == 100 and seen["validation"][-1, 0] == 100
    np.testing.assert_array_equal(seen["test"], test[:10])


def test_the_test_process_is_drawn_from_an_external_test(tmp_path, monkeypatch):
    from test_final_graph import _manager, _project

    _project(tmp_path)
    root = tmp_path / "artifacts" / "dev" / "generated_models" / "mesh_chicken"
    recorded, external = _per_attack(300)[0], _per_attack(600, seed=2)[0]
    _store(root / "metrics" / "attack_metrics", "test", "mean_attack_rank_curve", recorded)
    _store(root / "test" / "aaaa1111_et_1" / "metrics" / "attack_metrics", "test", "mean_attack_rank_curve", external)
    seen = _spy(monkeypatch)
    _manager(tmp_path, monkeypatch, [_job(processes=["test"], test=1)]).execute(9)
    np.testing.assert_array_equal(seen["test"], external)


def _registry(tmp_path, tests):
    import json

    (tmp_path / "artifacts" / "test_registry.json").write_text(json.dumps({"aaaa1111": {
        "finished_at": "2026-10-01T00:00:00+00:00", "external_tests": tests}}))


def test_a_job_draws_the_latest_test_of_its_kind_never_an_earlier_one(tmp_path, monkeypatch):
    """Owner 2026-10-27: by default the latest All test; budget, the latest Budget test; recorded, the training's."""
    from test_final_graph import _manager, _project

    _project(tmp_path)
    root = tmp_path / "artifacts" / "dev" / "generated_models" / "mesh_chicken"
    recorded, older, newest, budget = (_per_attack(300, seed=s)[0] for s in (0, 1, 2, 3))
    (root / "metrics" / "attack_metrics" / "scalars" / "mesh_chicken_metrics_test_history.jsonl").write_text("{}\n")
    _store(root / "metrics" / "attack_metrics", "test", "mean_attack_rank_curve", recorded)
    for key, curve in (("aaaa1111_et_1", older), ("aaaa1111_et_2", newest), ("aaaa1111_et_3", budget)):
        _store(root / "test" / key / "metrics" / "attack_metrics", "test", "mean_attack_rank_curve", curve)
    rec = lambda i, kind, day, status="DONE": {"index": i, "kind": kind, "status": status, "finished_at": f"2026-10-{day:02d}T00:00:00+00:00"}
    _registry(tmp_path, {"aaaa1111_et_1": rec(1, "all", 2), "aaaa1111_et_2": rec(2, "all", 4), "aaaa1111_et_3": rec(3, "budget", 3),
                         "aaaa1111_et_4": rec(4, "all", 9, "FAILED")})
    for choice, expected in ((None, newest), ("budget", budget), ("recorded", recorded), (1, older)):
        seen = _spy(monkeypatch)
        _manager(tmp_path, monkeypatch, [_job(processes=["test"], **({} if choice is None else {"test": choice}))]).execute(9)
        np.testing.assert_array_equal(seen["test"], expected)


def test_latest_test_reads_the_kind_and_skips_unfinished_tests(tmp_path):
    from services.side_channel_core.utils.latest_test import latest_test, test_sources

    run = tmp_path / "run"
    (run / "metrics" / "attack_metrics" / "scalars").mkdir(parents=True)
    (run / "metrics" / "attack_metrics" / "scalars" / "r_metrics_test_history.jsonl").write_text("{}\n")
    for key in ("h_et_1", "h_et_2", "h_et_3"):
        (run / "test" / key / "metrics" / "attack_metrics").mkdir(parents=True)
    old_budget = {"index": 1, "status": "DONE", "finished_at": "2026-10-02T00:00:00+00:00",
                  "settings": {"perform_attack": {"test": {"attack_trace_budget": 250}}}}      # before the kinds: a budget test
    entry = {"finished_at": "2026-10-01T00:00:00+00:00", "external_tests": {
        "h_et_1": old_budget, "h_et_2": {"index": 2, "status": "FAILED", "kind": "all", "finished_at": "2026-10-09T00:00:00+00:00"},
        "h_et_3": {"index": 3, "status": "DONE", "kind": "budget", "finished_at": "2026-10-03T00:00:00+00:00"}}}
    assert [t["id"] for t in test_sources(run, entry)] == ["recorded", "h_et_1", "h_et_3"]
    assert latest_test(run, entry, "all")["id"] == "recorded" and latest_test(run, entry, "budget")["id"] == "h_et_3"
    assert latest_test(tmp_path / "none", {}, "budget") is None
