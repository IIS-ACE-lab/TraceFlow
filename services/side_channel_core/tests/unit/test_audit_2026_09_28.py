"""Regression tests for the 2026-09-28 system audit fixes."""

import numpy as np
import pytest
import torch


def _attack_cfg(n, plan, step=10):
    return {"sca": {"attack_byte": 2, "leakage_model": "ID", "rank_base": 1, "variant": "fixed_key"},
            "perform_attack": {"enabled": True, "attack_step": step, "ge_thresholds": 1, "test": plan},
            "metrics_cfg": {"rank_log_eps": 1e-40}, "data": {"name": "ascad", "test_range": [0, n]}, "seed": 42}


def _leaky_attack(n=400, key=0xE0, seed=3):
    rng = np.random.default_rng(seed)
    plain = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    meta = [{"plaintext": plain[i], "key": np.full(16, key, np.uint8)} for i in range(n)]
    from services.side_channel_core.src.sca.leakage_models import build_label_table
    logits = rng.standard_normal((n, 256))
    logits[np.arange(n), build_label_table("ID")[plain[:, 2], key]] += 1.0
    return torch.log_softmax(torch.as_tensor(logits), dim=1), meta


def test_full_population_endpoint_is_one_success_rate_trial():
    from services.side_channel_core.src.sca.perform_attack import perform_attacks

    preds, meta = _leaky_attack()
    plan = {"attack_times": 20, "shuffle": True, "attack_distribution": "full_population"}
    result = perform_attacks(preds, meta, _attack_cfg(400, plan), process="test", preds_are_log_probs=True)
    assert result["attack_repetitions"] == 20 and result["sr_trials"] == 1
    assert result["successful_attacks"] in (0, 1)
    assert result["attack_protocol"]["score_rule"] == "log_likelihood"


def test_disjoint_key_evidence_is_the_combined_attack():
    from services.side_channel_core.src.sca.perform_attack import perform_attacks

    preds, meta = _leaky_attack()
    plan = {"attack_times": 4, "shuffle": False, "attack_distribution": "disjoint"}
    result = perform_attacks(preds, meta, _attack_cfg(400, plan), process="test", preds_are_log_probs=True)
    evidence = result["__key_recovery_evidence__"]
    combined = perform_attacks(
        preds, meta, _attack_cfg(400, {"attack_times": 1, "shuffle": False, "attack_distribution": "full_population"}),
        process="test", preds_are_log_probs=True,
    )["__key_recovery_evidence__"]
    # Rank, recovery traces and top hypothesis all describe one attack on all traces.
    for field in ("final_true_key_rank", "top_key_hypothesis", "first_rank1_trace", "stable_rank1_trace"):
        assert evidence[field] == combined[field]
    assert evidence["ranking_basis"] == "combined_attack_on_all_disjoint_traces"
    assert result["sr_trials"] == 4


def test_trace_budget_must_be_a_curve_point():
    from services.side_channel_core.src.sca.metrics import _rank_at_trace_budget

    curve = np.column_stack([np.arange(10, 110, 10), np.linspace(50, 1, 10)])
    assert _rank_at_trace_budget(curve, 50) == curve[4, 1]
    for budget in (2000, 3, 55):
        with pytest.raises(ValueError, match="not a point of the rank curve"):
            _rank_at_trace_budget(curve, budget)


def test_probability_head_is_rejected():
    from services.side_channel_core.utils.epoch_utils import _reject_probability_head

    logits = torch.randn(4, 256)
    _reject_probability_head(logits, where="t")
    _reject_probability_head(torch.log_softmax(logits, dim=1), where="t")   # LogSoftmax head is exact
    with pytest.raises(ValueError, match="softmax twice"):
        _reject_probability_head(torch.softmax(logits, dim=1), where="t")


def test_hash_covers_settings_that_change_numbers():
    import copy
    from services.side_channel_core.utils.model_naming import make_experiment_hash

    base = {"data": {"name": "ascad"}, "model": {"id": "m"}, "trainer": {"epochs": 1}, "seed": 42,
            "project": {"name": "p", "root": "/r", "scripts": "/s"}, "sca": {"attack_byte": 2},
            "metrics_cfg": {"rank_log_eps": 1e-40}, "validation": {"rank_step": 10, "rank_eval_interval": 1},
            "key_recovery_reporting": {"recovery_rank": 1, "require_stable_recovery": True}, "device": "cuda"}
    reference = make_experiment_hash(base)
    for path, value in ((("metrics_cfg", "rank_log_eps"), 0.0), (("validation", "rank_step"), 1),
                        (("validation", "rank_eval_interval"), 5),
                        (("key_recovery_reporting", "require_stable_recovery"), False), (("device",), "cpu")):
        changed = copy.deepcopy(base)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert make_experiment_hash(changed) != reference, path


def test_graph_summary_uses_trace_counts():
    from services.graph_manager.metric_loader import _summarize_curve

    x = np.arange(10, 1010, 10)
    curve = np.column_stack([x, np.where(x >= 500, 1.0, 50.0)])
    assert _summarize_curve(curve, summary_cfg={"type": "traces_to_threshold", "threshold": 1}) == 500
    assert _summarize_curve(curve, summary_cfg={"type": "value_at_trace", "trace_index": 300}) == 50.0
    with pytest.raises(ValueError, match="not a point of the curve"):
        _summarize_curve(curve, summary_cfg={"type": "value_at_trace", "trace_index": 305})


def test_dataset_typo_raises(monkeypatch):
    from services.side_channel_core.utils.path_utils import _pick_data_block

    cfg = {"data": [{"name": "ascad"}, {"name": "ascad_desync50"}]}
    monkeypatch.setenv("DATASET", "ascad_desync_50")
    with pytest.raises(ValueError, match="matches no entry"):
        _pick_data_block(cfg)
    monkeypatch.delenv("DATASET")
    assert _pick_data_block(cfg) is None                     # several datasets, none chosen


def _fold_report(model_hash="abcd1234", epoch=3):
    """A valid final-epoch fold report of model m (validation process)."""
    from src.sca.key_recovery import build_key_recovery_report
    from src.sca.perform_attack import perform_attacks

    preds, meta = _leaky_attack()
    cfg = _attack_cfg(400, None)
    cfg["perform_attack"].pop("test")
    cfg["perform_attack"]["validation"] = {"attack_times": 1, "shuffle": False,
                                           "attack_distribution": "full_population"}
    cfg["data"] = {"name": "ascad", "validation_range": [0, 400]}
    cfg["key_recovery_reporting"] = {"enabled": True, "recovery_rank": 1, "require_stable_recovery": True}
    evidence = perform_attacks(preds, meta, cfg, process="validation",
                               preds_are_log_probs=True)["__key_recovery_evidence__"]
    return build_key_recovery_report(cfg=cfg, process="validation", evidence=evidence,
                                     epoch=epoch, model_name="m", model_hash=model_hash)


def test_cv_report_rejects_damaged_and_foreign_fold_reports(tmp_path, monkeypatch, caplog):
    import json
    import logging
    import types
    import yaml
    import utils.cross_validation as cv
    from services.side_channel_core.utils.key_recovery_helper import _load_cv_headline

    summary = tmp_path / "summary.yaml"
    summary.write_text(yaml.safe_dump({"seed": 42, "headline": {
        "n_folds": 4, "model_name": "m", "model_hash": "abcd1234", "epoch": 3}}))
    monkeypatch.setattr(cv, "resolve_cross_validation_paths",
                        lambda cfg, create: types.SimpleNamespace(summary=summary))
    monkeypatch.setattr(cv, "get_fold_artifact_paths",
                        lambda paths, fold_index, **kw: {"key_recovery": tmp_path / f"fold_{fold_index}"})
    reports = {0: _fold_report(), 1: "{truncated", 2: _fold_report(model_hash="ffff9999"),
               3: _fold_report(epoch=2)}
    for index, report in reports.items():
        (tmp_path / f"fold_{index}").mkdir()
        text = report if isinstance(report, str) else json.dumps(report, default=str)
        (tmp_path / f"fold_{index}" / "validation_best.json").write_text(text)

    with caplog.at_level(logging.ERROR):
        folds = _load_cv_headline({}, logging.getLogger("t"))["fold_reports"]
    assert [fold["seed"] for fold in folds] == [43, 44, 45, 46]      # the seeds after the refit's 42
    assert folds[0]["report"] is not None and folds[0]["problem"] is None
    assert folds[1]["report"] is None and folds[1]["problem"].startswith("report damaged")
    assert folds[2]["report"] is None and "model.hash 'ffff9999' != 'abcd1234'" in folds[2]["problem"]
    assert folds[3]["report"] is None and "epoch 2 != 3" in folds[3]["problem"]
    assert sum("rejected" in record.getMessage() for record in caplog.records) == 3
