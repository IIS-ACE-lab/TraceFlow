# services/side_channel_core/tests/unit/test_attack_progress.py
"""utils/attack_progress.py: attack i of n and the attack metrics so far while the test attacks, written beside the
heartbeat for the TraceFlow Monitor; after the last attack the live values are the ones the test records."""

import json

import numpy as np
import torch

from sca.metrics import _stable_threshold_trace
from sca.perform_attack import perform_attacks
from test_inference import TARGET_BYTE, TRUE_KEY, _metadata, _synthetic_logits
from utils import attack_progress
from utils.seal import load_json

METRICS = ["mean_attack_rank_curve", "per_attack_curves", "final_ranks", "traces_to_ge_1", "per_attack_traces_to_ge_1",
           "mean_accuracy"]


def _attack(process, attacks=5):
    logits, plaintexts = _synthetic_logits(n=120, scale=2.0)
    meta = np.empty(120, dtype=object)
    for i, row in enumerate(_metadata(plaintexts, TRUE_KEY)):
        meta[i] = row
    cfg = {"seed": 1, "sca": {"attack_byte": TARGET_BYTE, "leakage_model": "ID", "variant": "fixed_key", "rank_base": 1},
           "perform_attack": {"enabled": True, "attack_step": 1, "ge_thresholds": [1],
                              process: {"attack_times": attacks, "shuffle": True, "attack_distribution": "full_population"}},
           "metrics_cfg": {"rank_log_eps": 1e-40}, "metrics_to_log": METRICS, "model_name": "run", "model_hash": "abcd1234",
           "data": {"name": "syn", "profiling_range": [0, 10], "attack_range": [0, 120], "validation_range": [0, 120],
                    "test_range": [0, 120]}}
    log64 = torch.log_softmax(logits.double(), dim=1)
    return cfg, lambda: perform_attacks(preds=log64, metadata=meta, cfg=cfg, process=process, preds_are_log_probs=True)


def _run_dir(tmp_path):
    run = tmp_path / "artifacts" / "dev" / "generated_models" / "run"
    run.mkdir(parents=True)
    return run


def test_after_the_last_attack_the_live_values_are_the_ones_the_test_records(tmp_path):
    run = _run_dir(tmp_path)
    cfg, attack = _attack("test")
    with attack_progress.reporting(cfg, run):
        summary = attack()
    progress = load_json(run.parent.parent / "attack_progress.json")          # sealed, as every artifact JSON
    assert (progress["attack"], progress["attacks"], progress["running"]) == (5, 5, False)
    assert (progress["model_name"], progress["model_hash"]) == ("run", "abcd1234")
    assert 0 < progress["started_ns"] <= progress["written_ns"]                 # when the test began: its speed for the ETA
    live = progress["metrics"]
    assert set(live) == {"mean_attack_rank_curve", "final_ranks", "traces_to_ge_1", "per_attack_traces_to_ge_1"}
    # the formulas of sca/metrics.py on the summary the test computes its metrics from
    finals = np.asarray(summary["final_ranks"], dtype=np.float64)
    assert live["mean_attack_rank_curve"] == live["final_ranks"] == float(np.mean(finals))
    assert live["mean_attack_rank_curve"] == float(np.asarray(summary["mean_attack_rank_curve"])[-1, 1])
    assert live["traces_to_ge_1"] == _stable_threshold_trace(summary["mean_attack_rank_curve"], threshold=1.0, direction="le")
    crossings = [_stable_threshold_trace(c, threshold=1.0, direction="le")
                 for c in np.asarray(summary["per_attack_curves"], dtype=np.float64)]
    reached = [float(c) for c in crossings if c is not None]
    assert live["per_attack_traces_to_ge_1"] == {"value": float(np.mean(reached)) if reached else None,
                                                 "reached": len(reached), "of": 5}


def test_the_file_is_rewritten_after_every_attack(tmp_path, monkeypatch):
    run = _run_dir(tmp_path)
    cfg, attack = _attack("test", attacks=4)
    seen = []
    real = attack_progress._Writer.write
    monkeypatch.setattr(attack_progress._Writer, "write", lambda self, done, total: (seen.append((done, total)), real(self, done, total)))
    with attack_progress.reporting(cfg, run):
        attack()
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert json.loads((run.parent.parent / "attack_progress.json").read_text().split("\n")[0])["running"] is False


def test_validation_attacks_and_attacks_outside_the_block_write_nothing(tmp_path):
    run = _run_dir(tmp_path)
    cfg, attack = _attack("validation")
    with attack_progress.reporting(cfg, run):
        attack()                                                       # the epoch's validation attacks
    _cfg, test_attack = _attack("test")
    test_attack()                                                      # a test outside reporting()
    assert not (run.parent.parent / "attack_progress.json").exists()


def test_a_folder_that_is_not_a_run_folder_gets_no_file(tmp_path):
    cfg, attack = _attack("test")
    with attack_progress.reporting(cfg, tmp_path / "somewhere"):
        attack()
    assert not list(tmp_path.parent.glob("attack_progress.json")) and not list(tmp_path.rglob("attack_progress.json"))
