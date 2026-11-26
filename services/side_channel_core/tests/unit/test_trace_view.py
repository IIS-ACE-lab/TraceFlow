# services/side_channel_core/tests/unit/test_trace_view.py
"""The planner's trace lines (services/visualization/vscode_monitor/trace_view.py): each layer's line is that
layer's output in the model's own forward pass (PyTorch and Keras), the trace is the data entry's attack trace as
the database holds it (window and desync offset), and a trained run's checkpoint is the one its manifest names."""

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from services.visualization.vscode_monitor import trace_view as TV
from services.visualization.vscode_monitor.model_check import walk

PROJECT = "DLSCA_ASCAD"
MLP = [{"Flatten": {}}, {"Linear": {"in_features": 700, "out_features": 16, "activation": "relu"}},
       {"Dropout": {"p": 0.5}}, {"Linear": {"in_features": 16, "out_features": "auto"}}]


def _entry(name):
    cfg = TV._project_cfg(PROJECT)
    entry = next((e for e in TV._entries_of(cfg) if e.get("name") == name), None)
    path, why = TV._database(cfg, entry) if entry else (None, "no entry")
    if why:
        pytest.skip(f"{name}: {why}")
    return entry, path


def test_a_line_is_at_most_POINTS_bin_means():
    assert TV._line(np.arange(10)) == [float(v) for v in range(10)]
    line = TV._line(np.arange(700, dtype=float))
    assert len(line) == TV.POINTS
    assert line[0] == pytest.approx(np.arange(6).mean())               # 700 = 100 bins of 6 + 20 of 5


@pytest.mark.parametrize("framework", ["pytorch", "keras"])
def test_the_walk_ends_at_the_models_own_output(framework):
    from core.model import build_model

    cfg = TV._project_cfg(PROJECT)
    cfg["data"] = TV._entries_of(cfg)[0]
    cfg["sca"] = {k: TV._first(v) for k, v in (cfg.get("sca") or {}).items()}
    cfg["model_id"] = "T"
    cfg["model"] = [{"model_id": "T", "architecture": "MLP", "architecture_args": {"layers": copy.deepcopy(MLP)}}]
    cfg.setdefault("deep_learning", {})["framework"] = framework
    torch.manual_seed(0)
    net = build_model(cfg, "T")
    net.eval()
    x = torch.randn(1, 1, 700)
    with torch.no_grad():
        steps = list(walk(net, MLP, x))
        whole = net(x)
    assert [s[1] for s in steps] == ["Flatten", "Linear", "Dropout", "Linear"]
    assert torch.allclose(torch.as_tensor(steps[-1][2]).float(), torch.as_tensor(whole).float(), atol=1e-5)
    kind, shape, lines = TV._lines(steps[1][2])
    assert kind == "vector" and shape == [16] and lines.shape == (1, 16) and lines.min() >= 0      # relu


def test_a_design_trace_is_the_entrys_attack_trace_as_the_database_holds_it():
    entry, path = _entry("ascad")
    out = TV.design(PROJECT, MLP, "MLP", "ascad", 3)
    assert out["error"] is None, out["error"]
    with h5py.File(path, "r") as handle:
        lo, _ = entry["attack_range"]
        start, end = entry["target_range"]
        raw = handle["Attack_traces"]["traces"][lo + 3, start:end].astype(np.float64)
    assert out["input"]["values"] == TV._line(raw)
    assert out["source"]["row"] == lo + 3 and out["weights"] == "untrained"
    assert out["target"]["expression"] == f"Sbox[p[{out['target']['attack_byte']}] ^ k[{out['target']['attack_byte']}]]" or out["target"]["leakage_model"] != "ID"
    assert path.as_posix().endswith(out["source"]["file"]) and out["source"]["group"] == "Attack_traces/traces"
    assert out["source"]["samples"] == [start, end]
    assert [layer["type"] for layer in out["layers"]] == ["Flatten", "Linear", "Dropout", "Linear"]
    assert out["layers"][0]["values"] == out["input"]["values"]      # Flatten passes the trace on
    again = TV.design(PROJECT, MLP, "MLP", "ascad", 3)
    assert again["layers"] == out["layers"], "the same untrained weights on every call (seeded)"


def test_a_raw_entrys_trace_is_its_window_at_the_desync_offset_drawn():
    entry, path = _entry("ascad_desync100_raw")
    out = TV.design(PROJECT, MLP, "MLP", "ascad_desync100_raw", 2)
    assert out["error"] is None, out["error"]
    off = out["source"]["desync_offset"]
    assert 0 <= off <= int(entry["attack_desync"])
    lo, _ = entry["attack_range"]
    start, end = entry["target_range"]
    with h5py.File(path, "r") as handle:
        raw = handle["traces"][lo + 2, start + off:end + off].astype(np.float64)
    assert out["input"]["values"] == TV._line(raw)
    assert out["source"]["group"] == "traces" and out["source"]["samples"] == [start + off, end + off]


def test_a_range_is_the_mean_of_its_traces_with_a_band_from_their_lowest_to_their_highest_value():
    _entry("ascad")
    ones = [TV.design(PROJECT, MLP, "MLP", "ascad", k) for k in (10, 11, 12)]
    out = TV.design(PROJECT, MLP, "MLP", "ascad", 10, 12)
    assert out["error"] is None, out["error"]
    assert out["count"] == 3 and out["source"]["row_to"] == out["source"]["row"] + 2 and "top" not in out
    inputs = np.array([o["input"]["values"] for o in ones])                     # 700 samples: each bin's mean
    assert np.allclose(out["input"]["values"], inputs.mean(axis=0), rtol=1e-3, atol=1e-3)
    for layer in out["layers"]:
        lo, hi = (np.array(b) for b in layer["band"])
        assert np.all(lo <= np.array(layer["values"]) + 1e-6) and np.all(np.array(layer["values"]) <= hi + 1e-6)
    last = np.array([o["layers"][-1]["values"] for o in ones])                    # 256 classes: bins of 2 or 3
    assert np.allclose(out["layers"][-1]["values"], last.mean(axis=0), rtol=1e-3, atol=1e-3)


def test_a_range_is_at_most_MAX_RANGE_traces_and_ends_after_it_starts():
    _entry("ascad")
    assert TV.design(PROJECT, MLP, "MLP", "ascad", 0, TV.MAX_RANGE)["error"] == f"traces 0 to {TV.MAX_RANGE}: a range takes at most {TV.MAX_RANGE} traces"
    assert TV.design(PROJECT, MLP, "MLP", "ascad", 5, 4)["error"] == "traces 5 to 4: the range ends before it starts"


def test_a_trace_outside_the_attack_set_is_refused_with_its_range():
    entry, _ = _entry("ascad")
    lo, hi = entry["attack_range"]
    out = TV.design(PROJECT, MLP, "MLP", "ascad", hi - lo)
    assert out["error"] == f"trace {hi - lo}: the attack set of ascad has traces 0 to {hi - lo - 1}"


def _a_done_run():
    from utils.run_map import load_run_map

    root = Path(TV._project_cfg(PROJECT)["project"]["root"])
    for rec in load_run_map(root).values():
        folder = root / str(rec.get("path") or "")
        name = rec.get("model_name")
        if str(rec.get("status")).upper() == "DONE" and name and (folder / "model" / f"{name}_trained.pt").is_file() \
                and (folder / "data" / "test.h5").is_file():
            return str(name)
    pytest.skip("no finished run with its weights and test.h5 on this machine")


def test_a_trained_trace_uses_the_checkpoint_its_manifest_names_and_ranks_the_true_class():
    name = _a_done_run()
    out = TV.run(PROJECT, name, 0)
    assert out["error"] is None, out["error"]
    assert out["checkpoint"]["path"].endswith(f"/{name}/model/{name}_trained.pt") or out["checkpoint"]["policy"] == "best_validation"
    assert out["source"]["file"].endswith(f"/{name}/data/test.h5")
    classes = out["layers"][-1]["shape"][0]
    assert 1 <= out["label_rank"] <= classes and len(out["top"]) == TV.TOP
    assert out["target"]["classes"] == classes == out["outputs"] and out["target"]["from"] == f"{name}'s training snapshot"
    if out["target"]["target"] == "sbox" and out["target"]["leakage_model"] == "ID":   # the true class is made of these bytes
        from sca.leakage_models import compute_leakage_label
        assert out["label"] == int(compute_leakage_label(out["bytes"]["plaintext"], out["bytes"]["key"], "ID"))
    many = TV.run(PROJECT, name, 0, 19)
    assert many["error"] is None and many["count"] == 20 and many["ranks"]["count"] == 20
    assert many["ranks"]["best"] <= out["label_rank"] <= many["ranks"]["worst"]      # trace 0 is one of them


def test_a_manifest_naming_another_runs_checkpoint_is_refused(monkeypatch):
    from sca.test_request import load_snapshot, resolve_models
    from utils import checkpoint_policy
    from utils.checkpoint_policy import CheckpointPolicyError
    from utils.path_utils import resolve_paths_dict

    name = _a_done_run()
    cfg = load_snapshot(PROJECT, resolve_models(PROJECT, [name])[0])
    cfg["paths_abs"] = resolve_paths_dict(cfg["paths"], cfg, inject=True, mkdir=False)
    real = checkpoint_policy.load_checkpoint_manifest(cfg, model_name=name)
    forged = json.loads(json.dumps(real))
    forged["final"]["checkpoint"] = forged["final"]["checkpoint"].replace(f"/{name}/", "/someone_else/")
    monkeypatch.setattr(checkpoint_policy, "load_checkpoint_manifest", lambda *a, **k: forged)
    if cfg["evaluation"]["test"]["load_model"] == "best_validation" and cfg["evaluation"]["validation"]["checkpointing"]:
        pytest.skip("this run tests its best-validation checkpoint")
    with pytest.raises(CheckpointPolicyError, match="the manifest names"):
        TV._checkpoint(cfg, name)


def test_key_recovery_is_what_the_registries_recorded_for_the_run():
    from utils.seal import load_json

    name = _a_done_run()
    out = TV.recovery(PROJECT, name)
    root = Path(TV._project_cfg(PROJECT)["project"]["root"])
    entry = load_json(root / "artifacts" / "run_registry.json")[out["model_hash"]]
    assert out["error"] is None and out["true_key"] == entry["true_key"]
    assert out["training"]["cv"] == isinstance(entry.get("fold_key_recovery"), list)
    if out["training"]["cv"]:
        assert out["training"]["n_folds"] == len(out["folds"]) and out["training"]["profiling"] > 0
        # each fold's traces as its sealed report recorded them, and the rules the reports were made under
        assert all(f["traces"] for f in out["folds"]) and sum(f["traces"] for f in out["folds"]) == out["training"]["profiling"]
    assert out["rules"]["keyspace"] == 256 and out["rules"]["recovery_rank"] >= 1 and isinstance(out["rules"]["stable_required"], bool)
    assert set(out["training"]["attack"]) == {"validation", "test"}
    if isinstance(entry.get("fold_key_recovery"), list):
        assert out["cv"] and [f["recovered_key"] for f in out["folds"]] == [f["recovered_key"] for f in entry["fold_key_recovery"]]
        assert [f["key_recovered"] for f in out["folds"]] == [f["key_recovered"] for f in entry["fold_key_recovery"]]
    else:
        assert out["validation"]["recovered_key"] == entry["recovered_key"]
