"""deep_learning.<framework>.optimizer_args wired into trainer.optimizer.args by the sweep."""

import copy

import pytest

from utils.sweep_engine import count_all_param_combinations, generate_param_combinations


def _cfg(framework="keras"):
    return {
        "deep_learning": {
            "framework": framework,
            "pytorch": {"optimizer_args": {"RMSprop": {"alpha": 0.9, "eps": 1.0e-7},
                                           "Adam": {"betas": [0.9, 0.999], "eps": 1.0e-7}}},
            "keras": {"optimizer_args": {"RMSprop": {"rho": 0.9, "epsilon": 1.0e-7}}},
        },
        "trainer": {
            "epochs": [10],
            "optimizer": [{"type": "RMSprop", "learning_rate": [1.0e-5],
                           "scheduler": {"type": "none"}, "weight_decay": 0.0}],
        },
    }


def _args(cfg):
    return [p["trainer"]["optimizer"]["args"] for p in generate_param_combinations(cfg)]


def test_selected_framework_args_are_wired():
    assert _args(_cfg("keras")) == [{"rho": 0.9, "epsilon": 1.0e-7}]
    assert _args(_cfg("pytorch")) == [{"alpha": 0.9, "eps": 1.0e-7}]


def test_type_matches_case_insensitively_and_missing_type_gets_library_defaults():
    cfg = _cfg("keras")
    cfg["trainer"]["optimizer"][0]["type"] = "rmsprop"
    assert _args(cfg) == [{"rho": 0.9, "epsilon": 1.0e-7}]
    cfg["trainer"]["optimizer"][0]["type"] = "SGD"
    assert _args(cfg) == [{}]


def test_framework_sweep_gives_each_framework_its_own_args():
    cfg = _cfg(["pytorch", "keras"])
    assert _args(cfg) == [{"alpha": 0.9, "eps": 1.0e-7}, {"rho": 0.9, "epsilon": 1.0e-7}]
    assert count_all_param_combinations(cfg) == 2


def test_pairs_stay_pairs_and_lists_sweep():
    cfg = _cfg("pytorch")
    cfg["trainer"]["optimizer"][0]["type"] = "Adam"
    assert _args(cfg) == [{"betas": [0.9, 0.999], "eps": 1.0e-7}]
    assert count_all_param_combinations(cfg) == 1
    cfg = _cfg("keras")
    cfg["deep_learning"]["keras"]["optimizer_args"]["RMSprop"]["epsilon"] = [1.0e-7, 1.0e-8]
    assert [a["epsilon"] for a in _args(cfg)] == [1.0e-7, 1.0e-8]
    assert count_all_param_combinations(cfg) == 2


def test_trainer_args_are_rejected_unless_they_are_the_wired_ones():
    cfg = _cfg("keras")
    cfg["trainer"]["optimizer"][0]["args"] = {"rho": 0.5}
    with pytest.raises(ValueError, match="deep_learning.<framework>.optimizer_args"):
        list(generate_param_combinations(cfg))
    # a params_out snapshot records the wired arguments in the trainer too
    snap = _cfg("keras")
    snap["deep_learning"]["keras"]["optimizer_args"]["RMSprop"]["epsilon"] = [1.0e-7, 1.0e-8]
    snap["trainer"]["optimizer"][0]["args"] = {"rho": 0.9, "epsilon": 1.0e-8}
    assert len(_args(snap)) == 2


def test_without_optimizer_args_trainer_args_are_used_as_written():
    cfg = _cfg("keras")
    del cfg["deep_learning"]["keras"]["optimizer_args"]
    cfg["trainer"]["optimizer"][0]["args"] = {"rho": 0.5}
    assert _args(cfg) == [{"rho": 0.5}]
    no_block = copy.deepcopy(cfg)
    del no_block["deep_learning"]
    assert _args(no_block) == [{"rho": 0.5}]
