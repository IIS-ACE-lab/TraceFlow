# services/side_channel_core/tests/unit/test_one_cycle_loop.py
"""OneCycleLR with each training loop's own length (owner ruling 2026-10-22): torch's own class and
arguments; the loop gives epochs and steps_per_epoch, so folds, the refit and a hold-out training each
finish their cycle. CyclicLR and a written length are refused under cross-validation; early stopping
is refused with OneCycleLR; Keras has no one-cycle schedule."""

import logging

import pytest
import torch

from test_final_cv import N_TRACES, _cv_cfg, _loader, _loop_cfg

MAX_LR, DIV, FINAL_DIV = 0.1, 10.0, 100.0
ONE_CYCLE = {"type": "OneCycleLR", "args": {"max_lr": MAX_LR, "div_factor": DIV, "final_div_factor": FINAL_DIV,
                                            "anneal_strategy": "linear", "three_phase": True,
                                            "cycle_momentum": False}}
FLOOR = MAX_LR / DIV / FINAL_DIV


def _train(eu, cfg, n_traces, tmp_path):
    from utils.optimizer_util import build_optimizer_from_config

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 256))
    optimizer, scheduler, metadata = build_optimizer_from_config(cfg["trainer"], model.parameters())
    used = []                                                   # the learning rate of every optimizer step
    optimizer.register_step_pre_hook(lambda opt, args, kwargs: used.append(opt.param_groups[0]["lr"]))
    eu.run_training_loop(model=model, device=torch.device("cpu"), optimizer=optimizer, scheduler=scheduler,
                         train_loader=_loader(n_traces), val_loader=_loader(), cfg=cfg,
                         logger=logging.getLogger("t"), base_dir=tmp_path, model_name="m", env="dev")
    return scheduler, metadata, used


@pytest.fixture
def loop(tmp_path, monkeypatch):
    """utils.epoch_utils with its artifact writers stubbed, as test_final_cv's harness does."""
    import types

    import utils.epoch_utils as eu

    artifacts = types.SimpleNamespace(final_epoch=tmp_path / "final.pt", best_validation=tmp_path / "best.pt",
                                      epoch_checkpoint=lambda epoch: tmp_path / f"e{epoch}.pt")
    monkeypatch.setattr(eu, "set_seed", lambda cfg: torch.manual_seed(0) and 0)
    monkeypatch.setattr(eu, "validate_key_recovery_contract", lambda cfg: {"enabled": False})
    monkeypatch.setattr(eu, "get_validation_checkpointing", lambda cfg: False)
    monkeypatch.setattr(eu, "get_monitor_metric_and_mode", lambda cfg: (None, "minimize"))
    monkeypatch.setattr(eu, "get_requested_test_policy", lambda cfg: "final_epoch")
    monkeypatch.setattr(eu, "clear_checkpoint_artifacts", lambda cfg, model_name: artifacts)
    monkeypatch.setattr(eu, "assert_head_sanity", lambda *a, **k: None)
    monkeypatch.setattr(eu, "print_blocked_results", lambda **k: None)
    monkeypatch.setattr(eu, "append_metrics_row", lambda cfg, results, process, epoch: None)
    monkeypatch.setattr(eu, "_log_results_to_clearml", lambda *a, **k: None)
    monkeypatch.setattr(eu, "_maybe_upload_file_to_clearml", lambda *a, **k: None)
    monkeypatch.setattr(eu, "_save_model_atomically", lambda *, model, path: path)
    monkeypatch.setattr(eu, "write_checkpoint_manifest", lambda **kwargs: tmp_path / "manifest.json")
    return eu


def _torch_cycle(total_steps):
    """The learning rate of each of total_steps optimizer steps under torch's OneCycleLR, stepped by hand."""
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **ONE_CYCLE["args"], total_steps=total_steps)
    rates = []
    for _ in range(total_steps):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return rates


# ------------------------------------------------------------------ building it
def test_the_optimizer_builder_leaves_a_lengthless_one_cycle_to_the_loop():
    from utils.optimizer_util import build_optimizer_from_config

    model = torch.nn.Linear(2, 2)
    trainer = _loop_cfg([], scheduler=ONE_CYCLE)["trainer"]
    _optimizer, scheduler, metadata = build_optimizer_from_config(trainer, model.parameters())
    assert scheduler is None
    assert metadata["scheduler"]["length"] == "trainer.epochs x len(train_loader) of each training loop"
    # every other scheduler, and a OneCycleLR with a written length, is built with the optimizer as before
    for spec in ({"type": "StepLR", "args": {"step_size": 1}},
                 {"type": "OneCycleLR", "args": {"max_lr": 0.1, "total_steps": 8}}):
        _o, built, meta = build_optimizer_from_config(_loop_cfg([], scheduler=spec)["trainer"], model.parameters())
        assert type(built).__name__ == spec["type"] and "length" not in meta["scheduler"]


def test_the_loop_scheduler_is_torchs_one_cycle_with_the_loops_length():
    from utils.optimizer_util import build_loop_scheduler

    optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
    trainer = _loop_cfg([], scheduler=ONE_CYCLE)["trainer"]
    scheduler = build_loop_scheduler(optimizer, trainer, epochs=3, steps_per_epoch=225)
    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) and scheduler.total_steps == 675
    assert build_loop_scheduler(optimizer, _loop_cfg([])["trainer"], epochs=3, steps_per_epoch=225) is None


# ------------------------------------------------------------- training with it
@pytest.mark.parametrize("n_traces, batches", [(24, 3), (32, 4)])          # a fold and a larger refit-sized loop
def test_every_loop_completes_its_own_cycle_to_the_floor(loop, tmp_path, n_traces, batches):
    eu = loop
    cfg = _loop_cfg(["accuracy"], epochs=2, scheduler=ONE_CYCLE)
    scheduler, _meta, used = _train(eu, cfg, n_traces, tmp_path)
    assert scheduler is None                                    # built inside the loop, not by the builder
    assert len(used) == 2 * batches                             # torch raises past total_steps; none was
    assert used == pytest.approx(_torch_cycle(2 * batches), rel=1e-12)   # torch's own cycle of this length
    assert used[-1] == pytest.approx(FLOOR, rel=1e-9)           # the last step trained at the annealed floor


def test_the_cv_refit_loop_builds_its_own_cycle_too(loop, tmp_path):
    eu = loop
    cfg = _loop_cfg(["accuracy"], epochs=2, scheduler=ONE_CYCLE, refit=True)
    _s, _m, used = _train(eu, cfg, N_TRACES, tmp_path)
    assert len(used) == 2 * (N_TRACES // 8) and used[-1] == pytest.approx(FLOOR, rel=1e-9)


def test_a_written_length_that_misses_the_loop_is_still_refused(loop, tmp_path):
    eu = loop
    written = {"type": "OneCycleLR", "args": {"max_lr": 0.1, "total_steps": 7}}
    with pytest.raises(ValueError, match="OneCycleLR.total_steps=7 but the loop performs 8"):
        _train(eu, _loop_cfg(["accuracy"], epochs=2, scheduler=written), N_TRACES, tmp_path)


def test_early_stopping_is_refused_with_one_cycle():
    from utils.optimizer_util import check_scheduler_training

    trainer = _loop_cfg([], scheduler=ONE_CYCLE)["trainer"]
    check_scheduler_training(trainer)
    with pytest.raises(ValueError, match="early_stopping cannot be used with OneCycleLR"):
        check_scheduler_training({**trainer, "early_stopping": True})
    check_scheduler_training({**_loop_cfg([])["trainer"], "early_stopping": True})   # other schedulers: as before


# ------------------------------------------------------------- cross-validation
def test_cross_validation_takes_one_cycle_without_a_length(caplog):
    from utils.cross_validation import _validate_cv_training_contract

    _validate_cv_training_contract(_cv_cfg(ONE_CYCLE))
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError) as error:
        _validate_cv_training_contract(_cv_cfg({"type": "OneCycleLR", "args": {"max_lr": 0.1, "epochs": 2,
                                                                             "steps_per_epoch": 225}}))
    assert "['epochs', 'steps_per_epoch']" in str(error.value) and "remove total_steps" in str(error.value)
    assert any("rejected before any fold" in r.getMessage() for r in caplog.records)


def test_cross_validation_refuses_cyclic_lr(caplog):
    from utils.cross_validation import _validate_cv_training_contract

    cyclic = {"type": "CyclicLR", "args": {"base_lr": 1e-4, "max_lr": 1e-3, "step_size_up": 2000}}
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError) as error:
        _validate_cv_training_contract(_cv_cfg(cyclic))
    for text in ("CyclicLR", "optimizer steps", "225", "250", "without cross-validation"):
        assert text in str(error.value)


# ------------------------------------------------------------------------- Keras
def test_keras_refuses_one_cycle_at_build_and_under_cross_validation():
    from core.framework_switch import _keras_schedule
    from utils.cross_validation import _validate_cv_training_contract

    with pytest.raises(ValueError, match="'OneCycleLR' is not a keras.optimizers.schedules class: it is a "
                                         "torch.optim.lr_scheduler class, available under deep_learning.framework: pytorch"):
        _keras_schedule({"type": "OneCycleLR", "args": {"max_lr": 0.01}}, 1e-3)
    with pytest.raises(ValueError, match="'OneCycleLR' is not a keras.optimizers.schedules class: it is a torch"):
        _validate_cv_training_contract(dict(_cv_cfg(ONE_CYCLE), deep_learning={"framework": "keras"}))
    with pytest.raises(ValueError, match="is not a keras.optimizers.schedules class"):
        _keras_schedule({"type": "NoSuchSchedule", "args": {}}, 1e-3)              # a misspelling: the old message
