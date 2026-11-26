# services/side_channel_core/tests/unit/test_final_cv.py
"""Codebase finalisation 2026-09-29: CV protocol and training loop (lane C).

DL-03 OneCycleLR rejected under CV; SYS-26 fold seeds; SCA-09 leakage-aware
label-model guess; STAT-05 training traces are never attacked; SYS-07
an epoch error ends the run (continue_on_epoch_error removed 2026-10-23);
registry arrays beside the run metrics; CV registry diagnostics context.
"""

import copy
import json
import logging
import types

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# DL-03
# ---------------------------------------------------------------------------
def _cv_cfg(scheduler):
    return {
        "evaluation": {"validation": {"checkpointing": False}},
        "trainer": {"early_stopping": False, "optimizer": {
            "type": "SGD", "learning_rate": 0.1, "weight_decay": 0.0, "args": {},
            "scheduler": scheduler}},
    }


def test_onecyclelr_is_rejected_under_cv_with_a_hint(caplog):
    from utils.cross_validation import _validate_cv_training_contract

    one_cycle = {"type": "OneCycleLR", "args": {"max_lr": 0.1, "total_steps": 450}}
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError) as error:
        _validate_cv_training_contract(_cv_cfg(one_cycle))
    for text in ("total_steps", "45,000", "50,000", "StepLR", "CosineAnnealingLR"):
        assert text in str(error.value)
    assert any("rejected before any fold" in r.getMessage() for r in caplog.records)
    # Per-epoch schedulers and no scheduler stay valid.
    _validate_cv_training_contract(_cv_cfg({"type": "StepLR", "args": {"step_size": 1}}))
    _validate_cv_training_contract(_cv_cfg({"type": "none"}))


@pytest.mark.parametrize("schedule", [
    {"type": "CosineDecay", "args": {"decay_steps": 16875}},
    {"type": "ExponentialDecay", "args": {"decay_steps": 225, "decay_rate": 0.9}},
    {"type": "PiecewiseConstantDecay", "args": {"boundaries": [2250], "values": [1e-3, 1e-4]}},
])
def test_every_keras_schedule_is_rejected_under_cv_with_a_hint(caplog, schedule):
    """A Keras schedule advances per optimizer step; folds and refit differ in steps per epoch."""
    from utils.cross_validation import _validate_cv_training_contract

    cfg = dict(_cv_cfg(schedule), deep_learning={"framework": "keras"})
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError) as error:
        _validate_cv_training_contract(cfg)
    for text in (schedule["type"], "per optimizer step", "225", "250", "none"):
        assert text in str(error.value)
    assert any("rejected before any fold" in r.getMessage() for r in caplog.records)
    _validate_cv_training_contract(dict(_cv_cfg({"type": "none"}), deep_learning={"framework": "keras"}))


# ---------------------------------------------------------------------------
# SYS-26
# ---------------------------------------------------------------------------
def test_folds_take_the_next_seeds_and_refit_keeps_seed(tmp_path, monkeypatch):
    import utils.cross_validation as cv

    assert [cv.fold_seed(42, i) for i in range(3)] == [43, 44, 45]
    assert cv.FOLD_SEED_RULE == "refit uses seed; folds use the next n_folds seeds in fold order"
    monkeypatch.setattr(cv, "resolve_paths_dict", lambda paths, cfg, **kw: dict(paths))
    paths = types.SimpleNamespace(folds=tmp_path / "folds", fold_metrics=tmp_path / "metrics")
    cfg = {"seed": 42, "model_name": "m", "paths": {}, "cross_validation": {"n_folds": 3}}
    fold_cfg = cv.build_fold_runtime_config(cfg, paths=paths, fold_index=0, n_folds=3)
    assert fold_cfg["seed"] == 43 and fold_cfg["cross_validation_runtime"]["seed"] == 43
    refit_cfg = cv.build_refit_runtime_config(cfg, resolved_epochs=2)
    assert refit_cfg["seed"] == 42 and refit_cfg["cross_validation_runtime"]["seed"] == 42


def _fold_training_result(epochs=2):
    snapshot = {"epoch": epochs, "metrics": {"accuracy": 0.1}}
    return {"execution": {"type": "fold", "fold_index": 0}, "selection": "final_configured_epoch",
            "checkpointing": {"model_persisted": False}, "epochs_configured": epochs,
            "epochs_completed": epochs, "final_train": copy.deepcopy(snapshot),
            "final_validation": copy.deepcopy(snapshot),
            "trajectory": {"epochs": list(range(1, epochs + 1))}}


def test_fold_result_rejects_the_old_seed_rule():
    from utils.cross_validation import FoldPlan
    from utils.cross_validation_helpers import compact_fold_result

    fold = FoldPlan(fold_index=0, n_folds=2, train_indices=np.arange(2, 4),
                    validation_indices=np.arange(0, 2))
    kwargs = dict(fold=fold, universal_seed=42, validation_indices=[0, 1], profiling_count=4)
    with pytest.raises(RuntimeError, match="fold seed 42 != 43"):      # the pre-2026-09-29 rule: fold 1 = refit seed
        compact_fold_result(seed_used=42, training_result=_fold_training_result(), **kwargs)
    assert compact_fold_result(seed_used=43, training_result=_fold_training_result(), **kwargs)["seed"] == 43


def test_cv_fold_line_shows_the_rejection_reason():
    from src.sca.key_recovery import _cv_fold_line

    line = _cv_fold_line(2, 44, None, key_display="full", problem="report damaged (x.json): bad")
    assert line == "  fold 02 | seed 44 | report damaged (x.json): bad"
    assert _cv_fold_line(3, 45, None, key_display="full").endswith("report unavailable")



# ---------------------------------------------------------------------------
# SCA-09
# ---------------------------------------------------------------------------
def _labelled_h5(path, leakage_model, byte=2, n=300):
    from services.side_channel_core.src.sca.leakage_models import build_label_table

    rng = np.random.default_rng(0)
    plaintext = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    key = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    labels = build_label_table(leakage_model)[plaintext[:, byte], key[:, byte]]
    metadata = np.zeros(n, dtype=[("plaintext", np.uint8, (16,)), ("key", np.uint8, (16,))])
    metadata["plaintext"], metadata["key"] = plaintext, key
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Profiling_traces")
        group["labels"] = labels.astype(np.uint8)
        group["metadata"] = metadata
    return path


@pytest.mark.parametrize("stored", ["ID", "HW", "MSB"])
def test_label_model_guess_tests_the_leakage_models(tmp_path, stored):
    from utils.label_checks import infer_label_model

    path = _labelled_h5(tmp_path / "train.h5", stored)
    for active in ("ID", "HW", "MSB"):
        guess = infer_label_model(str(path), leakage_model=active, n=300)
        assert (guess.leakage_model, guess.byte, guess.mode) == (stored, 2, "unmasked")
        assert guess.matches == guess.checked == 300
    with pytest.raises(TypeError):
        infer_label_model(str(path), n=300)                        # leakage model is required


def test_cv_data_preparation_logs_the_active_leakage_model(tmp_path, caplog):
    from utils.cross_validation_helpers import _log_inferred_label_model

    path = _labelled_h5(tmp_path / "train.h5", "HW")
    with caplog.at_level(logging.INFO):
        _log_inferred_label_model(path, "train.h5", logging.getLogger("t"), "HW")
    message = caplog.records[-1].getMessage()
    assert "sca.leakage_model=HW" in message and "leakage_model='HW'" in message


# ---------------------------------------------------------------------------
# Training-loop harness (CPU, fake model and loaders)
# ---------------------------------------------------------------------------
N_TRACES, TRACE_LEN, N_CLASSES = 32, 8, 256


def _loop_cfg(metrics, *, epochs=2, scheduler=None, refit=False):
    cfg = {
        "seed": 0, "model_name": "m", "model_hash": "abcd1234",
        "metrics_to_log": list(metrics),
        "metrics_cfg": {"preds_are_probs": False, "ce_eps": 1.0e-12},
        "sca": {"attack_byte": 2, "leakage_model": "ID", "rank_base": 1},
        "perform_attack": {"validation": {"attack_times": 1, "shuffle": False,
                                          "attack_distribution": "full_population"},
                           "test": {"attack_times": 0}},
        "trainer": {"epochs": epochs, "early_stopping": False,
                    "loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean"}},
                    "evaluation": {"output_transformation": {"type": "Softmax"}},
                    "optimizer": {"type": "SGD", "learning_rate": 0.1, "weight_decay": 0.0,
                                  "args": {}, "scheduler": scheduler or {"type": "none"}}},
        "paths": {"metrics": "metrics"},
    }
    if refit:
        cfg["cross_validation"] = {"enabled": True, "n_folds": 2}
        cfg["cross_validation_runtime"] = {"active": True, "is_fold": False, "is_refit": True,
                                           "n_folds": 2, "resolved_refit_epochs": epochs,
                                           "seed": 0}
    return cfg


class _Recorder:
    def __init__(self):
        self.rows, self.manifest, self.saved, self.metric_calls = [], None, None, []


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import utils.epoch_utils as eu

    rec = _Recorder()
    artifacts = types.SimpleNamespace(final_epoch=tmp_path / "final.pt",
                                      best_validation=tmp_path / "best.pt",
                                      epoch_checkpoint=lambda epoch: tmp_path / f"e{epoch}.pt")

    def save_model(*, model, path):
        rec.saved = {k: v.clone() for k, v in model.state_dict().items()}
        return path

    def manifest(**kwargs):
        rec.manifest = kwargs
        return tmp_path / "manifest.json"

    monkeypatch.setattr(eu, "set_seed", lambda cfg: torch.manual_seed(0) and 0)
    monkeypatch.setattr(eu, "validate_key_recovery_contract", lambda cfg: {"enabled": False})
    monkeypatch.setattr(eu, "get_validation_checkpointing", lambda cfg: False)
    monkeypatch.setattr(eu, "get_monitor_metric_and_mode", lambda cfg: (None, "minimize"))
    monkeypatch.setattr(eu, "get_requested_test_policy", lambda cfg: "final_epoch")
    monkeypatch.setattr(eu, "clear_checkpoint_artifacts", lambda cfg, model_name: artifacts)
    monkeypatch.setattr(eu, "assert_head_sanity", lambda *a, **k: None)
    monkeypatch.setattr(eu, "print_blocked_results", lambda **k: None)
    monkeypatch.setattr(eu, "append_metrics_row",
                        lambda cfg, results, process, epoch: rec.rows.append((process, epoch, dict(results))))
    monkeypatch.setattr(eu, "_log_results_to_clearml", lambda *a, **k: None)
    monkeypatch.setattr(eu, "_maybe_upload_file_to_clearml", lambda *a, **k: None)
    monkeypatch.setattr(eu, "_save_model_atomically", save_model)
    monkeypatch.setattr(eu, "write_checkpoint_manifest", manifest)
    return eu, rec


def _loader(n=N_TRACES):
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(n, TRACE_LEN, generator=generator)
    y = torch.randint(0, N_CLASSES, (n,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def _stack(cfg):
    from utils.optimizer_util import build_optimizer_from_config

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(TRACE_LEN, N_CLASSES))
    optimizer, scheduler, _ = build_optimizer_from_config(cfg["trainer"], model.parameters())
    return model, optimizer, scheduler


def _run(eu, cfg, model, optimizer, scheduler, tmp_path, val_loader=None):
    return eu.run_training_loop(
        model=model, device=torch.device("cpu"), optimizer=optimizer, scheduler=scheduler,
        train_loader=_loader(), val_loader=val_loader if val_loader is not None else _loader(),
        cfg=cfg, logger=logging.getLogger("t"), base_dir=tmp_path, model_name="m", env="dev",
    )


def _count_steps(optimizer):
    steps = []
    optimizer.register_step_post_hook(lambda *args: steps.append(1))
    return steps


# ---------------------------------------------------------------------------
# SYS-07
# ---------------------------------------------------------------------------
def test_a_failed_evaluation_ends_the_run_and_publishes_nothing(harness, tmp_path, monkeypatch):
    """An epoch whose evaluation fails after its optimisation pass is never skipped: the run fails."""
    eu, rec = harness
    real = eu.compute_metrics
    validation_calls = []

    def compute_metrics(preds, labels, cfg, metadata=None, process=None):
        if process == "validation":
            validation_calls.append(1)
            if len(validation_calls) == 2:                          # epoch 2, after its optimiser pass
                raise OSError("disk full while evaluating")
        return real(preds, labels, cfg, metadata=metadata, process=process)

    monkeypatch.setattr(eu, "compute_metrics", compute_metrics)
    cfg = _loop_cfg(["accuracy"], scheduler={"type": "StepLR", "args": {"step_size": 1, "gamma": 0.5}})
    with pytest.raises(OSError, match="disk full while evaluating"):
        _run(eu, cfg, *_stack(cfg), tmp_path)
    assert rec.saved is None and rec.manifest is None               # no final model, no manifest


def test_a_failed_validation_objective_ends_the_run(harness, tmp_path, monkeypatch):
    eu, _rec = harness
    monkeypatch.setattr(eu, "_mean_validation_objective",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    cfg = _loop_cfg(["accuracy"])
    with pytest.raises(OSError, match="boom"):
        _run(eu, cfg, *_stack(cfg), tmp_path)


def test_reduce_lr_on_plateau_steps_on_the_validation_objective(harness, tmp_path, monkeypatch):
    """The configured loss, label_smoothing included (plain cross-entropy before 2026-10-20)."""
    eu, _rec = harness
    real = eu._mean_validation_objective
    objectives = []

    def recorded(*args, **kwargs):
        objectives.append(real(*args, **kwargs))
        return objectives[-1]

    monkeypatch.setattr(eu, "_mean_validation_objective", recorded)
    cfg = _loop_cfg(["accuracy"], scheduler={"type": "ReduceLROnPlateau", "args": {"patience": 0}})
    cfg["trainer"]["loss"]["args"]["label_smoothing"] = 0.1
    model, optimizer, scheduler = _stack(cfg)
    stepped = []
    step = scheduler.step
    scheduler.step = lambda metric=None: (stepped.append(metric), step(metric))
    _run(eu, cfg, model, optimizer, scheduler, tmp_path)
    assert stepped == objectives and len(stepped) == 2


class _NaNLoss(torch.nn.Module):
    def forward(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels) * float("nan")


class _BrokenLoader:
    """Yields one batch, then fails in the middle of the optimisation pass."""

    def __init__(self):
        self.inner = _loader()
        self.dataset = self.inner.dataset

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        iterator = iter(self.inner)
        yield next(iterator)
        raise RuntimeError("HDF5 read failed mid-epoch")


@pytest.mark.parametrize(
    "failure", ["nan_loss", "nan_weights_after_last_step", "broken_batch", "non_finite_after_optimisation"]
)
def test_failed_optimisation_pass_is_never_swallowed(harness, tmp_path, monkeypatch, failure, caplog):
    eu, rec = harness
    cfg = _loop_cfg(["accuracy"], epochs=1)
    model, optimizer, scheduler = _stack(cfg)
    train_loader = _loader()
    if failure == "nan_loss":
        monkeypatch.setattr(eu, "build_loss", lambda cfg: _NaNLoss())
        expected, match = FloatingPointError, "Training loss is NaN"
    elif failure == "nan_weights_after_last_step":
        # The loss of every batch is finite; the last step breaks the weights.
        steps = _count_steps(optimizer)
        optimizer.register_step_post_hook(
            lambda *args: model[0].weight.data.fill_(float("nan")) if len(steps) == len(train_loader) else None
        )
        expected, match = FloatingPointError, "weights are NaN"
    elif failure == "non_finite_after_optimisation":
        # Finite weights, but NaN/Inf found while evaluating them (key scores,
        # overflowing activations): a broken model, not a failed evaluation.
        def collect(*args):
            raise FloatingPointError("Accumulated key scores became non-finite")

        monkeypatch.setattr(eu, "collect_logits_labels_meta", collect)
        expected, match = FloatingPointError, "key scores became non-finite"
    else:
        train_loader = _BrokenLoader()
        expected, match = RuntimeError, "HDF5 read failed mid-epoch"
    with caplog.at_level(logging.ERROR), pytest.raises(expected, match=match):
        eu.run_training_loop(
            model=model, device=torch.device("cpu"), optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=_loader(), cfg=cfg, logger=logging.getLogger("t"),
            base_dir=tmp_path, model_name="m", env="dev",
        )
    assert rec.saved is None and rec.manifest is None
    assert any("Epoch 1 failed during" in r.getMessage() for r in caplog.records)


def _best_validation(eu, monkeypatch, objectives):
    """Checkpointing on, best_validation tested, validation_objective monitored."""
    values = iter(objectives)
    monkeypatch.setattr(eu, "get_validation_checkpointing", lambda cfg: True)
    monkeypatch.setattr(eu, "get_monitor_metric_and_mode", lambda cfg: ("validation_objective", "minimize"))
    monkeypatch.setattr(eu, "get_requested_test_policy", lambda cfg: "best_validation")
    monkeypatch.setattr(eu, "_mean_validation_objective", lambda *a, **k: next(values))


def _torch_save(*, model, path):
    torch.save(model.state_dict(), path)
    return path


@pytest.mark.parametrize("failure", ["best_checkpoint_save", "after_new_best"])
def test_a_failure_while_saving_or_after_a_new_best_epoch_ends_the_run(harness, tmp_path, monkeypatch, failure):
    """No epoch is skipped: a failed best-checkpoint save or a failure after a new best epoch ends the run,
    and no final model or manifest is published."""
    eu, rec = harness
    _best_validation(eu, monkeypatch, [3.0, 2.0, 1.0, 5.0])        # epoch 3 would be the best

    def save(*, model, path):
        if failure == "best_checkpoint_save" and path.name == "e3.pt":
            raise OSError("disk full")
        return _torch_save(model=model, path=path)

    def print_block(**kwargs):
        if failure == "after_new_best" and kwargs["epoch"] == 3:    # best.pt already holds epoch 3
            raise OSError("terminal gone")

    monkeypatch.setattr(eu, "_save_model_atomically", save)
    monkeypatch.setattr(eu, "print_blocked_results", print_block)
    cfg = _loop_cfg(["accuracy"], epochs=4)
    with pytest.raises(OSError, match="disk full" if failure == "best_checkpoint_save" else "terminal gone"):
        _run(eu, cfg, *_stack(cfg), tmp_path)
    assert rec.manifest is None


def test_a_failure_after_validation_best_report_ends_the_run(harness, tmp_path, monkeypatch, caplog):
    eu, rec = harness
    _best_validation(eu, monkeypatch, [3.0, 2.0])
    monkeypatch.setattr(eu, "_save_model_atomically", _torch_save)
    # Key-recovery reporting on: every improving epoch replaces validation_best.json.
    real_collect, real_metrics = eu.collect_logits_labels_meta, eu.compute_metrics
    monkeypatch.setattr(eu, "validate_key_recovery_contract", lambda cfg: {"enabled": True})
    monkeypatch.setattr(eu, "_needs_rank_metrics", lambda cfg: True)
    monkeypatch.setattr(eu, "collect_logits_labels_meta", lambda *a: (*real_collect(*a)[:2], "meta"))
    monkeypatch.setattr(eu, "_assert_metadata_alignment", lambda **k: None)

    def compute_metrics(preds, labels, cfg, metadata=None, process=None):
        if process == "validation":
            return {"accuracy": 0.5}
        return real_metrics(preds, labels, cfg, metadata=metadata, process=process)

    def print_block(**kwargs):
        if kwargs["epoch"] == 2:                                    # epoch 2, validation_best.json replaced
            raise RuntimeError("failed after the report was replaced")

    monkeypatch.setattr(eu, "compute_metrics", compute_metrics)
    monkeypatch.setattr(eu, "pop_key_recovery_evidence", lambda results: "evidence")
    monkeypatch.setattr(eu, "build_key_recovery_report", lambda **k: {"model": {}})
    monkeypatch.setattr(eu, "save_validation_report", lambda **k: {
        "validation_history": None, "validation_best": tmp_path / "validation_best.json"})
    monkeypatch.setattr(eu, "print_blocked_results", print_block)
    cfg = _loop_cfg(["accuracy"], epochs=3)
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="failed after the report was replaced"):
        _run(eu, cfg, *_stack(cfg), tmp_path)
    assert rec.manifest is None
    assert any("Epoch 2 failed during" in r.getMessage() for r in caplog.records)


def test_the_test_loop_records_its_metrics_row_and_nothing_else(harness, tmp_path):
    from utils.checkpoint_policy import file_sha256

    eu, rec = harness
    cfg = _loop_cfg(["accuracy"], epochs=1)
    model, optimizer, scheduler = _stack(cfg)
    _run(eu, cfg, model, optimizer, scheduler, tmp_path)
    checkpoint, manifest = tmp_path / "tested.pt", tmp_path / "tested_manifest.json"
    _torch_save(model=model, path=checkpoint)
    manifest.write_text("{}")
    provenance = {"requested_policy": "final_epoch", "effective_policy": "final_epoch",
                  "validation_checkpointing": False, "checkpoint_path": str(checkpoint),
                  "checkpoint_epoch": 1, "checkpoint_sha256": file_sha256(checkpoint),
                  "manifest_path": str(manifest), "fallback_reason": None,
                  "selection_metric": None, "selection_mode": None, "selection_score": None}
    eu.run_test_loop(model=model, device=torch.device("cpu"), test_loader=_loader(), cfg=cfg,
                     logger=logging.getLogger("t"), base_dir=tmp_path, model_name="m", env="dev",
                     checkpoint_provenance=provenance)
    # The run folder is the only record: one test row, no index beside the run.
    test_rows = [(epoch, row) for process, epoch, row in rec.rows if process == "test"]
    assert len(test_rows) == 1 and test_rows[0][0] == 1 and "accuracy" in test_rows[0][1]
    assert not (tmp_path / "bench").exists()


# ---------------------------------------------------------------------------
# STAT-05
# ---------------------------------------------------------------------------
_RANK_REQUEST = ["accuracy", "mean_attack_rank_curve", "guessing_entropy", "traces_to_ge_1"]
_RANK_NAMES = {"mean_attack_rank_curve", "guessing_entropy", "traces_to_ge_1", "key_rank",
               "success_rate", "full_ranks"}


def test_refit_never_attacks_its_training_traces(harness, tmp_path, monkeypatch):
    eu, rec = harness
    real = eu.compute_metrics
    processes = []

    def compute_metrics(preds, labels, cfg, metadata=None, process=None):
        processes.append((process, metadata))
        return real(preds, labels, cfg, metadata=metadata, process=process)

    monkeypatch.setattr(eu, "compute_metrics", compute_metrics)
    cfg = _loop_cfg(_RANK_REQUEST, refit=True)
    result = _run(eu, cfg, *_stack(cfg), tmp_path)

    assert {process for process, _ in processes} == {"train"}
    assert all(metadata is None for _, metadata in processes)
    assert result["epochs_completed"] == 2 and "final_validation" not in result
    assert result["validation"] == {} and not _RANK_NAMES & set(result["train"])
    assert isinstance(result["train"]["accuracy"], float)           # eval-mode accuracy still recorded
    for process, _epoch, row in rec.rows:
        assert process == "train" and not _RANK_NAMES & set(row)


def test_standard_training_never_attacks_training_traces(harness, tmp_path, monkeypatch):
    eu, _rec = harness
    real = eu.compute_metrics
    calls = []

    def compute_metrics(preds, labels, cfg, metadata=None, process=None):
        calls.append((process, cfg.get("metrics_to_log"), metadata))
        return real(preds, labels, cfg, metadata=None if process == "validation" else metadata,
                    process=process)

    monkeypatch.setattr(eu, "compute_metrics", compute_metrics)
    monkeypatch.setattr(eu, "_process_attacked", lambda cfg, process: False)
    cfg = _loop_cfg(_RANK_REQUEST)
    _run(eu, cfg, *_stack(cfg), tmp_path)
    train_calls = [c for c in calls if c[0] == "train"]
    assert train_calls and all(metrics == ["accuracy"] and meta is None for _, metrics, meta in train_calls)


def test_refit_evaluation_loader_carries_no_metadata(monkeypatch):
    import utils.data_split as data_split

    seen = {}

    def dataset(cfg, *, return_metadata=False, train_path=None):
        seen["return_metadata"] = return_metadata
        return TensorDataset(torch.zeros(4, TRACE_LEN), torch.zeros(4, dtype=torch.long))

    monkeypatch.setattr(data_split, "get_cross_validation_dataset", dataset)
    monkeypatch.setattr(data_split, "_get_dataloader_cfg", lambda cfg: (
        {"pin_memory": False, "persistent_workers": False, "prefetch_factor": None}, {"val": 0}))
    monkeypatch.setattr(data_split, "_make_loader", lambda **kwargs: kwargs["dataset"])
    monkeypatch.setattr(data_split, "_resolve_training_batch_size", lambda trainer_cfg: 2)
    data_split.get_full_profiling_eval_loader({"cross_validation": {"enabled": True}}, {}, seed=0)
    assert seen["return_metadata"] is False


def test_refit_population_cannot_be_attacked():
    from src.sca.perform_attack import _resolve_evaluation_population

    cfg = {"cross_validation": {"enabled": True},
           "cross_validation_runtime": {"active": True, "is_fold": False, "is_refit": True,
                                        "n_folds": 10, "profiling_count": 4,
                                        "validation_indices": [0, 1, 2, 3]}}
    with pytest.raises(ValueError, match="never attacked"):
        _resolve_evaluation_population(cfg, process="validation", n_available=4)


def _judge_project(tmp_path, ranking):
    root = tmp_path / "projects" / "P"
    for sub in ("artifacts", "config"):
        (root / sub).mkdir(parents=True)
    (root / "config" / "model_judge.yaml").write_text(yaml.safe_dump({"model_judge": {
        "ranking": ranking, "filters": {"require_done": True, "require_processes": ["train"]}}}))
    runs, tests = {}, {}
    for name, digest, accuracy in (("a", "aaaa0001", 0.9), ("b", "bbbb0002", 0.5)):
        # The run folder is what the judge reads: the train row of the tested (final) epoch.
        run = root / "artifacts" / "dev" / "generated_models" / name
        (run / "model").mkdir(parents=True)
        (run / "model" / f"{name}_checkpoint_manifest.json").write_text(json.dumps(
            {"validation_checkpointing": False, "selection": None, "final": {"epoch": 1}}))
        scalars = run / "metrics" / "attack_metrics" / "scalars"
        scalars.mkdir(parents=True)
        (scalars / f"{name}_metrics_train_history.jsonl").write_text(json.dumps(
            {"epoch": 1, "accuracy": accuracy, "dependencies": {"fit_time_s": 10.0 * accuracy}}) + "\n")
        runs[digest] = {"model_name": name, "status": "DONE", "env": "dev", "model_id": "M",
                        "path": f"P/artifacts/dev/generated_models/{name}"}
        tests[digest] = {"model_name": name, "status": "DONE", "env": "dev"}
    (root / "artifacts" / "run_registry.json").write_text(json.dumps(runs))
    (root / "artifacts" / "test_registry.json").write_text(json.dumps(tests))
    return root


def test_model_judge_flags_train_quality_metrics(tmp_path, caplog):
    from services.model_judge.model_judge import run_model_judge

    root = _judge_project(tmp_path, {"primary": {"process": "train", "metric": "accuracy"},
                                     "tie_breakers": [{"process": "train", "metric": "fit_time_s"}]})
    with caplog.at_level(logging.WARNING, logger="ModelJudge"):
        result = run_model_judge(root / "config" / "model_judge.yaml", project="P", env="dev",
                                 projects_root=tmp_path / "projects")
    assert result["selection_uses_train_quality_metrics"] == ["train.accuracy"]
    assert any("training-set quality metrics" in r.getMessage() for r in caplog.records)
    saved = json.loads((root / "model_selection" / "latest_selection.json").read_text())
    assert saved["selection_uses_train_quality_metrics"] == ["train.accuracy"]

    # A cost such as fit time is a legitimate train metric: no flag.
    root = _judge_project(tmp_path / "cost", {"primary": {"process": "train", "metric": "fit_time_s"}})
    result = run_model_judge(root / "config" / "model_judge.yaml", project="P", env="dev",
                             projects_root=tmp_path / "cost" / "projects")
    assert result["selection_uses_train_quality_metrics"] == []


# ---------------------------------------------------------------------------
# TODO.md §22: disjoint validation attacks under cross-validation
# ---------------------------------------------------------------------------
def _disjoint_plan_and_folds(attack_times, distribution, curve_end):
    import utils.cross_validation as cv

    n_folds, held, epochs = 2, 10, 3
    fold_plans = [types.SimpleNamespace(fold_index=i, validation_indices=np.arange(i * held, (i + 1) * held),
                                        train_indices=np.arange((1 - i) * held, (2 - i) * held)) for i in range(n_folds)]
    plan = types.SimpleNamespace(spec=types.SimpleNamespace(n_folds=n_folds, seed=42, partition=42), folds=fold_plans,
                                 paths=types.SimpleNamespace(aggregate="/a", fold_indices="/a/fold_indices.npz"))
    cfg = {"trainer": {"epochs": epochs}, "sca": {"attack_byte": 2},
           "metrics_to_log": ["mean_attack_rank_curve"],
           "perform_attack": {"validation": {"attack_times": attack_times, "shuffle": False,
                                             "attack_distribution": distribution}, "ge_thresholds": 1}}
    curve = [[float(n), float(max(1, 50 - 10 * n))] for n in range(1, curve_end + 1)]
    results = [{"fold_index": f.fold_index, "fold_number": f.fold_index + 1, "n_folds": n_folds,
                "seed": cv.fold_seed(42, f.fold_index), "epochs_completed": epochs, "epochs_configured": epochs,
                "final_validation": {"epoch": epochs, "metrics": {"mean_attack_rank_curve": curve},
                                     "population_source": "Profiling_traces", "validation_indices": f.validation_indices,
                                     "profiling_count": 2 * held}} for f in fold_plans]
    return cv, plan, cfg, results


def test_a_disjoint_fold_curve_ends_at_held_out_over_attack_times_and_is_aggregated():
    cv, plan, cfg, results = _disjoint_plan_and_folds(attack_times=2, distribution="disjoint", curve_end=5)
    aggregate = cv._aggregate_final_fold_curves(plan, cfg, results)
    assert aggregate["status"] == "completed" and aggregate["traces_per_attack"] == 5
    assert aggregate["attack_distribution"]["attack_distribution"] == "disjoint"
    assert aggregate["fold_rank_curves"]["trace_counts"][-1] == 5


def test_a_full_population_fold_curve_must_still_reach_the_whole_held_out_part():
    cv, plan, cfg, results = _disjoint_plan_and_folds(attack_times=1, distribution="full_population", curve_end=5)
    with pytest.raises(ValueError, match="curve must end at 10 traces"):
        cv._aggregate_final_fold_curves(plan, cfg, results)


def test_disjoint_attacks_that_do_not_divide_the_held_out_part_are_refused_with_a_hint():
    import utils.cross_validation as cv

    _cv, _plan, cfg, _results = _disjoint_plan_and_folds(attack_times=3, distribution="disjoint", curve_end=3)
    with pytest.raises(cv.CrossValidationError, match="attack_times=3 does not divide the held-out fold size 10"):
        cv.validation_traces_per_attack(cfg, 10)
    assert cv.validation_traces_per_attack(_disjoint_plan_and_folds(5, "disjoint", 2)[2], 10) == 2


@pytest.mark.parametrize("listed", [False, True])
def test_the_run_records_the_loss_only_when_metrics_to_log_lists_it(harness, tmp_path, monkeypatch, listed):
    """Owner 2026-10-27: the training result (fold_result.yaml, the CV aggregate) and its trajectory keep what
    metrics_to_log names and its dependencies; an unlisted loss is kept nowhere, the heartbeat included."""
    eu, rec = harness
    from utils import heartbeat

    beats = []
    monkeypatch.setattr(heartbeat, "update", lambda **kw: beats.append(kw))
    cfg = _loop_cfg(["mean_accuracy"] + (["training_objective", "validation_objective"] if listed else []))
    result = _run(eu, cfg, *_stack(cfg), tmp_path)
    train, validation = result["final_train"]["metrics"], result["final_validation"]["metrics"]
    assert ("training_objective" in train, "validation_objective" in validation) == (listed, listed)
    assert ("loss_type" in train) == listed and "train_loss_reduction" not in train
    assert {"mean_accuracy", "accuracy"} <= set(train) and {"mean_accuracy", "accuracy"} <= set(validation)
    assert ({"training_objective", "validation_objective"} <= set(result["trajectory"])) == listed
    assert {"epochs", "training_accuracy", "validation_accuracy"} <= set(result["trajectory"])
    assert not any("objective" in beat for beat in beats)
