# services/side_channel_core/utils/epoch_utils.py
"""
Epoch Utilities for Side-Channel Analysis (SCA)

This module provides training/validation/test loops for SCA models with:
- Correct handling of metadata for SCA rank/GE metrics (validation/test only)
- Reproducibility hooks using a single cfg["seed"] source
- Rank-eval throttling:
    cfg["validation"]["rank_eval_interval"] (default=1)
  Rank metrics are evaluated only on scheduled epochs unless metric monitoring
  is actively needed for checkpointing or early stopping.
- Train metrics come from the optimisation pass itself, with no second pass over
  train_loader; the CV refit alone, having no held-out traces, makes one
  eval-mode pass over its training population per epoch (training-set accuracy
  and loss, never a selection signal, outside the fit time)
- Optional ClearML logging through a centralized tracker
- Configurable validation checkpointing independent of early stopping.
- CV folds with validation checkpointing disabled persist assessment data only.
- Standard training and the mandatory CV refit publish final model artifacts.
- The run's own folder is the only record: metric history rows, array files,
  key-recovery reports, the checkpoint manifest. No separate index is written.

SCA-fit gates enforced:
1) Rank metrics require metadata in validation/test
2) Determinism uses cfg["seed"]
3) Rank evaluation uses held-out profiling folds or dedicated evaluation loaders
4) Explicit metadata alignment checks
5) No extra train dataloader pass for train metrics (the CV refit's eval-mode
   pass over its training population is the one exception, above)
6) Clear logging and safe fallbacks
7) Train metrics are kept classification/loss-safe; rank/GE metrics are stripped
   from train because train normally has no attack metadata
8) Early stopping, checkpointing and metric logging are treated as separate concerns
 
Author: Shraddha Mishra

"""


from __future__ import annotations

import gc
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union
from numbers import Integral
import torch
import torch.nn.functional as F
from core.framework_switch import build_loss, cross_entropy_per_trace, gradient_clipper, resolve_framework
from utils import heartbeat
from services.side_channel_core.src.sca.leakage_models import (
    get_num_classes,
    resolve_runtime_leakage_model,
)
from src.sca.metrics import (
    compute_metrics,
    fit_time_values,
    loop_state_values,
    loss_signature,
    model_size_values,
    eta_t,
    pop_key_recovery_evidence,
    requested_fit_time_metrics,
    requested_loop_state_metrics,
    requested_model_size_metrics,
    split_requested,
)
from services.visualization.clearml.clearml_tracker import ClearMLTracker
from utils.checkpoint_policy import (
    CheckpointPolicyError,
    clear_checkpoint_artifacts,
    checkpoint_content_sha256,
    file_sha256,
    get_requested_test_policy,
    get_validation_checkpointing,
    write_checkpoint_manifest,
)
from utils.cross_validation import fold_artifact_name, resolve_cross_validation_runtime
from utils.h5_metadata_utils import (
    _maybe_sync_metrics_key,
    _monitor_is_rank_metric,
    _needs_rank_metrics,
    _rank_eval_interval,
    _strip_rank_metrics,
    collect_logits_labels_meta,
    filter_metrics_for_metadata,
    unpack_batch,
)
from src.sca.perform_attack import attack_plan
from src.sca.dataset_types import random_profiling_keys
from src.sca.key_recovery import (
    KeyRecoveryError,
    build_key_recovery_report,
    save_test_report,
    save_validation_report,
    validate_key_recovery_contract,
)
from utils.logger import get_logger, print_blocked_results
from utils.metrics_logger import append_metrics_row
from utils.monitor_metric import check_monitor_metric, get_monitor_metric_and_mode, is_better_score
from utils.optimizer_util import (
    build_loop_scheduler,
    check_scheduler_training,
    resolve_scheduler_interval,
    validate_scheduler_for_loop,
)
from utils.save_utils import save_model
from utils import attack_progress
from utils.validation_judge import ValidationJudge, judge_log_enabled
from utils.seal import seal_zip_file
from utils.seed import set_seed

logger = get_logger("Epoch")


def _strict_bool(value: Any, *, field: str) -> bool:
    """Return one strict boolean and reject truthy strings."""
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be true or false, got {value!r}")
    return value


def _finite_float(value: Any, *, field: str) -> float:
    """Return one finite scalar metric value."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric, got {value!r}") from exc
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{field} must be finite, got {result!r}")
    return result


def _optional_metric(results: Mapping[str, Any], *names: str) -> float | None:
    """Return the first finite scalar metric found under the supplied names."""
    for name in names:
        if name not in results:
            continue
        try:
            return _finite_float(results[name], field=name)
        except (TypeError, ValueError):
            continue
    return None


def _append_trajectory_value(
    trajectory: Dict[str, list],
    key: str,
    value: float | None,
) -> None:
    """Append one finite scalar, preserving missing epochs as null."""
    trajectory[key].append(None if value is None else float(value))


def _recorded(results: Mapping[str, Any], cfg: dict) -> Dict[str, Any]:
    """What the run's records keep of one process's results (owner 2026-10-27): the metrics_to_log columns and
    their dependencies (sca/metrics.split_requested, as the history rows), never the loop's own bookkeeping
    (an unlisted loss, loss_type, train_loss_reduction)."""
    metrics, dependencies = split_requested(results or {}, cfg)
    return {**metrics, **dependencies}


# a trajectory series: the process whose records hold its value, and that value's names
_TRAJECTORY_SOURCES = {
    "training_objective": ("train", ("training_objective",)),
    "validation_objective": ("validation", ("validation_objective",)),
    "training_accuracy": ("train", ("accuracy", "categorical_accuracy", "acc")),
    "validation_accuracy": ("validation", ("accuracy", "categorical_accuracy", "acc")),
}


def _recorded_trajectory(trajectory: Dict[str, list], train: Mapping[str, Any],
                         validation: Mapping[str, Any]) -> Dict[str, list]:
    """The trajectory's series of the values the run records (train, validation: _recorded results): a loss only
    when metrics_to_log lists its objective, an accuracy when accuracy is recorded. A missing series is
    insufficient evidence to the CV diagnostics."""
    records = {"train": train or {}, "validation": validation or {}}
    return {"epochs": trajectory["epochs"],
            **{key: trajectory[key] for key, (process, names) in _TRAJECTORY_SOURCES.items()
               if any(name in records[process] for name in names)}}


def _mean_validation_objective(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    reduction: str,
    device: torch.device,
) -> float:
    """Evaluate the configured loss on validation logits as a per-trace mean."""
    with torch.no_grad():
        value = criterion(
            logits.to(device, non_blocking=True),
            labels.to(device, non_blocking=True).long(),
        )
    result = _finite_float(value.item(), field="validation_objective")
    if reduction == "sum":
        count = int(labels.numel())
        if count < 1:
            raise RuntimeError("Validation produced zero labels")
        result /= count
    return result


def _process_attacked(cfg: dict, process: str) -> bool:
    """False when rank metrics are requested but perform_attack.<process>.attack_times is 0."""
    return not _needs_rank_metrics(cfg) or attack_plan(cfg, process)["attack_times"] > 0


def _strict_int(value: Any, *, field: str, minimum: int) -> int:
    """Return one integer without silently truncating floats or accepting bool."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    return result


# =============================================================================
# Internal helpers
# =============================================================================
def _prediction_contract(cfg: Mapping[str, Any]) -> tuple[bool, float]:
    """Resolve the prediction representation declared by ``params.yaml``."""
    metrics_cfg = cfg.get("metrics_cfg") or {}
    if not isinstance(metrics_cfg, Mapping):
        raise TypeError("cfg['metrics_cfg'] must be a mapping")

    preds_are_probs = _strict_bool(
        metrics_cfg.get("preds_are_probs", False),
        field="metrics_cfg.preds_are_probs",
    )
    ce_eps = _finite_float(
        metrics_cfg.get("ce_eps", 1.0e-12),
        field="metrics_cfg.ce_eps",
    )
    if not 0.0 < ce_eps < 1.0:
        raise ValueError(
            f"metrics_cfg.ce_eps must be between 0 and 1, got {ce_eps!r}"
        )
    return preds_are_probs, ce_eps


def _classification_cross_entropy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    preds_are_probs: bool,
    ce_eps: float,
    reduction: str,
    framework_cfg: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """
    Compute CE from logits or probabilities according to ``params.yaml``.

    framework_cfg (the run config): under deep_learning.framework: keras the
    value is keras.losses.categorical_crossentropy, the function the Keras
    loss trains on; otherwise PyTorch's cross-entropy.
    """
    if predictions.ndim != 2:
        raise ValueError(
            "Model output must have shape [batch, classes], got "
            f"{tuple(predictions.shape)}"
        )
    if labels.ndim != 1 or predictions.shape[0] != labels.shape[0]:
        raise ValueError(
            "Labels must have shape [batch] aligned with model output; got "
            f"predictions={tuple(predictions.shape)}, labels={tuple(labels.shape)}"
        )
    if not torch.isfinite(predictions).all():
        raise FloatingPointError("Model output contains NaN or infinity")

    num_classes = int(predictions.shape[1])
    if labels.numel() and (
        int(labels.min().item()) < 0 or int(labels.max().item()) >= num_classes
    ):
        raise ValueError(
            f"Labels must be in [0, {num_classes - 1}] for this model output"
        )

    if framework_cfg is not None:
        per_trace = cross_entropy_per_trace(
            framework_cfg,
            logits=None if preds_are_probs else predictions,
            probabilities=predictions if preds_are_probs else None,
            targets=labels,
        )
        if per_trace is not None:
            if reduction == "none":
                return per_trace
            return per_trace.sum() if reduction == "sum" else per_trace.mean()

    if not preds_are_probs:
        return F.cross_entropy(predictions, labels, reduction=reduction)

    if torch.any(predictions < 0.0) or torch.any(predictions > 1.0):
        raise ValueError(
            "metrics_cfg.preds_are_probs=true, but model output contains values "
            "outside [0, 1]"
        )
    row_sums = predictions.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1.0e-3,
        rtol=1.0e-3,
    ):
        raise ValueError(
            "metrics_cfg.preds_are_probs=true, but model output rows do not sum to 1"
        )

    return F.nll_loss(
        predictions.clamp_min(ce_eps).log(),
        labels,
        reduction=reduction,
    )

def _save_model_atomically(
    *,
    model: torch.nn.Module,
    path: Union[str, Path],
) -> Path:
    """Serialize one model to a temporary sibling and atomically publish it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )

    try:
        save_model(model=model, path=temporary, verbose=False)
        if not temporary.is_file():
            raise RuntimeError(
                f"save_model() did not create the expected temporary file: {temporary}"
            )
        # On disk, and readable as this model's weights, before it is sealed and published:
        # a truncated file must never become the checkpoint the manifest certifies (TODO §20).
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        written = torch.load(temporary, map_location="cpu", weights_only=True)
        expected = list(model.state_dict())
        if list(written) != expected:
            raise RuntimeError(
                f"{temporary} reads back with other weights than the model holds "
                f"({len(written)} tensors, {len(expected)} expected)"
            )
        del written
        # Signed before publishing, so the checkpoint manifest's SHA-256 covers the seal.
        seal_zip_file(temporary, scope_path=destination)
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination


def _copy_file_atomically(source: Path, destination: Path) -> Path:
    """Atomically publish a checkpoint alias: the same archive, sealed for its own path."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint source does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )

    try:
        with source.open("rb") as src_handle, temporary.open("wb") as dst_handle:
            shutil.copyfileobj(src_handle, dst_handle, length=1024 * 1024)
            dst_handle.flush()
            os.fsync(dst_handle.fileno())
        # A seal covers the file's place in the run: the copy is signed for its own path.
        seal_zip_file(temporary, scope_path=destination)
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    if checkpoint_content_sha256(source) != checkpoint_content_sha256(destination):
        destination.unlink(missing_ok=True)
        raise CheckpointPolicyError(
            "Canonical best-validation checkpoint copy failed SHA-256 verification"
        )

    return destination


def _infer_model_input_length(train_loader: Any) -> int:
    """Infer trace length without advancing the DataLoader iterator."""
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__len__") or len(dataset) < 1:
        raise ValueError(
            "Cannot infer model input length from an empty or missing training dataset"
        )

    current = dataset
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attribute in ("_length", "trace_length", "input_length"):
            raw = getattr(current, attribute, None)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                return raw
        current = getattr(current, "dataset", None)

    sample = dataset[0]
    if not isinstance(sample, (tuple, list)) or len(sample) < 2:
        raise TypeError(
            "Training dataset items must be tuple/list beginning with (trace, label)"
        )

    trace = sample[0]
    if not isinstance(trace, torch.Tensor):
        trace = torch.as_tensor(trace)
    if trace.numel() < 1 or trace.ndim < 1:
        raise ValueError(
            f"Training trace must be non-empty, got shape={tuple(trace.shape)}"
        )

    input_length = int(trace.shape[-1])
    if input_length < 1:
        raise ValueError(f"Invalid inferred input length: {input_length}")
    return input_length


def _normalize_checkpoint_provenance(
    value: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate test-time checkpoint provenance supplied by ``test.py``.

    The test loop refuses untracked model weights. ``test.py`` must first call
    ``checkpoint_policy.resolve_test_checkpoint()`` and pass its ``to_dict()``
    result here. The checkpoint digest is independently verified again at the
    evaluation boundary.
    """
    if value is None:
        raise CheckpointPolicyError(
            "run_test_loop requires checkpoint_provenance from "
            "resolve_test_checkpoint(); refusing an untracked test model"
        )
    if not isinstance(value, Mapping):
        raise TypeError(
            "checkpoint_provenance must be a mapping, got "
            f"{type(value).__name__}"
        )

    required = {
        "requested_policy",
        "effective_policy",
        "validation_checkpointing",
        "checkpoint_path",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "manifest_path",
        "fallback_reason",
        "selection_metric",
        "selection_mode",
        "selection_score",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise KeyError(
            f"checkpoint_provenance is missing required fields: {missing}"
        )

    requested = str(value["requested_policy"]).strip().lower()
    effective = str(value["effective_policy"]).strip().lower()
    allowed = {"best_validation", "final_epoch"}
    if requested not in allowed or effective not in allowed:
        raise ValueError(
            "checkpoint_provenance policies must be best_validation or final_epoch"
        )

    checkpointing = value["validation_checkpointing"]
    if not isinstance(checkpointing, bool):
        raise TypeError(
            "checkpoint_provenance.validation_checkpointing must be boolean"
        )

    epoch = value["checkpoint_epoch"]
    if isinstance(epoch, bool):
        raise TypeError("checkpoint_provenance.checkpoint_epoch must be an integer")
    epoch = int(epoch)
    if epoch < 1:
        raise ValueError("checkpoint_provenance.checkpoint_epoch must be >= 1")

    checkpoint_path = Path(str(value["checkpoint_path"])).resolve()
    manifest_path = Path(str(value["manifest_path"])).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Resolved test checkpoint does not exist: {checkpoint_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint manifest does not exist: {manifest_path}"
        )

    digest = str(value["checkpoint_sha256"])
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(
            "checkpoint_provenance.checkpoint_sha256 must be lowercase SHA-256"
        )
    observed_digest = file_sha256(checkpoint_path)
    if observed_digest != digest:
        raise CheckpointPolicyError(
            "Test checkpoint SHA-256 changed after policy resolution: "
            f"expected={digest}, observed={observed_digest}"
        )

    fallback = value["fallback_reason"]
    if fallback is not None:
        fallback = str(fallback).strip() or None

    selection_metric_raw = value["selection_metric"]
    selection_mode_raw = value["selection_mode"]
    selection_score_raw = value["selection_score"]

    selection_metric = (
        str(selection_metric_raw) if selection_metric_raw is not None else None
    )
    selection_mode = (
        str(selection_mode_raw) if selection_mode_raw is not None else None
    )
    if selection_score_raw is None:
        selection_score = None
    else:
        if isinstance(selection_score_raw, bool):
            raise TypeError(
                "checkpoint_provenance.selection_score must be numeric or null"
            )
        selection_score = float(selection_score_raw)
        if not torch.isfinite(torch.tensor(selection_score)):
            raise ValueError(
                "checkpoint_provenance.selection_score must be finite"
            )

    if effective == "best_validation":
        if not checkpointing:
            raise CheckpointPolicyError(
                "effective_policy=best_validation requires validation checkpointing"
            )
        if selection_metric is None or selection_score is None:
            raise CheckpointPolicyError(
                "best-validation provenance requires validation selection metric "
                "and score"
            )
    elif selection_metric is not None or selection_score is not None:
        raise CheckpointPolicyError(
            "final-epoch checkpoint provenance must not claim validation "
            "selection metric/score"
        )

    return {
        "requested_policy": requested,
        "effective_policy": effective,
        "validation_checkpointing": checkpointing,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": digest,
        "manifest_path": str(manifest_path),
        "fallback_reason": fallback,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "selection_score": selection_score,
    }


def _runtime_series_name(
    base_series: str,
    runtime: Dict[str, Any],
) -> str:
    """
    Keep normal ClearML series names unchanged.

    CV folds and refits receive isolated series names:
        fold_01/train
        fold_01/validation
        refit/train
        refit/validation
    """
    if not runtime.get("active", False):
        return str(base_series)

    return f"{runtime['tag']}/{base_series}"

def _runtime_artifact_name(
    base_name: str,
    runtime: Dict[str, Any],
) -> str:
    """
    Return a runtime-isolated tracking artifact name.

    Standard training keeps existing artifact names. CV artifacts are isolated
    by runtime tag, for example ``fold_01_best_checkpoint`` or
    ``refit_final_trained_model``. Assessment-only folds never call this path.
    """
    if not runtime.get("active", False):
        return str(base_name)

    return f"{runtime['tag']}_{base_name}"
def _assert_metadata_alignment(
    *,
    preds: torch.Tensor,
    labels: torch.Tensor,
    metadata: Any,
    where: str,
) -> None:
    n = int(preds.shape[0])
    if labels.dim() != 1 or int(labels.shape[0]) != n:
        raise ValueError(f"{where}: labels must be shape [N], got {tuple(labels.shape)} for N={n}")

    if metadata is None:
        return

    try:
        mlen = len(metadata)
    except Exception as e:
        raise TypeError(f"{where}: metadata must be sized (len(metadata) must work): {e}") from e

    if mlen != n:
        raise ValueError(f"{where}: metadata misalignment: len(meta)={mlen} != N={n}")


def _log_monitor_throttle(
    *,
    monitor_metric: Optional[str],
    compute_rank_now: bool,
    val_results: Dict[str, Any],
    log,
) -> None:
    if not monitor_metric:
        return
    if monitor_metric in val_results:
        return

    log.info(
        "Monitor metric '%s' not computed this epoch (rank_eval=%s). "
        "Checkpoint/early-stopping will wait for next eval epoch.",
        monitor_metric,
        str(compute_rank_now),
    )


def _looks_like_probabilities(x: torch.Tensor) -> bool:
    """True when every row of ``x`` is a probability vector (a Softmax head)."""
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.numel() == 0:
        return False

    x_min = float(x.min().item())
    x_max = float(x.max().item())
    if x_min < 0.0 or x_max > 1.0 + 1e-6:
        return False

    row_sum = x.sum(dim=1)
    return torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3, rtol=1e-3)


def _reject_probability_head(out: torch.Tensor, *, where: str) -> None:
    """
    Fail when the model emits probabilities instead of logits.

    The paper's Keras models end in Softmax and train with
    categorical_crossentropy on those probabilities. TraceFlow's heads emit
    logits and train with CrossEntropyLoss(logits) = NLL(log_softmax(logits)),
    the same objective. Under deep_learning.framework: keras the loss is
    keras.losses.CategoricalCrossentropy of the logits: with
    trainer.loss.args.from_logits: false (the paper's loss) of
    keras.ops.softmax(logits) clipped to [1e-7, 1 - 1e-7], as Keras 2.1.1
    computed it on the paper's Softmax layer; with from_logits: true of
    log_softmax(logits), unclipped (framework_switch.py). Evaluation applies the Softmax once
    (trainer.evaluation.output_transformation) and the attack takes a float64
    log_softmax of the logits. A Softmax layer in the model would be applied a
    second time by the loss and by evaluation, flattening the outputs and
    changing both training and every rank metric. A LogSoftmax head is exact
    (log_softmax is idempotent) and passes.
    """
    if _looks_like_probabilities(out.detach().float()):
        raise ValueError(
            f"{where}: the model outputs probabilities (rows in [0,1] summing "
            "to 1), so CrossEntropyLoss and the evaluation softmax would apply "
            "softmax twice. Remove the final Softmax layer: the model must "
            "output logits (the paper's Keras Softmax + categorical_crossentropy "
            "is CrossEntropyLoss on logits here)."
        )


def log_last_linear_grad(model: torch.nn.Module, log, tag: str = "") -> None:
    if getattr(model, "framework", None) == "keras":
        # deep_learning.framework: keras holds Keras Dense layers, not
        # torch.nn.Linear; check the last Dense kernel (a torch Parameter on
        # the torch backend) instead of warning that no Linear exists.
        dense = [layer for layer in model.keras_model._flatten_layers()
                 if type(layer).__name__ == "Dense"]
        if dense and dense[-1].kernel.value.grad is None:
            log.warning("GradCheck%s: last Dense has no grad (graph broken or frozen params)", tag)
        return

    last_linear = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module

    if last_linear is None:
        log.warning("GradCheck%s: no Linear layers found", tag)
        return

    if last_linear.weight.grad is None:
        log.warning("GradCheck%s: last Linear has no grad (graph broken or frozen params)", tag)
        return

    # log.info(
    #     "GradCheck%s last_linear | mean_abs=%.3e max_abs=%.3e",
    #     tag,
    #     float(grad.abs().mean().item()),
    #     float(grad.abs().max().item()),
    # )


def assert_head_sanity(
    model: torch.nn.Module,
    device: torch.device,
    input_len: int,
    cfg: dict,
) -> None:
    """
    Validate model output head shape against the active SCA leakage model.

    ID  -> [N, 256]
    HW  -> [N, 9]
    MSB -> [N, 2]
    """
    leakage_model = resolve_runtime_leakage_model(cfg, mutate=True)
    expected_classes = get_num_classes(leakage_model)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros((2, 1, input_len), device=device)
        out = model(dummy)
        # A local generator: the global RNG stream (weights, batch order)
        # must not depend on this check.
        probe = torch.randn(
            (4, 1, input_len), generator=torch.Generator().manual_seed(0)
        ).to(device)
        probe_out = model(probe)

    if was_training:
        model.train()

    if not isinstance(out, torch.Tensor):
        raise TypeError(
            f"Model forward must return a torch.Tensor, got {type(out)}"
        )

    if out.dim() != 2 or int(out.shape[1]) != int(expected_classes):
        raise ValueError(
            f"Model head must output [N,{expected_classes}] "
            f"for leakage_model={leakage_model}. "
            f"Got {tuple(out.shape)}. "
            "Fix your leakage_model/num_classes wiring or model builder."
        )
    for sample in (out, probe_out):
        _reject_probability_head(sample, where="Model head sanity check")


def _maybe_set_progress(task: Any, progress_fraction: float) -> None:
    if not task:
        return
    try:
        progress_pct = max(0, min(100, int(round(float(progress_fraction) * 100))))
        task.set_progress(progress_pct)
    except Exception:
        pass


def _log_results_to_clearml(
    tracker: Optional[ClearMLTracker],
    results: Dict[str, Any],
    *,
    step: int,
    series: str,
    logger_obj,
) -> None:
    if not tracker or not tracker.enabled:
        return

    try:
        tracker.log_metrics(results, step=step, series=series)

        curve_axis_map = {
            "full_ranks": ("traces", "rank"),
            "mean_attack_rank_curve": ("traces", "GE"),
            "sr_curve": ("traces", "success_rate"),
            "LDD": ("key_guess", "distance"),
        }

        for key, (xaxis, yaxis) in curve_axis_map.items():
            if key in results:
                tracker.log_curve_from_2d_array(
                    title=f"{series}/{key}",
                    curve=results[key],
                    step=step,
                    series=series,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

    except Exception as e:
        logger_obj.warning("ClearML logging failed for %s step %d: %s", series, step, e, exc_info=True)


def _maybe_upload_file_to_clearml(
    tracker: Optional[ClearMLTracker],
    name: str,
    path: Union[str, Path],
    logger_obj,
) -> None:
    if not tracker or not tracker.enabled:
        return
    try:
        tracker.upload_file(name, path)
    except Exception as e:
        logger_obj.warning("ClearML file upload failed for '%s': %s", name, e, exc_info=True)


# =============================================================================
# Training loop
# =============================================================================
def run_training_loop(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    train_loader,
    val_loader,
    cfg: dict,
    logger,
    base_dir: Union[str, Path],
    model_name: str,
    env: str,
    task: Any = None,
    tracker: Optional[ClearMLTracker] = None,
) -> Dict[str, Any]:
    """
    Train with validation each epoch, or without validation for a CV final refit.

    Notes
    -----
    - Train metrics are computed in a single pass using buffered logits/labels.
    - Rank metrics are stripped from train: training traces are never
      attacked, also not by the CV refit's evaluation pass.
    - Validation rank metrics are throttled by rank_eval_interval unless forced
      by monitor_metric or enabled key-recovery reporting.
    - CV (cross validation) folds use fold-local artifact paths injected through cfg["paths"].
    - Any epoch error fails the run at once (standard training, CV folds and
      the refit alike); no epoch is skipped.
    - Fold assessment always uses the final configured epoch.
    - With validation checkpointing disabled, folds return assessment evidence
      without publishing model artifacts.
    """
    set_seed(cfg)
    _maybe_sync_metrics_key(cfg)
    key_recovery_contract = validate_key_recovery_contract(cfg)
    runtime = resolve_cross_validation_runtime(cfg)
    if runtime["is_refit"]:
        # The final training has no held-out data. An optional val_loader must
        # be the training population itself (ordered, no metadata) and is
        # used only to report training-set accuracy and loss each epoch.
        if val_loader is not None and len(val_loader.dataset) != len(train_loader.dataset):
            raise ValueError(
                "Final training evaluation loader must cover exactly the training "
                "population; attack data must remain reserved"
            )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise ValueError("ReduceLROnPlateau requires validation in this loop and cannot be used for validation-free refit")
    elif val_loader is None:
        raise ValueError("Standard training and CV folds require a validation loader")
    train_series = _runtime_series_name("train", runtime)
    validation_series = _runtime_series_name("validation", runtime)

    logger.info(
        "Training execution: %s | tag=%s",
        runtime["description"],
        runtime["tag"],
    )

    trainer_cfg = cfg.get("trainer") or {}
    if not isinstance(trainer_cfg, dict):
        raise TypeError("cfg['trainer'] must be a mapping")

    early_stopping = _strict_bool(
        trainer_cfg.get("early_stopping", False),
        field="trainer.early_stopping",
    )
    if runtime["active"] and early_stopping:
        raise ValueError("CV folds and final refit require trainer.early_stopping=false")
    checkpointing = (
        False
        if runtime["is_refit"]
        else get_validation_checkpointing(cfg)
    )
    fold_assessment_only = runtime["is_fold"] and not checkpointing
    monitor_metric, monitor_mode = (
        (None, "minimize")
        if runtime["is_refit"] or fold_assessment_only
        else get_monitor_metric_and_mode(cfg)
    )
    # trainer.monitor_metric drives early stopping (patience, min_delta) and
    # best-validation checkpointing. Without either, every configured epoch
    # runs and the final epoch is selected, so the monitor is inert: it does
    # not force a validation attack each epoch or choose any epoch. When one
    # of them reads it, the run must record it (refused before the first epoch).
    check_monitor_metric(cfg, monitor_metric, early_stopping=bool(early_stopping), checkpointing=bool(checkpointing))
    if monitor_metric and not (early_stopping or checkpointing):
        logger.info(
            "trainer.monitor_metric '%s' is inactive: early_stopping=false and "
            "validation checkpointing is off",
            monitor_metric,
        )
        monitor_metric = None
    elif early_stopping and not monitor_metric:
        raise ValueError("trainer.early_stopping=true requires trainer.monitor_metric")
    if monitor_metric:
        logger.info("Monitoring metric: '%s' (%s)", monitor_metric, monitor_mode)
    else:
        logger.info(
            "Metric-based checkpoint selection is inactive for %s",
            runtime["description"],
        )
    validation_attacked = _process_attacked(cfg, "validation")
    # Folds of a database with random profiling keys (ASCADr) are not attacked: no single key to rank.
    folds_attacked = runtime["is_fold"] and not random_profiling_keys(cfg)
    if not validation_attacked and (folds_attacked or _monitor_is_rank_metric(monitor_metric)):
        raise ValueError(
            "perform_attack.validation.attack_times is 0, but the cross-validation "
            "fold rank curves / the rank monitor_metric need the validation attack"
        )

    best_score = float("inf") if monitor_mode == "minimize" else float("-inf")
    best_epoch: Optional[int] = None
    epochs_completed = 0

    requested_test_policy = (
        "final_epoch"
        if runtime["is_fold"]
        else get_requested_test_policy(cfg)
    )

    if checkpointing and requested_test_policy == "best_validation" and not monitor_metric:
        raise CheckpointPolicyError(
            "evaluation.test.load_model=best_validation requires a validation "
            "monitor_metric when validation checkpointing is enabled"
        )

    # The model that leaves this loop for testing is the best-validation
    # checkpoint only when checkpointing is on AND best_validation is
    # requested; otherwise it is the final (or early-stopping) epoch. Every
    # published record of "the trained model" (the returned train/validation
    # metrics, best_score, validation_best.json) must describe that same epoch.
    select_final_epoch = not (
        checkpointing and requested_test_policy == "best_validation"
    )

    # Clear canonical paths for every mode so rerunning a fold cannot leave
    # stale weights produced by an older implementation.
    # Folds share paths.checkpoints/paths.model_out with the refit, so their
    # checkpoint files are named <model_name>_fold_NN_*.
    checkpoint_name = (
        fold_artifact_name(model_name, runtime["fold_number"])
        if runtime["is_fold"]
        else model_name
    )
    checkpoint_artifacts = clear_checkpoint_artifacts(
        cfg,
        model_name=checkpoint_name,
    )
    patience = (
        _strict_int(
            trainer_cfg.get("patience", 3),
            field="trainer.patience",
            minimum=1,
        )
        if early_stopping
        else 1
    )
    min_delta = _finite_float(
        trainer_cfg.get("min_delta", 0.0),
        field="trainer.min_delta",
    )
    if min_delta < 0.0:
        raise ValueError(f"trainer.min_delta must be >= 0, got {min_delta}")

    epochs_raw = (
        runtime["resolved_refit_epochs"]
        if runtime["is_refit"]
        else trainer_cfg.get("epochs", 1)
    )
    epochs = _strict_int(
        epochs_raw,
        field=(
            "cross_validation_runtime.resolved_refit_epochs"
            if runtime["is_refit"]
            else "trainer.epochs"
        ),
        minimum=1,
    )
    # TraceFlow Monitor: which part of the run this loop is (utils/heartbeat.py).
    heartbeat.update(
        phase="fold" if runtime["is_fold"] else ("refit" if runtime["is_refit"] else "train"),
        fold=runtime.get("fold_number"),
        epoch=0,
        batch=0,
        batches_per_epoch=len(train_loader),
        **({"fold_epochs": epochs} if runtime["is_fold"] else {"refit_epochs": epochs}),
    )
    # A OneCycleLR without a length in its args gets this loop's: trainer.epochs x
    # len(train_loader), so a fold, the refit and a hold-out training each finish
    # their cycle (optimizer_util.build_loop_scheduler). PyTorch only: Keras has no
    # OneCycleLR (refused when its optimizer is built).
    if not resolve_framework(cfg).is_keras:
        check_scheduler_training(trainer_cfg)
        if scheduler is None:
            scheduler = build_loop_scheduler(
                optimizer, trainer_cfg, epochs=epochs, steps_per_epoch=len(train_loader)
            )
            if scheduler is not None:
                logger.info(
                    "OneCycleLR length of this training loop: %d epochs x %d batches = %d optimizer steps",
                    epochs,
                    len(train_loader),
                    int(scheduler.total_steps),
                )
    # Per-step schedulers (OneCycleLR, CyclicLR) advance after every
    # optimizer.step(); per-epoch schedulers advance once per epoch.
    scheduler_interval = resolve_scheduler_interval(trainer_cfg, scheduler)
    validate_scheduler_for_loop(
        scheduler,
        scheduler_interval,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    if scheduler is not None:
        logger.info(
            "LR scheduler %s stepped per %s",
            type(scheduler).__name__,
            scheduler_interval,
        )
    grad_clip = trainer_cfg.get("grad_clip_norm", None)
    clip_gradients = None
    if grad_clip is not None:
        grad_clip = _finite_float(
            grad_clip,
            field="trainer.grad_clip_norm",
        )
        if grad_clip <= 0.0:
            raise ValueError(
                "trainer.grad_clip_norm must be > 0 when configured, "
                f"got {grad_clip}"
            )
        # torch.nn.utils.clip_grad_norm_ (pytorch); None under keras, whose
        # optimizer clips with its own global_clipnorm.
        clip_gradients = gradient_clipper(cfg)
    criterion = build_loss(cfg).to(device)
    loss_reduction = trainer_cfg["loss"]["args"].get("reduction", "mean")
    preds_are_probs, ce_eps = _prediction_contract(cfg)
    if preds_are_probs:
        raise ValueError("Configured training losses require raw model logits; set metrics_cfg.preds_are_probs=false")

    # if not early_stopping:
    #     logger.info("early_stopping=false: training will run all configured epochs")

    logger.info(
        "Validation checkpointing=%s | requested test model=%s",
        checkpointing,
        requested_test_policy.upper(),
    )
    if fold_assessment_only:
        logger.info(
            "Validation checkpointing disabled for CV fold: assessment data only; "
            "no weights or checkpoint manifest will be written"
        )
    elif not checkpointing:
        logger.info(
            "Best-validation checkpoint saving is disabled; "
            "test.py will explicitly fall back to FINAL_EPOCH when requested"
        )
    wait = 0
    rank_every = _rank_eval_interval(cfg)
    # evaluation.validation.judge_log: one line per epoch reading train and validation accuracy against chance
    validation_judge = ValidationJudge() if judge_log_enabled(cfg) else None
    # the epoch before's train and validation values, for the arrows of each epoch's block
    previous_train_block: Optional[Dict[str, Any]] = None
    previous_block: Optional[Dict[str, Any]] = None
    # logger.info("Rank-metrics evaluation interval: every %d epoch(s)", rank_every)

    best_train_metrics: Dict[str, Any] = {}
    best_val_metrics: Dict[str, Any] = {}
    final_train_metrics: Dict[str, Any] = {}
    final_val_metrics: Dict[str, Any] = {}
    trajectory: Dict[str, list] = {
        "epochs": [],
        "training_objective": [],
        "validation_objective": [],
        "training_accuracy": [],
        "validation_accuracy": [],
    }
    best_epoch_checkpoint_path: Optional[Path] = None
    best_validation_checkpoint_path: Optional[Path] = None
    final_trained_path: Optional[Path] = None
    checkpoint_manifest_path: Optional[Path] = None

    input_len = _infer_model_input_length(train_loader)
    #logger.info("Model-head sanity input length: %d", input_len)
    assert_head_sanity(model, device, input_len=input_len, cfg=cfg)

    cfg_train = _strip_rank_metrics(filter_metrics_for_metadata(cfg, has_metadata=False))
    train_samples = len(train_loader.dataset)
    # The CV pipeline announces the refit (data, seed, epochs, no held-out
    # data) in a framed block before calling this loop.

    # Time spent in the optimisation passes only (excludes validation attacks).
    training_seconds = 0.0
    # Time spent evaluating the held-out traces after each epoch.
    validation_seconds = 0.0
    fit_time_requested = requested_fit_time_metrics(cfg)
    model_size_requested = requested_model_size_metrics(cfg)
    loop_state_requested = requested_loop_state_metrics(cfg)
    last_used_learning_rate: Optional[float] = None    # the rate the latest optimizer step used (the metric η_t)
    loop_started = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        heartbeat.update(epoch=epoch, batch=0)

        try:
            # ---------------- TRAIN ----------------
            model.train()

            loss_mass = 0.0
            total = 0
            train_logits_buf = []
            train_labels_buf = []

            for batch_idx, batch in enumerate(train_loader):
                x, y, _ = unpack_batch(batch)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).long()

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)

                # Kept on the device; moved once per epoch. Per-batch .cpu()
                # and .item() calls each forced a full device synchronisation.
                train_logits_buf.append(logits.detach())
                train_labels_buf.append(y.detach())

                loss_raw = criterion(logits, y)
                bs = int(y.numel())

                # float64 on the device, added in batch order: the same values
                # the former Python-float accumulation produced.
                loss_value = loss_raw.detach().to(dtype=torch.float64)
                if loss_reduction == "mean":
                    loss_mass = loss_mass + loss_value * bs
                else:
                    loss_mass = loss_mass + loss_value

                loss_raw.backward()

                if batch_idx == 0:
                    log_last_linear_grad(model, logger, tag=f" [ep{epoch}]")

                if clip_gradients is not None:
                    clip_gradients(model)

                optimizer.step()
                last_used_learning_rate = eta_t(optimizer)   # before the scheduler moves on
                if scheduler_interval == "step":
                    scheduler.step()
                total += bs
                heartbeat.update(batch=batch_idx + 1)

            if total <= 0 or not train_logits_buf or not train_labels_buf:
                raise RuntimeError("Training loader produced zero samples")
            if not bool(torch.isfinite(torch.as_tensor(loss_mass)).all()):
                raise FloatingPointError("Training loss is NaN or infinity")
            # The loss precedes the last optimiser step, which can still break
            # the weights; one device sync per epoch.
            if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
                raise FloatingPointError("Model weights are NaN or infinity after the optimiser steps")
            epoch_train_seconds = time.time() - t0
            training_seconds += epoch_train_seconds
            # The model now holds epoch-N weights: the epoch label follows the
            # completed optimisation passes, and nothing from epoch N-1 may
            # describe them.
            epochs_completed = int(epoch)
            final_train_metrics, final_val_metrics = {}, {}

            train_loss = float(float(loss_mass) / total)

            y_pred_train = torch.cat(train_logits_buf, dim=0).cpu().float()
            y_true_train = torch.cat(train_labels_buf, dim=0).cpu().long()

            train_results = compute_metrics(
                y_pred_train,
                y_true_train,
                cfg_train,
                metadata=None,
                process="train",
            )
            logger.info("Training objective (%s), mean per trace: %.8f", trainer_cfg["loss"]["type"], train_loss)
            train_results["train_loss_reduction"] = str(loss_reduction)
            train_results["training_objective"] = train_loss
            train_results["loss_type"] = loss_signature(cfg)
            train_results.update(
                fit_time_values(fit_time_requested, seconds=training_seconds, epoch=epoch)
            )
            train_results.update(model_size_values(model_size_requested, model))
            train_results.update(loop_state_values(loop_state_requested, learning_rate=last_used_learning_rate))
            final_train_metrics = dict(train_results)

            # CV final training persists its row after the evaluation pass below.
            if not (runtime["is_refit"] and val_loader is not None):
                append_metrics_row(cfg, train_results, process="train", epoch=epoch)
                _log_results_to_clearml(tracker, train_results, step=epoch, series=train_series, logger_obj=logger)

            # Fixed-duration refit consumes only profiling data. Publish the
            # final model through the common checkpoint path after the loop.
            if runtime["is_refit"]:
                if scheduler_interval == "epoch":
                    scheduler.step()
                best_train_metrics = dict(train_results)
                trajectory["epochs"].append(int(epoch))
                _append_trajectory_value(
                    trajectory,
                    "training_objective",
                    train_loss,
                )
                _append_trajectory_value(
                    trajectory,
                    "validation_objective",
                    None,
                )
                _append_trajectory_value(
                    trajectory,
                    "training_accuracy",
                    _optional_metric(
                        train_results,
                        "accuracy",
                        "categorical_accuracy",
                        "acc",
                    ),
                )
                _append_trajectory_value(
                    trajectory,
                    "validation_accuracy",
                    None,
                )
                block_results = train_results
                if val_loader is not None:
                    # Eval-mode pass over the training population itself:
                    # training-set accuracy and loss (optimistic), never a
                    # selection signal. Training traces are never attacked:
                    # cfg_train holds no rank metrics, and rank metrics come
                    # only from held-out data (CV folds, test).
                    y_pred_fit, y_true_fit, _meta_fit = collect_logits_labels_meta(
                        model, device, val_loader
                    )
                    block_results = compute_metrics(
                        y_pred_fit, y_true_fit, cfg_train, metadata=None, process="train",
                    )
                    # This evaluation over the whole profiling set is the final
                    # training's record in its metric history.
                    block_results["training_objective"] = train_loss
                    block_results["train_loss_reduction"] = str(loss_reduction)
                    block_results["loss_type"] = loss_signature(cfg)
                    block_results.update(
                        fit_time_values(fit_time_requested, seconds=training_seconds, epoch=epoch)
                    )
                    block_results.update(model_size_values(model_size_requested, model))
                    block_results.update(loop_state_values(loop_state_requested, learning_rate=last_used_learning_rate))
                    final_train_metrics = dict(block_results)
                    best_train_metrics = dict(block_results)
                    append_metrics_row(cfg, block_results, process="train", epoch=epoch)
                    _log_results_to_clearml(
                        tracker, block_results, step=epoch, series=train_series, logger_obj=logger
                    )
                    logger.info(
                        "Final-training metrics on its %d training traces "
                        "(eval mode after the epoch; training-set, optimistic):",
                        int(y_true_fit.numel()),
                    )
                elapsed = time.time() - t0
                # Same epoch block as standard and CV-fold epochs: the refit has train values only.
                print_blocked_results(
                    logger=logger,
                    results=block_results,
                    elapsed=elapsed,
                    eta=elapsed * (epochs - epoch),
                    epoch=epoch,
                    epochs=epochs,
                    cfg=cfg,
                    process="train",
                    previous=previous_block,
                )
                previous_block = dict(block_results)
                _maybe_set_progress(
                    tracker.task if tracker and tracker.enabled else task,
                    epoch / float(epochs),
                )
                continue

            # ---------------- VALIDATION ----------------
            validation_started = time.time()
            compute_rank_now = (
                (epoch % rank_every == 0)
                or _monitor_is_rank_metric(monitor_metric)
                or (
                    key_recovery_contract["enabled"]
                    and not fold_assessment_only
                )
                or (runtime["is_fold"] and epoch == epochs)
            )

            y_pred_val, y_true_val, meta_val = collect_logits_labels_meta(model, device, val_loader)

            if compute_rank_now and validation_attacked:
                cfg_val = filter_metrics_for_metadata(cfg, has_metadata=(meta_val is not None))
            else:
                cfg_val = _strip_rank_metrics(
                    filter_metrics_for_metadata(cfg, has_metadata=(meta_val is not None))
                )

            wants_rank = _needs_rank_metrics(cfg_val)
            if wants_rank and meta_val is None:
                raise ValueError(
                    "Validation requested rank-metrics but metadata is None. "
                    "Fix: ensure val_loader returns (x,y,meta) and meta includes plaintext/key."
                )

            _assert_metadata_alignment(
                preds=y_pred_val,
                labels=y_true_val,
                metadata=meta_val,
                where="VAL",
            )

            val_results = compute_metrics(
                y_pred_val,
                y_true_val,
                cfg_val,
                metadata=meta_val,
                process="validation",
            )

            validation_objective = _mean_validation_objective(
                criterion,
                y_pred_val,
                y_true_val,
                reduction=loss_reduction,
                device=device,
            )
            val_results["validation_objective"] = validation_objective
            val_results["loss_type"] = loss_signature(cfg)
            validation_seconds += time.time() - validation_started
            val_results.update(
                fit_time_values(fit_time_requested, seconds=validation_seconds, epoch=epoch)
            )

            # Remove the structured evidence before generic metric logging.
            # append_metrics_row(), ClearML, and print_blocked_results() must only receive
            # ordinary scalar/array metrics.
            val_recovery_evidence = pop_key_recovery_evidence(val_results)
            final_val_metrics = dict(val_results)

            trajectory["epochs"].append(int(epoch))
            _append_trajectory_value(
                trajectory,
                "training_objective",
                train_loss,
            )
            _append_trajectory_value(
                trajectory,
                "validation_objective",
                validation_objective,
            )
            _append_trajectory_value(
                trajectory,
                "training_accuracy",
                _optional_metric(
                    train_results,
                    "accuracy",
                    "categorical_accuracy",
                    "acc",
                ),
            )
            _append_trajectory_value(
                trajectory,
                "validation_accuracy",
                _optional_metric(
                    val_results,
                    "accuracy",
                    "categorical_accuracy",
                    "acc",
                ),
            )

            val_recovery_report = None

            if (
                key_recovery_contract["enabled"]
                and compute_rank_now
                and wants_rank
            ):
                if val_recovery_evidence is None:
                    raise KeyRecoveryError(
                        "Validation rank evaluation completed without producing "
                        "key-recovery evidence. Ensure perform_attacks() returns "
                        "__key_recovery_evidence__ and compute_metrics() propagates it."
                    )

                val_recovery_report = build_key_recovery_report(
                    cfg=cfg,
                    process="validation",
                    evidence=val_recovery_evidence,
                    epoch=epoch,
                    model_name=model_name,
                    model_hash=cfg.get("model_hash"),
                    selection_metric=monitor_metric,
                    selection_score=(
                        val_results.get(monitor_metric)
                        if monitor_metric and monitor_metric in val_results
                        else None
                    ),
                )


            append_metrics_row(
                cfg,
                val_results,
                process="validation",
                epoch=epoch,
            )
            if validation_judge is not None:
                logger.info(validation_judge.line(
                    epoch=epoch, epochs=epochs,
                    train=(y_pred_train, y_true_train), validation=(y_pred_val, y_true_val),
                    train_objective=train_loss, validation_objective=validation_objective,
                    validation_results=val_results,
                ))

            _log_results_to_clearml(
                tracker,
                val_results,
                step=epoch,
                series=validation_series,
                logger_obj=logger,
            )
            # ---------------- SCHEDULER ----------------
            if scheduler_interval == "epoch":
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    # The validation objective of this pass: the configured loss
                    # (label_smoothing included) on the held-out traces, as
                    # optimizer_util.resolve_scheduler_config promises; it never
                    # depends on whether cross_entropy is logged.
                    scheduler.step(float(validation_objective))
                else:
                    scheduler.step()

            # logger.info(
            #     "VAL: N=%d meta=%s rank_eval=%s metrics=%d",
            #     int(y_pred_val.shape[0]),
            #     "None" if meta_val is None else str(len(meta_val)),
            #     str(compute_rank_now),
            #     len(cfg_val.get("metrics", []) if isinstance(cfg_val.get("metrics"), list) else []),
            # )

            _log_monitor_throttle(
                monitor_metric=monitor_metric,
                compute_rank_now=compute_rank_now,
                val_results=val_results,
                log=logger,
            )

            # ---------------- CHECKPOINT / EARLY STOPPING ----------------
            validation_is_best = False
            stop_training = False

            if fold_assessment_only:
                if epoch == epochs:
                    # logger.info(
                    #     "CV fold assessment fixed at final configured epoch %d",
                    #     epoch,
                    # )
                    # The final configured epoch is this fold's selected
                    # snapshot; publish it as the fold's validation_best.json.
                    validation_is_best = val_recovery_report is not None

            elif monitor_metric and _optional_metric(val_results, monitor_metric) is not None:
                # None (threshold never reached) or a non-finite value is "not
                # observed this epoch" and leaves the selection state unchanged.
                current_val = _finite_float(
                    val_results[monitor_metric],
                    field=f"validation.{monitor_metric}",
                )

                improved = is_better_score(
                    current_val,
                    best_score,
                    monitor_metric,
                    monitor_mode,
                    min_delta,
                )

                if improved:
                    best_score = current_val
                    best_epoch = int(epoch)
                    wait = 0
                    validation_is_best = True

                    if checkpointing:
                        ckpt_path = checkpoint_artifacts.epoch_checkpoint(epoch)
                        best_path = checkpoint_artifacts.best_validation

                        _save_model_atomically(
                            model=model,
                            path=ckpt_path,
                        )
                        _copy_file_atomically(ckpt_path, best_path)

                        best_epoch_checkpoint_path = ckpt_path
                        best_validation_checkpoint_path = best_path

                        logger.info(
                            "New best validation checkpoint: epoch=%d | %s=%s | %s",
                            epoch,
                            monitor_metric,
                            str(current_val),
                            best_path,
                        )

                        if val_recovery_report is not None:
                            val_recovery_report["model"]["source_checkpoint"] = str(
                                ckpt_path
                            )
                            val_recovery_report["model"][
                                "source_checkpoint_epoch"
                            ] = int(epoch)
                            val_recovery_report["model"][
                                "checkpoint_role"
                            ] = "best_validation"
                            val_recovery_report["model"][
                                "source_checkpoint_sha256"
                            ] = file_sha256(ckpt_path)
                            val_recovery_report["model"][
                                "canonical_checkpoint"
                            ] = str(best_path)

                        if task:
                            try:
                                task.upload_artifact(
                                    _runtime_artifact_name(
                                        "best_checkpoint",
                                        runtime,
                                    ),
                                    artifact_object=str(ckpt_path),
                                )
                            except Exception:
                                pass

                        _maybe_upload_file_to_clearml(
                            tracker,
                            _runtime_artifact_name(
                                "best_epoch_checkpoint",
                                runtime,
                            ),
                            ckpt_path,
                            logger,
                        )
                        _maybe_upload_file_to_clearml(
                            tracker,
                            _runtime_artifact_name(
                                "best_validation_checkpoint",
                                runtime,
                            ),
                            best_path,
                            logger,
                        )

                    else:
                        logger.info(
                            "New best metric observed but checkpointing=false: %s=%s",
                            monitor_metric,
                            str(current_val),
                        )

                    best_train_metrics = dict(train_results)
                    best_val_metrics = dict(val_results)

                else:
                    wait += 1

                    logger.info(
                        "No improvement on '%s' (%d/%d)",
                        monitor_metric,
                        wait,
                        patience,
                    )

                    if early_stopping and wait >= patience:
                        epochs_completed = int(epoch)

                        logger.info(
                            "Early stopping at epoch %d",
                            epoch,
                        )

                        logger.info(
                            "Final trained model will be saved once after the loop "
                            "at the actual stopping epoch"
                        )

                        # Do not break yet. The validation recovery report for this epoch
                        # must be persisted first.
                        stop_training = True

            elif monitor_metric is None:
                best_train_metrics = dict(train_results)
                best_val_metrics = dict(val_results)

                # With no formal monitor, only a rank-evaluated epoch can become
                # the selected key-recovery snapshot. This keeps validation_best.json
                # aligned with any selected checkpoint.
                if val_recovery_report is not None:
                    validation_is_best = True
                    best_epoch = int(epoch)

            elif monitor_metric not in val_results and (
                compute_rank_now or not _monitor_is_rank_metric(monitor_metric)
            ):
                # Early stopping / checkpointing would silently never fire.
                raise ValueError(
                    f"trainer.monitor_metric '{monitor_metric}' is not computed on "
                    f"validation (epoch {epoch}); add it to metrics_to_log or "
                    "monitor a metric that is logged"
                )
            else:
                logger.info(
                    "Monitor metric %r not reached at epoch %d; selection state "
                    "left unchanged.",
                    monitor_metric,
                    epoch,
                )


            if select_final_epoch and not fold_assessment_only:
                # The test loads the final-epoch model (or the early-stopping
                # epoch), so the selected validation snapshot must describe
                # that same model, not the best epoch.
                validation_is_best = (
                    val_recovery_report is not None
                    and (epoch == epochs or stop_training)
                )

            # ---------------- VALIDATION KEY-RECOVERY ARTIFACTS ----------------
            if val_recovery_report is not None:
                saved_recovery_paths = save_validation_report(
                    cfg=cfg,
                    report=val_recovery_report,
                    is_best=validation_is_best,
                    model_name=model_name,
                )

                history_path = saved_recovery_paths.get(
                    "validation_history"
                )

                if history_path is not None:
                    logger.info(
                        "Validation key-recovery history updated: %s",
                        history_path,
                    )

                if validation_is_best:
                    best_report_path = saved_recovery_paths.get(
                        "validation_best"
                    )

                    if best_report_path is not None:
                        logger.info(
                            "Selected validation key-recovery report updated: %s",
                            best_report_path,
                        )


            if stop_training:
                break
            elapsed = time.time() - t0
            eta = elapsed * (epochs - epoch)

            # The block shows the metrics of metrics_to_log and nothing else, the epoch's train values
            # beside its validation values; its fit time is the time of the optimisation passes.
            print_blocked_results(
                logger=logger,
                results=val_results,
                elapsed=elapsed,
                eta=eta,
                epoch=epoch,
                epochs=epochs,
                cfg=cfg,
                process="validation",
                label="validation (CV)" if runtime["is_fold"] else "validation",
                previous=previous_block,
                train=train_results,
                previous_train=previous_train_block,
            )
            previous_block, previous_train_block = dict(val_results), dict(train_results)

            progress_fraction = epoch / float(epochs)

            if tracker and tracker.enabled:
                _maybe_set_progress(tracker.task, progress_fraction)
            else:
                _maybe_set_progress(task, progress_fraction)


        except Exception as e:
            logger.error(
                "Epoch %d failed during %s: %s",
                epoch,
                runtime["description"],
                e,
                exc_info=True,
            )

            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        # ---------------- FINAL MODEL + CHECKPOINT MANIFEST ----------------
    if epochs_completed < 1:
        raise RuntimeError(
            "Training completed without one fully successful epoch; "
            "no final model or checkpoint manifest will be published"
        )

    if fold_assessment_only:
        if epochs_completed != epochs:
            raise RuntimeError(
                "CV fold did not complete its fixed epoch count: "
                f"completed={epochs_completed}, configured={epochs}"
            )
        logger.info(
            "CV fold %d/%d completed at epoch %d; model weights released without persistence",
            runtime["fold_number"],
            runtime["n_folds"],
            epochs_completed,
        )
        # the fold's records keep what metrics_to_log names and its dependencies (_recorded), as its history rows
        recorded_train, recorded_val = _recorded(final_train_metrics, cfg), _recorded(final_val_metrics, cfg)
        return {
            "train": recorded_train,
            "validation": recorded_val,
            "final_train": {
                "epoch": int(epochs_completed),
                "metrics": recorded_train,
            },
            "final_validation": {
                "epoch": int(epochs_completed),
                "metrics": recorded_val,
            },
            "trajectory": _recorded_trajectory(trajectory, recorded_train, recorded_val),
            "training_time_s": float(training_seconds),
            "selection": "final_configured_epoch",
            "epochs_configured": int(epochs),
            "epochs_completed": int(epochs_completed),
            "checkpointing": {
                "validation_enabled": False,
                "model_persisted": False,
                "best_epoch_checkpoint": None,
                "best_validation_checkpoint": None,
                "final_epoch_checkpoint": None,
                "manifest": None,
            },
            "execution": {
                "type": "fold",
                "tag": runtime["tag"],
                "fold_index": runtime["fold_index"],
                "fold_number": runtime["fold_number"],
                "n_folds": runtime["n_folds"],
            },
        }

    try:
        final_trained_path = _save_model_atomically(
            model=model,
            path=checkpoint_artifacts.final_epoch,
        )
        logger.info(
            "Final-epoch model saved: epoch=%d | %s",
            epochs_completed,
            final_trained_path,
        )

        _maybe_upload_file_to_clearml(
            tracker,
            _runtime_artifact_name("final_trained_model", runtime),
            final_trained_path,
            logger,
        )

        manifest_best_epoch = best_epoch if monitor_metric is not None else None
        manifest_best_score = (
            _finite_float(
                best_score,
                field=f"best validation {monitor_metric}",
            )
            if manifest_best_epoch is not None
            else None
        )
        manifest_epoch_checkpoint = (
            best_epoch_checkpoint_path
            if manifest_best_epoch is not None
            else None
        )
        manifest_best_checkpoint = (
            best_validation_checkpoint_path
            if manifest_best_epoch is not None
            else None
        )

        if (
            checkpointing
            and requested_test_policy == "best_validation"
            and manifest_best_epoch is None
        ):
            raise CheckpointPolicyError(
                "Validation checkpointing was enabled and best_validation was "
                "requested, but no valid monitored validation checkpoint was produced"
            )

        checkpoint_manifest_path = write_checkpoint_manifest(
            cfg=cfg,
            model_name=checkpoint_name,
            model_hash=cfg.get("model_hash"),
            validation_checkpointing=checkpointing,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            best_epoch=manifest_best_epoch,
            best_score=manifest_best_score,
            epoch_checkpoint=manifest_epoch_checkpoint,
            best_validation_checkpoint=manifest_best_checkpoint,
            final_epoch=epochs_completed,
            final_checkpoint=final_trained_path,
            epochs_configured=epochs,
            execution={
                "type": (
                    "fold"
                    if runtime["is_fold"]
                    else "refit"
                    if runtime["is_refit"]
                    else "standard"
                ),
                "tag": runtime["tag"],
                "fold_index": runtime["fold_index"],
                "fold_number": runtime["fold_number"],
                "n_folds": runtime["n_folds"],
                "seed": cfg.get("seed"),
                "train_samples": int(train_samples),
                "epochs": int(epochs_completed),
                "training_time_s": round(float(training_seconds), 3),
                "total_time_s": round(time.time() - loop_started, 3),
            },
        )
        logger.info(
            "Checkpoint manifest saved: %s",
            checkpoint_manifest_path,
        )

    except Exception as e:
        logger.error(
            "Failed to publish final checkpoint artifacts during %s: %s",
            runtime["description"],
            e,
            exc_info=True,
        )
        raise

    # Records of the tested model: the final epoch unless the best-validation
    # checkpoint is the one that will be tested. Never mix epochs.
    if select_final_epoch or best_epoch is None:
        selected_train_metrics = dict(final_train_metrics)
        selected_val_metrics = dict(final_val_metrics)
        selected_epoch = int(epochs_completed)
    else:
        selected_train_metrics = dict(best_train_metrics)
        selected_val_metrics = dict(best_val_metrics)
        selected_epoch = int(best_epoch)
        logger.info(
            "Selected epoch %d by validation %s: its validation values are the best over the "
            "epochs on the set that chose it (selection-biased, optimistic); the test is the "
            "estimate of this model's performance.",
            selected_epoch, monitor_metric,
        )
    for record in (selected_train_metrics, selected_val_metrics):
        if record:
            record["selected_epoch"] = selected_epoch

    best_score_out: Optional[float] = None

    if best_epoch is not None and monitor_metric is not None:
        best_score_out = _finite_float(
            best_score,
            field=f"best validation {monitor_metric}",
        )

    # the run's records keep what metrics_to_log names and its dependencies (_recorded), as its history rows
    final_train_metrics, final_val_metrics = _recorded(final_train_metrics, cfg), _recorded(final_val_metrics, cfg)
    result: Dict[str, Any] = {
        # Metrics of the epoch whose weights are tested (see select_final_epoch).
        "train": _recorded(selected_train_metrics, cfg),
        "validation": _recorded(selected_val_metrics, cfg),
        "selected_epoch": selected_epoch,
        "final_train": {
            "epoch": int(epochs_completed),
            "metrics": final_train_metrics,
        },
        "trajectory": _recorded_trajectory(trajectory, final_train_metrics, final_val_metrics),
        "training_time_s": float(training_seconds),
        "total_time_s": float(time.time() - loop_started),
        "best_epoch": best_epoch,
        "best_score": best_score_out,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "epochs_configured": int(epochs),
        "epochs_completed": int(epochs_completed),
        "checkpointing": {
            "validation_enabled": checkpointing,
            "model_persisted": True,
            "requested_test_policy": requested_test_policy,
            "best_epoch_checkpoint": (
                str(best_epoch_checkpoint_path)
                if best_epoch_checkpoint_path is not None
                else None
            ),
            "best_validation_checkpoint": (
                str(best_validation_checkpoint_path)
                if best_validation_checkpoint_path is not None
                else None
            ),
            "final_epoch_checkpoint": (
                str(final_trained_path)
                if final_trained_path is not None
                else None
            ),
            "manifest": (
                str(checkpoint_manifest_path)
                if checkpoint_manifest_path is not None
                else None
            ),
        },
        "execution": {
            "type": (
                "fold"
                if runtime["is_fold"]
                else "refit"
                if runtime["is_refit"]
                else "standard"
            ),
            "tag": runtime["tag"],
            "fold_index": runtime["fold_index"],
            "fold_number": runtime["fold_number"],
            "n_folds": runtime["n_folds"],
        },
    }

    if not runtime["is_refit"]:
        result["final_validation"] = {
            "epoch": int(epochs_completed),
            "metrics": final_val_metrics,
        }

    return result


# =============================================================================
# Test loop
# =============================================================================
def run_test_loop(
    model: torch.nn.Module,
    device: torch.device,
    test_loader,
    cfg: dict,
    logger,
    base_dir: Union[str, Path],
    model_name: str,
    env: str,
    task: Any = None,
    tracker: Optional[ClearMLTracker] = None,
    checkpoint_provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    One-shot test loop for final evaluation on attack-derived test data.

    This is intentionally one-shot, not epoch-based.
    """
    checkpoint_provenance = _normalize_checkpoint_provenance(
        checkpoint_provenance
    )

    logger.info(
        "\nTEST CHECKPOINT RESOLUTION\n"
        "Requested test model     : %s\n"
        "Validation checkpointing : %s\n"
        "Effective test model     : %s\n"
        "Fallback reason          : %s\n"
        "Loaded checkpoint        : %s\n"
        "Checkpoint epoch         : %d\n"
        "Checkpoint SHA-256       : %s",
        checkpoint_provenance["requested_policy"].upper(),
        (
            "ENABLED"
            if checkpoint_provenance["validation_checkpointing"]
            else "DISABLED"
        ),
        checkpoint_provenance["effective_policy"].upper(),
        checkpoint_provenance["fallback_reason"] or "none",
        checkpoint_provenance["checkpoint_path"],
        checkpoint_provenance["checkpoint_epoch"],
        checkpoint_provenance["checkpoint_sha256"],
    )

    set_seed(cfg)
    _maybe_sync_metrics_key(cfg)
    key_recovery_contract = validate_key_recovery_contract(cfg)
    monitor_metric, monitor_mode = get_monitor_metric_and_mode(cfg)
    if monitor_metric:
        logger.info("Test monitor metric: '%s' (%s)", monitor_metric, monitor_mode)

    model.eval()
    with torch.inference_mode():
        y_pred, y_true, meta = collect_logits_labels_meta(model, device, test_loader)

    if not _process_attacked(cfg, "test"):
        cfg = _strip_rank_metrics(cfg)
    wants_rank = _needs_rank_metrics(cfg)
    if wants_rank and meta is None:
        raise ValueError(
            "Test requested rank-metrics but metadata is None. "
            "Fix: ensure test_loader returns (x,y,meta) and meta includes plaintext/key."
        )

    _assert_metadata_alignment(preds=y_pred, labels=y_true, metadata=meta, where="TEST")

    metrics_cfg = cfg.get("metrics_cfg") or {}
    if not isinstance(metrics_cfg, Mapping):
        raise TypeError("cfg['metrics_cfg'] must be a mapping")
    preds_are_probs = _strict_bool(
        metrics_cfg.get("preds_are_probs", False),
        field="metrics_cfg.preds_are_probs",
    )

    if not preds_are_probs:
        _reject_probability_head(y_pred, where="Test predictions")

    cfg_test = filter_metrics_for_metadata(
        cfg,
        has_metadata=(meta is not None),
    )

    # attack i of n and the attack metrics so far, for the TraceFlow Monitor (utils/attack_progress.py)
    with attack_progress.reporting(cfg_test, base_dir):
        test_results = compute_metrics(
            y_pred,
            y_true,
            cfg_test,
            metadata=meta,
            process="test",
        )

    # Remove structured recovery evidence before ordinary metric persistence.
    test_recovery_evidence = pop_key_recovery_evidence(
        test_results
    )

    test_recovery_report = None
    test_recovery_report_path = None

    if key_recovery_contract["enabled"] and wants_rank:
        if test_recovery_evidence is None:
            raise KeyRecoveryError(
                "Test rank evaluation completed without producing "
                "key-recovery evidence. Ensure perform_attacks() returns "
                "__key_recovery_evidence__ and compute_metrics() propagates it."
            )

        test_recovery_report = build_key_recovery_report(
            cfg=cfg,
            process="test",
            evidence=test_recovery_evidence,
            model_name=model_name,
            model_hash=cfg.get("model_hash"),
            source_checkpoint=(
                checkpoint_provenance.get("checkpoint_path")
                if checkpoint_provenance
                else None
            ),
            source_checkpoint_epoch=(
                checkpoint_provenance.get("checkpoint_epoch")
                if checkpoint_provenance
                else None
            ),
            source_checkpoint_sha256=(
                checkpoint_provenance.get("checkpoint_sha256")
                if checkpoint_provenance
                else None
            ),
            selection_metric=checkpoint_provenance["selection_metric"],
            selection_score=checkpoint_provenance["selection_score"],
        )

        if checkpoint_provenance:
            test_recovery_report["model"].update(
                {
                    "checkpoint_policy_requested": checkpoint_provenance[
                        "requested_policy"
                    ],
                    "checkpoint_policy_effective": checkpoint_provenance[
                        "effective_policy"
                    ],
                    "validation_checkpointing": checkpoint_provenance[
                        "validation_checkpointing"
                    ],
                    "checkpoint_sha256": checkpoint_provenance[
                        "checkpoint_sha256"
                    ],
                    "checkpoint_manifest": checkpoint_provenance[
                        "manifest_path"
                    ],
                    "checkpoint_fallback_reason": checkpoint_provenance[
                        "fallback_reason"
                    ],
                    "validation_selection_metric": checkpoint_provenance[
                        "selection_metric"
                    ],
                    "validation_selection_mode": checkpoint_provenance[
                        "selection_mode"
                    ],
                    "validation_selection_score": checkpoint_provenance[
                        "selection_score"
                    ],
                }
            )

        test_recovery_report_path = save_test_report(
            cfg=cfg,
            report=test_recovery_report,
            model_name=model_name,
        )

        if test_recovery_report_path is not None:
            logger.info(
                "Test key-recovery report saved: %s",
                test_recovery_report_path,
            )

    try:
        append_metrics_row(
            cfg,
            test_results,
            process="test",
            epoch=1,
        )
    except Exception as e:
        # The history row is the record of the test metrics (model_judge and
        # the graph manager read it); without it the test is FAILED.
        logger.error("Failed to write the test metrics row: %s", e)
        raise

    _log_results_to_clearml(tracker, test_results, step=1, series="test", logger_obj=logger)

    # Testing is inference-only. The loaded checkpoint is immutable evidence;
    # saving <model>_tested.pt would duplicate it and imply that testing created
    # a new model artifact. That legacy behavior is deliberately removed.

    print_blocked_results(
        logger=logger,
        results=test_results,
        elapsed=None,
        eta=None,
        epoch=1,
        epochs=1,
        cfg=cfg,
        process="test",
    )

    _maybe_set_progress(task, 1.0)
    return test_results
