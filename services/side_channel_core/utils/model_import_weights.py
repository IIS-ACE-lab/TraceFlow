# services/side_channel_core/utils/model_import_weights.py
"""
The imported run's step in place of training (src/sca/train.py, a snapshot with a
``model_import`` record; utils/model_import.py registered it).

1. The released file is the one registered: its SHA-256 again, its layer list
   checked again (no Lambda layer, nothing unpickled).
2. The trace scaler is fitted and stored exactly as a trained run's (the training
   loaders, utils/trace_scaler.scale_training), so the test's scale_test reads it.
3. The weights: a Keras HDF5 file is loaded with Keras 3 (``keras.models.load_model``,
   compile=False, through a ``.h5``-named link: Keras 3 refuses a file without
   the extension) and each layer's weights are copied, in order, into the model
   TraceFlow built from the converted layer list. Conv1D and BatchNormalization
   weights are the same in both layouts; the first Dense layer after a Flatten
   gets its rows reordered, because the released model flattens channels-last
   (index l*C + c) and TraceFlow channels-first (c*L + l). A PyTorch state dict
   is read with ``torch.load(weights_only=True)`` and copied tensor by tensor, in
   order, with their shapes checked.
4. The equality gate. Keras: on every test trace (scaled as the test scales it),
   TraceFlow's model and the released model must give the same class
   probabilities (``GATE_ATOL`` / ``GATE_RTOL``), else the import fails and no
   checkpoint is published. The gate proves the conversion, not the
   preprocessing: both models see the same scaled traces, which are the import
   config's choice. A state dict carries no network, so its gate is that every
   tensor arrived unchanged.
5. The checkpoint is published as the trainer publishes one (sealed, read back,
   atomically: epoch_utils._save_model_atomically) with its manifest
   (execution type ``model_import``), and the import report beside it
   (``<model_name>_import.json``: source, SHA-256, Keras version, gate).

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from utils.model_import import IMPORT_KEY, ModelImportError, file_sha256, inspect_keras_h5   # as src/sca/train.py

GATE_ATOL = 1.0e-5          # class probabilities: absolute
GATE_RTOL = 1.0e-3          # and relative to the released model's
REPORT_SUFFIX = "_import.json"


def _record(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    record = cfg.get(IMPORT_KEY)
    if not isinstance(record, Mapping):
        raise ModelImportError("the run has no model_import record: it is not an imported run")
    return record


def released_file(cfg: Mapping[str, Any]) -> Path:
    """The released file of the run, checked against its registered SHA-256."""
    record = _record(cfg)
    path = Path(cfg["project"]["root"]) / str(record["file"])
    if not path.is_file():
        raise ModelImportError(f"the released file {path} is missing (the import registered {record['file']})")
    digest = file_sha256(path)
    if digest != record["sha256"]:
        raise ModelImportError(f"{path} has SHA-256 {digest}, not the registered {record['sha256']}: it changed "
                               "since make import registered it")
    return path


def load_released_keras(path: Path):
    """The released Keras model, loaded by Keras 3 without compiling (its layer list checked first)."""
    from services.side_channel_core.src.core.framework_switch import keras_module

    inspect_keras_h5(path)                          # refuses Lambda and unknown layers before any load
    keras = keras_module()
    with tempfile.TemporaryDirectory(prefix="traceflow_import_") as folder:
        link = Path(folder) / "released.h5"
        os.symlink(Path(path).resolve(), link)
        return keras.models.load_model(str(link), compile=False, safe_mode=True)


def _variables(layer) -> dict:
    """A Keras layer's weights by variable name (kernel, bias, gamma, beta, moving_mean, moving_variance)."""
    out = {}
    for variable in layer.weights:
        name = str(getattr(variable, "name", "") or variable.path).split("/")[-1].split(":")[0]
        out[name] = variable
    return out


def _flatten_reorder(kernel: np.ndarray, channels: int, length: int) -> np.ndarray:
    """A Dense kernel's rows from channels-last flatten order (l*C + c) to channels-first (c*L + l)."""
    units = kernel.shape[1]
    if kernel.shape[0] != channels * length:
        raise ModelImportError(f"a Dense kernel of {kernel.shape[0]} rows follows a flatten of {channels} x {length}")
    return kernel.reshape(length, channels, units).transpose(1, 0, 2).reshape(channels * length, units)


def copy_keras_weights(released, model, weighted: list) -> int:
    """Copy every weighted released layer into TraceFlow's model; the number of tensors copied."""
    target = getattr(model, "keras_model", None)
    if target is None:
        raise ModelImportError("a Keras file imports into a Keras model (deep_learning.framework: keras)")
    have = [layer for layer in released.layers if layer.weights]
    if [layer.name for layer in have] != [w["keras"] for w in weighted]:
        raise ModelImportError(f"the released model's weighted layers {[l.name for l in have]} are not the "
                               f"registered {[w['keras'] for w in weighted]}")
    layer_names = {kind: name for kind, name in (("conv", "conv1d"), ("batchnorm", "batchnorm1d"), ("dense", "linear"))}
    copied = 0
    for source, plan in zip(have, weighted):
        dest = target.get_layer(f"layer{plan['index']}_{layer_names[plan['kind']]}")
        src_vars, dest_vars = _variables(source), _variables(dest)
        if set(src_vars) != set(dest_vars):
            raise ModelImportError(f"layer {source.name}: weights {sorted(src_vars)} do not match TraceFlow's "
                                   f"{sorted(dest_vars)}")
        for name, variable in src_vars.items():
            value = np.asarray(variable.numpy())
            if plan["kind"] == "dense" and name == "kernel" and plan.get("flatten"):
                value = _flatten_reorder(value, *plan["flatten"])
            if tuple(value.shape) != tuple(dest_vars[name].shape):
                raise ModelImportError(f"layer {source.name}.{name}: shape {value.shape}, TraceFlow's "
                                       f"{tuple(dest_vars[name].shape)}")
            dest_vars[name].assign(value)
            if not np.array_equal(np.asarray(dest_vars[name].numpy()), value):
                raise ModelImportError(f"layer {source.name}.{name}: the weights did not arrive unchanged")
            copied += 1
    return copied


def copy_state_dict(state: Mapping[str, Any], model: torch.nn.Module) -> int:
    """A PyTorch state dict's tensors, in order, into TraceFlow's model (shapes checked); the number copied."""
    target = model.state_dict()
    src = [(k, v) for k, v in state.items() if isinstance(v, torch.Tensor)]
    dest = list(target.items())
    if len(src) != len(dest):
        raise ModelImportError(f"the state dict holds {len(src)} tensors, TraceFlow's model {len(dest)}: the model "
                               "entry is not the released network")
    loaded = {}
    for (src_key, value), (dest_key, current) in zip(src, dest):
        if tuple(value.shape) != tuple(current.shape):
            raise ModelImportError(f"{src_key} {tuple(value.shape)} does not fit {dest_key} {tuple(current.shape)}")
        loaded[dest_key] = value.to(dtype=current.dtype)
    model.load_state_dict(loaded, strict=True)
    after = model.state_dict()
    if any(not torch.equal(after[k].cpu(), loaded[k].cpu()) for k in loaded):
        raise ModelImportError("the state dict's tensors did not arrive unchanged")
    return len(loaded)


def keras_gate(released, model, loader, *, output_activation: str, data_format: str, device) -> dict:
    """Both models' class probabilities on every test trace: the largest differences, and whether they agree."""
    from services.side_channel_core.src.core.framework_switch import _keras_device

    model.eval()
    worst_abs, worst_rel, traces = 0.0, 0.0, 0
    agree_top = 0
    with torch.no_grad():
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            ours = torch.softmax(model(x), dim=1).detach().cpu().numpy().astype(np.float64)
            x_released = x.permute(0, 2, 1) if data_format == "channels_last" else x
            with _keras_device(device):
                out = released(x_released, training=False)
            theirs = torch.as_tensor(np.asarray(out.detach().cpu() if hasattr(out, "detach") else out))
            if output_activation != "softmax":
                theirs = torch.softmax(theirs, dim=1)
            theirs = theirs.numpy().astype(np.float64)
            diff = np.abs(ours - theirs)
            worst_abs = max(worst_abs, float(diff.max()))
            worst_rel = max(worst_rel, float((diff / np.maximum(np.abs(theirs), 1e-30)).max()))
            agree_top += int((ours.argmax(axis=1) == theirs.argmax(axis=1)).sum())
            traces += int(x.shape[0])
            if not np.allclose(ours, theirs, atol=GATE_ATOL, rtol=GATE_RTOL):
                return {"passed": False, "traces": traces, "max_abs_diff": worst_abs, "max_rel_diff": worst_rel,
                        "same_top_class": agree_top, "atol": GATE_ATOL, "rtol": GATE_RTOL}
    return {"passed": traces > 0, "traces": traces, "max_abs_diff": worst_abs, "max_rel_diff": worst_rel,
            "same_top_class": agree_top, "atol": GATE_ATOL, "rtol": GATE_RTOL}


def publish(cfg: Mapping[str, Any], model, *, gate: Mapping[str, Any], tensors: int) -> Path:
    """The imported weights as the run's final (only) checkpoint, its manifest and the import report."""
    from utils.atomic_io import atomic_write_text
    from utils.checkpoint_policy import checkpoint_paths, write_checkpoint_manifest
    from utils.epoch_utils import _save_model_atomically

    record = _record(cfg)
    model_name = str(cfg["model_name"])
    paths = checkpoint_paths(cfg, model_name=model_name)
    _save_model_atomically(model=model, path=paths.final_epoch)
    execution = {"type": "model_import", "source": record["source"], "released_file": record["file"],
                 "released_sha256": record["sha256"], "format": record["format"], "gate_passed": bool(gate["passed"])}
    write_checkpoint_manifest(
        cfg=cfg, model_name=model_name, model_hash=cfg.get("model_hash"), validation_checkpointing=False,
        monitor_metric=None, monitor_mode=None, best_epoch=None, best_score=None, epoch_checkpoint=None,
        best_validation_checkpoint=None, final_epoch=1, final_checkpoint=paths.final_epoch, epochs_configured=1,
        execution=execution,
    )
    report = {"model_name": model_name, "model_hash": cfg.get("model_hash"), **dict(record),
              "tensors_copied": tensors, "gate": dict(gate)}
    report.pop("weighted", None)
    out = paths.model_out_dir / f"{model_name}{REPORT_SUFFIX}"
    atomic_write_text(out, json.dumps(report, indent=1) + "\n")
    return out


def import_released_weights(cfg: dict, model, device, *, logger, seed=None) -> dict:
    """Steps 1-5 of the module docstring for one imported run; the result in place of a training result."""
    from utils.data_split import get_test_loader, get_train_val_loaders

    record = _record(cfg)
    path = released_file(cfg)
    get_train_val_loaders(cfg, cfg["trainer"], seed=seed)      # fits and stores the run's scaler, as training does
    if record["format"] == "keras_h5":
        released = load_released_keras(path)
        tensors = copy_keras_weights(released, model, list(record.get("weighted") or []))
        loader = get_test_loader(cfg, cfg["trainer"], seed=seed)
        gate = keras_gate(released, model, loader, output_activation=str(record.get("output_activation") or "linear"),
                          data_format=str(record.get("data_format") or "channels_last"), device=device)
        logger.info("Model import gate: %s on %d test traces (max |dp| %.3g, max rel %.3g; same top class %d)",
                    "passed" if gate["passed"] else "FAILED", gate["traces"], gate["max_abs_diff"],
                    gate["max_rel_diff"], gate["same_top_class"])
    elif record["format"] == "torch_state_dict":
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ModelImportError(f"{path.name} is not a state dict (a mapping of tensors)")
        state = state.get("state_dict", state) if isinstance(state.get("state_dict"), Mapping) else state
        tensors = copy_state_dict(state, model)
        gate = {"passed": True, "tensors": tensors, "check": "every tensor arrived unchanged (no network to compare)"}
    else:
        raise ModelImportError(f"format {record['format']!r} is not imported")
    if not gate["passed"]:
        raise ModelImportError(
            f"the equality gate failed: TraceFlow's model and the released model differ by up to "
            f"{gate['max_abs_diff']:.3g} in class probability (tolerance {GATE_ATOL} + {GATE_RTOL} relative). "
            "Nothing was published: the conversion is not the released network"
        )
    report = publish(cfg, model, gate=gate, tensors=tensors)
    logger.info("Model import: %s (%s) published as %s, report %s", record["file"], record["sha256"][:12],
                cfg["model_name"], report)
    return {"execution": {"type": "model_import"}, "imported": True, "gate": gate, "tensors_copied": tensors}


__all__ = ["GATE_ATOL", "GATE_RTOL", "copy_keras_weights", "copy_state_dict", "import_released_weights",
           "keras_gate", "load_released_keras", "publish", "released_file"]
