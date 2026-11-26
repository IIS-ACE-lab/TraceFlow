# services/graph_manager/explain.py
"""
Model-explanation figures of a trained run (graph job type: model_explanation).

A job reads the run's final model (the checkpoint the test uses, verified by
utils/checkpoint_policy.resolve_test_checkpoint) and its test traces (test.h5,
scaled by the run's stored scaler when trace_normalization.mode is fitted), and
draws one curve per model along the trace:

  weight_visualization  Zaid et al. 2020, Eqs. (4)-(5): |W| of the dense layer that
                        follows the flatten, averaged over the filters of each time
                        point, then over that layer's neurons. Weights only.
  gradient              Masure et al. 2019 (gradient visualisation): the gradient of
                        the true class's score with respect to the input,
                        captum.attr.Saliency (absolute value), over the test traces.
  gradient_x_input      Shrikumar et al. 2017, as Wouters et al. 2020 use it:
                        captum.attr.InputXGradient for the true class, over the
                        test traces.
  heatmap               Zaid et al. 2020, Eq. (6): the output of each requested
                        convolution layer averaged over its filters (a forward hook),
                        over the test traces; one figure per layer.

The score is the model's output for the trace's true label (its logit: TraceFlow's
networks end in a Linear layer, the softmax belongs to the loss). Over the test
traces the curves are averaged (aggregate: mean_abs, the default, or mean); the
papers do not state their averaging, so the job states it and sources.json
records it. PyTorch models only (deep_learning.framework: pytorch).

Author: Shraddha Mishra
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

METHODS = ("weight_visualization", "gradient", "gradient_x_input", "heatmap")
AGGREGATES = ("mean_abs", "mean")


def load_trained(cfg: Mapping[str, Any], run: Mapping[str, Any], registry_entry: Mapping[str, Any]):
    """The run's final model (eval mode, on the CPU) and its training config snapshot."""
    from services.side_channel_core.utils.sweep_test_runner import load_run_snapshot
    from utils.checkpoint_policy import resolve_test_checkpoint
    from core.model import build_model

    run_cfg = load_run_snapshot(dict(cfg), dict(registry_entry), model_name=str(run["model_name"]))
    run_cfg["model_name"] = str(run["model_name"])
    framework = str(((run_cfg.get("deep_learning") or {}).get("framework")) or "pytorch")
    if framework != "pytorch":
        raise ValueError(
            f"model_explanation reads PyTorch models; {run['model_name']} was trained with "
            f"deep_learning.framework: {framework}"
        )
    resolution = resolve_test_checkpoint(run_cfg, model_name=str(run["model_name"]))
    model_id = registry_entry.get("model_id") or run_cfg.get("model_id")
    model = build_model(run_cfg, model_id=model_id) if model_id else build_model(run_cfg)
    from utils.seal import check as check_seal

    check_seal(resolution.checkpoint_path)       # as the test: a model changed after training is refused
    state = torch.load(str(resolution.checkpoint_path), map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, run_cfg, resolution


def test_traces(run_cfg: Mapping[str, Any], limit: Optional[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """The first `limit` test traces (all when None) as the model reads them, and their labels."""
    from sca.ascad_data_loader import load_split_dataset
    from utils.trace_scaler import scale_test

    dataset = load_split_dataset(run_cfg, "test.h5", split="attack", return_metadata=False)
    scale_test(run_cfg, dataset)
    count = len(dataset) if limit is None else min(int(limit), len(dataset))
    items = [dataset[i] for i in range(count)]
    return torch.stack([x for x, _ in items]), torch.stack([y for _, y in items])


def _aggregate(values: np.ndarray, how: str) -> np.ndarray:
    return np.abs(values).mean(axis=0) if how == "mean_abs" else values.mean(axis=0)


def weight_visualization(model: torch.nn.Module, example: torch.Tensor) -> np.ndarray:
    """Zaid et al. Eqs. (4)-(5) for a channels-first flatten: W of shape (neurons, filters x time)."""
    modules = list(model.modules())
    flatten_shapes: list[torch.Size] = []
    flatten = next((m for m in modules if isinstance(m, torch.nn.Flatten)), None)
    if flatten is None:
        raise ValueError("weight_visualization needs a Flatten layer before a dense layer")
    handle = flatten.register_forward_hook(lambda module, inputs, output: flatten_shapes.append(inputs[0].shape))
    with torch.no_grad():
        model(example[:1])
    handle.remove()
    _, filters, time = flatten_shapes[0]
    after = modules[modules.index(flatten) + 1:]
    dense = next((m for m in after if isinstance(m, torch.nn.Linear)), None)
    if dense is None or dense.in_features != filters * time:
        raise ValueError("weight_visualization needs a Linear layer right after the Flatten")
    weights = dense.weight.detach().abs().reshape(dense.out_features, filters, time)
    return weights.mean(dim=1).mean(dim=0).cpu().numpy()


def attribution(model: torch.nn.Module, traces: torch.Tensor, labels: torch.Tensor, method: str,
                aggregate: str, batch: int = 256) -> np.ndarray:
    """captum attributions of the true class, aggregated over the traces: one value per sample."""
    from captum.attr import InputXGradient, Saliency

    explainer = Saliency(model) if method == "gradient" else InputXGradient(model)
    parts = []
    for start in range(0, len(traces), batch):
        x = traces[start:start + batch].clone().requires_grad_(True)
        target = labels[start:start + batch]
        values = explainer.attribute(x, target=target, abs=True) if method == "gradient" else explainer.attribute(x, target=target)
        parts.append(values.detach().reshape(len(x), -1).cpu().numpy())
    return _aggregate(np.concatenate(parts), aggregate)


def heatmaps(model: torch.nn.Module, traces: torch.Tensor, layers: Optional[Sequence[int]],
             aggregate: str) -> dict[int, np.ndarray]:
    """Zaid et al. Eq. (6): conv layer h's output averaged over its filters, then over the traces.
    Layers are counted from 1 over the network's Conv1d layers."""
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv1d)]
    if not convs:
        raise ValueError("heatmap needs Conv1d layers")
    chosen = list(range(1, len(convs) + 1)) if not layers else [int(h) for h in layers]
    for h in chosen:
        if not 1 <= h <= len(convs):
            raise ValueError(f"heatmap layer {h} is outside the model's {len(convs)} Conv1d layers")
    captured: dict[int, list[np.ndarray]] = {h: [] for h in chosen}
    handles = [convs[h - 1].register_forward_hook(
        lambda module, inputs, output, h=h: captured[h].append(output.detach().mean(dim=1).cpu().numpy()))
        for h in chosen]
    with torch.no_grad():
        for start in range(0, len(traces), 256):
            model(traces[start:start + 256])
    for handle in handles:
        handle.remove()
    return {h: _aggregate(np.concatenate(captured[h]), aggregate) for h in chosen}


__all__ = ["AGGREGATES", "METHODS", "attribution", "heatmaps", "load_trained", "test_traces", "weight_visualization"]
