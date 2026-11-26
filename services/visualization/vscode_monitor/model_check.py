# services/visualization/vscode_monitor/model_check.py
"""
A params.yaml model checked as a run would build it: core.model.build_model (TraceFlow's own builder,
torch.nn layers) on one zero trace of the data entry's length, each layer's output shape, or the
error the build or the forward pass raised and at which layer. The panel's [Check the model].

    python -m services.visualization.vscode_monitor.model_check --project P --model-id M [--dataset D]
    python -m services.visualization.vscode_monitor.model_check --project P --layers-json - [--dataset D | --input LENGTH]

Prints one JSON object: {model_id, framework, dataset, input: [channels, length], params, layers:
[{index, type, out}], error, at}. The shapes are PyTorch's (deep_learning.framework: keras builds the
same layers, Keras on its torch backend); the first value of a sweep list is the one built.

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Optional


PLANNED = "__planned_model__"          # the id an unsaved design is built under


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def check(project: str, model_id: str, dataset: Optional[str] = None) -> dict[str, Any]:
    """A saved model of params.yaml (the panel's [Check the model]). Without a data entry named, the entry the
    model fits (owner 2026-10-23: no choice to make and nothing read from the queues): the entries' distinct
    trace lengths are tried in the file's order and the first whose build passes is the model's; when none
    passes, the first entry's result says why."""
    from services.side_channel_core.utils.config_loader import get_project_config

    cfg = copy.deepcopy(get_project_config(project, validate_data=False))
    model = next((m for m in cfg.get("model") or [] if isinstance(m, dict) and m.get("model_id") == model_id), None)
    if dataset is not None or model is None:
        return _check(cfg, model, model_id, dataset)
    first, chosen, fits, seen = None, None, [], set()
    for entry in _entries(cfg):
        length = _length_of(entry)
        if length in seen or not entry.get("name"):
            continue
        seen.add(length)
        out = _check(copy.deepcopy(cfg), model, model_id, str(entry["name"]))
        first = first or out
        if out["error"] is None:
            fits.append(str(entry["name"]))              # one entry per trace length the model builds on
            chosen = chosen or out
    if chosen is not None:
        return {**chosen, "fits": fits}                  # more than one: the panel says so (the same flatten width)
    return first if first is not None else _check(cfg, model, model_id, None)


def check_layers(project: str, layers: list, architecture: str = "CNN", dataset: Optional[str] = None,
                 input_length: Optional[int] = None) -> dict[str, Any]:
    """A layer list not in params.yaml yet (the planner's design), checked exactly as a saved model is. A project
    without a data entry yet: input_length, the trace length typed on the planner page, stands in for the entry."""
    from services.side_channel_core.utils.config_loader import get_project_config

    cfg = copy.deepcopy(get_project_config(project, validate_data=False))
    model = {"model_id": PLANNED, "architecture": architecture, "architecture_args": {"layers": copy.deepcopy(layers)}}
    cfg["model"] = [m for m in cfg.get("model") or [] if isinstance(m, dict) and m.get("model_id") != PLANNED] + [model]
    out = _check(cfg, model, PLANNED, dataset, input_length)
    out["model_id"] = None
    return out


def _entries(cfg: dict) -> list[dict]:
    """The data entries as the loaded config holds them ({name: entry}), or as params.yaml lists them."""
    data = cfg.get("data")
    if isinstance(data, dict) and "database" not in data:
        return [{"name": k, **v} for k, v in data.items() if isinstance(v, dict)]
    return data if isinstance(data, list) else [data] if isinstance(data, dict) else []


def _length_of(entry: dict) -> Optional[int]:
    """The trace length a model of this entry sees (core.model._input_length: target_range), or None."""
    r = entry.get("target_range")
    if isinstance(r, (list, tuple)) and len(r) >= 2:
        try:
            return int(r[1]) - int(r[0])
        except (TypeError, ValueError):
            return None
    return None


def describe(project: str, params_file: str, dataset: Optional[str] = None, model_id: Optional[str] = None) -> dict[str, Any]:
    """What the planner opens with: the run's framework, the trace input its layers see, the data entry, every
    data entry with its trace length (the page's Input menu), and the models of params.yaml as the file holds
    them (the arrange edits index into the file's lists). No data entry: input None (the page asks for a length).
    With model_id (Visualise): the entry that model fits (check), so the page opens on the right length."""
    import yaml

    from services.side_channel_core.utils.config_loader import get_project_config

    probe = check(project, model_id, dataset) if model_id else check_layers(project, [{"Flatten": {}}], "MLP", dataset)
    if model_id and probe["input"] is None and probe["error"]:                  # the model builds nowhere: still the first entry's length
        probe = {**check_layers(project, [{"Flatten": {}}], "MLP", dataset), "error": probe["error"]}
    entries = [{"name": str(e.get("name")), "length": _length_of(e)}
               for e in _entries(copy.deepcopy(get_project_config(project, validate_data=False)))]
    with open(params_file, encoding="utf-8") as handle:
        models = (yaml.safe_load(handle) or {}).get("model") or []
    return {"framework": probe["framework"], "input": probe["input"], "dataset": probe["dataset"], "entries": entries,
            "models": [m for m in models if isinstance(m, dict)],
            "error": None if probe["input"] is not None else
            "no data entry in params.yaml yet: type the trace length the first layer sees (Add a Dataset gives it)" if not entries
            else probe["error"]}


def _check(cfg: dict, model: Optional[dict], model_id: str, dataset: Optional[str], input_length: Optional[int] = None) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch

    from core.framework_switch import keras_input_shape
    from core.model import _input_length, build_model
    from utils.parameter_count import trainable_parameter_count

    framework = str((cfg.get("deep_learning") or {}).get("framework") or "pytorch")
    # the loaded config keys the data entries by name ({ascad: {...}}); params.yaml lists them
    entries = _entries(cfg)
    entry = next((d for d in entries if isinstance(d, dict) and (dataset is None or d.get("name") == dataset)), None)
    if entry is None and dataset is None and input_length:
        entry = {"name": None, "target_range": [0, int(input_length)]}   # a typed length stands in for the entry
    out: dict[str, Any] = {"model_id": model_id, "framework": framework, "dataset": (entry or {}).get("name"),
                           "input": None, "params": None, "layers": [], "error": None, "at": None}
    if model is None:
        out["error"] = f"no model_id {model_id!r} in params.yaml"
        return out
    if entry is None:
        out["error"] = (f"no data entry {dataset!r} in params.yaml" if dataset is not None
                        else "no data entry in params.yaml yet: type the trace length the first layer sees (Add a Dataset gives it)")
        return out
    # one concrete run: this data entry, the sweep lists' first values, the PyTorch build of the layers
    cfg["data"] = entry
    cfg["model_id"] = model_id
    cfg["sca"] = {k: _first(v) for k, v in (cfg.get("sca") or {}).items()}
    cfg.setdefault("deep_learning", {})["framework"] = "pytorch"
    layer_defs = list(((model.get("architecture_args") or {}).get("layers")) or [])
    channels, length = keras_input_shape(layer_defs, _input_length(cfg))
    out["input"] = [channels, length]
    try:
        net = build_model(cfg, model_id)
    except Exception as exc:      # noqa: BLE001 - the builder's own message is the answer
        out["error"] = str(exc)
        return out
    out["params"] = trainable_parameter_count(net)
    x = torch.zeros(2, channels, length)
    if not isinstance(net, torch.nn.Sequential) or not layer_defs:            # a graph or a named model: its output
        try:
            with torch.no_grad():
                y = net(x)
            out["layers"] = [{"index": None, "type": "output", "out": list(y.shape[1:])}]
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
        return out
    try:
        with torch.no_grad():
            for index, name, x in walk(net, layer_defs, x):
                out["layers"].append({"index": index, "type": name, "out": list(x.shape[1:])})
    except LayerFailed as failed:
        out["error"], out["at"] = str(failed.cause), failed.index
        return out
    if framework == "keras":
        _keras_build(cfg, model_id, out)
    return out


class LayerFailed(Exception):
    """A layer's forward pass raised: which layer (its index in the list), and why."""

    def __init__(self, index: int, cause: Exception):
        super().__init__(str(cause))
        self.index, self.cause = index, cause


def walk(net: Any, layer_defs: list, x: Any):
    """Each layer's output on x, in order: (index, name, output). build_model appends a layer's module, then
    its activation's when it names one; a Keras model (core.framework_switch.KerasModel) names them
    layer<i>_<name> and layer<i>_activation. LayerFailed at the layer whose forward pass raises."""
    keras_model = getattr(net, "keras_model", None)
    if keras_model is not None:
        groups: dict[int, list] = {}
        for layer in keras_model.layers:
            groups.setdefault(int(layer.name.split("_", 1)[0][len("layer"):]), []).append(layer)
        steps = [groups.get(index, []) for index in range(len(layer_defs))]
        call = lambda step, value: step(value, training=False)   # noqa: E731 - inference: Dropout off, BatchNorm's running statistics
    else:
        modules, at, steps = list(net.children()), 0, []
        for layer_def in layer_defs:
            n = 2 if (next(iter(layer_def.values())) or {}).get("activation") is not None else 1
            steps.append(modules[at:at + n])
            at += n
        call = lambda step, value: step(value)                   # noqa: E731
    for index, (layer_def, step) in enumerate(zip(layer_defs, steps)):
        try:
            for module in step:
                x = call(module, x)
        except Exception as exc:  # noqa: BLE001
            raise LayerFailed(index, exc) from exc
        yield index, next(iter(layer_def)), x


def _keras_build(cfg: dict, model_id: str, out: dict[str, Any]) -> None:
    """Under deep_learning.framework: keras a run builds the layers with Keras: build them so, report its error
    (and the layer it names) and count Keras's trainable weights; the shapes stay the PyTorch pass's."""
    import re

    from core.model import build_model
    from utils.parameter_count import trainable_parameter_count

    keras_cfg = copy.deepcopy(cfg)
    keras_cfg["deep_learning"]["framework"] = "keras"
    try:
        net = build_model(keras_cfg, model_id)
    except Exception as exc:      # noqa: BLE001 - the builder's own message is the answer
        out["error"] = str(exc)
        named = re.search(r"layer (\d+)", str(exc))
        out["at"] = int(named.group(1)) if named else None
        return
    out["params"] = trainable_parameter_count(net)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a params.yaml model, or an unsaved layer list: each layer's output shape.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--model-id", default=None, help="a saved model of params.yaml")
    parser.add_argument("--layers-json", default=None, help="'-': an unsaved layer list, as JSON on stdin (the planner)")
    parser.add_argument("--architecture", default="CNN", help="with --layers-json: the model's kind")
    parser.add_argument("--describe", default=None, metavar="PARAMS_FILE",
                        help="the planner's start: framework, input, data entry and the models of PARAMS_FILE")
    parser.add_argument("--dataset", default=None, help="the data entry whose trace length is used (default: the first)")
    parser.add_argument("--input", default=None, type=int, metavar="LENGTH",
                        help="with --layers-json, a project without a data entry: the trace length typed on the planner page")
    args = parser.parse_args(argv)
    if sum(x is not None for x in (args.layers_json, args.describe)) > 1 or (args.model_id is None and args.layers_json is None and args.describe is None):
        parser.error("give --model-id, --layers-json -, or --describe PARAMS_FILE [--model-id M]")
    try:
        out = (describe(args.project, args.describe, args.dataset, args.model_id) if args.describe is not None
               else check(args.project, args.model_id, args.dataset) if args.model_id is not None
               else check_layers(args.project, json.load(sys.stdin), args.architecture, args.dataset, args.input))
    except Exception as exc:      # noqa: BLE001
        out = {"model_id": args.model_id, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out))
    return 0 if not out.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
