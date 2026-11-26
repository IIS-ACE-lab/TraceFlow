# services/side_channel_core/utils/model_import.py
"""
Model import: test a model another study released, through TraceFlow's own test.

    make import ENV=dev PROJECT=ZvW                       # every model of config/import_config.yaml
    make import ENV=dev PROJECT=ZvW MODEL=name_a,name_b   # these imports only

Nothing trains. The project's ``config/import_config.yaml`` (master.yaml
``extra_configs.import``, else that path; the panel's Import a Model writes it)
lists the released files; params.yaml is not touched (it holds the training
recipe). Each file becomes one run, in two steps:

1. ``register`` (this module, on the host, before any container starts). The
   file is read with h5py only, never loaded: its SHA-256 (checked against a
   pinned one when given), its layer list (``model_config``), its Keras version
   and backend. A Lambda layer or a layer class TraceFlow has no mapping for is
   refused (a Lambda layer is Python code; nothing is unpickled). The layer list
   is converted into TraceFlow's neutral layer list (``convert_keras_layers``)
   under ``deep_learning.framework: keras``, the run's config is composed (the
   import's settings over the project's non-recipe sections of params.yaml, the
   chosen data entry, the fixed ``IMPORT_TRAINER``), its hash and name are made
   the usual way (``model_naming.make_experiment_hash``: the released file's
   SHA-256 joins it, so every file is its own run), and the run is entered like a
   shipped snapshot (utils/reproduce.py): its params_out snapshot written, status
   NEW, ``imported_snapshot: true``, and its ``model_import`` record.
2. The orchestrator runs it (``make orchestrate REPRODUCE_MODELS=``, as make
   reproduce): ``src/sca/train.py`` sees the snapshot's ``model_import`` block and,
   in place of the training loop, imports the weights
   (utils/model_import_weights.py: TraceFlow builds the model, the released
   weights are copied into it, the equality gate compares both models' output
   probabilities on the test traces, the checkpoint is sealed and published).
   Then the standard test runs, unchanged: metrics, key recovery, per-attack
   curves, test_registry.json.

What the file gives (read automatically): the architecture, the weights, the
Keras version and backend, the output width (256 suggests sca.leakage_model ID,
9 HW: the import names it, never guessed), the input length (it must equal the
data entry's target window). What the import must give: the file and its source
(a citation, or repository @ commit), the data entry, the preprocessing (a
TraceFlow trace_normalization and its fit set), the leakage model, and the
attack settings and seed (the blueprint's defaults otherwise).

Formats: ``keras_h5`` (a Keras HDF5 file: layer list and weights, converted
automatically) and ``torch_state_dict`` (a PyTorch state dict: weights only,
read with ``torch.load(weights_only=True)``; the import gives the TraceFlow model
entry, and the gate can only check that every tensor arrived, since the file
holds no network to compare with). ONNX is not supported: TraceFlow would need
its own ONNX-to-layer converter (no official library builds a Keras or PyTorch
module from an ONNX graph), so it is refused by name.

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from services.side_channel_core.utils.config_errors import ConfigError

CONFIG_KEY = "model_import"            # import_config.yaml's top-level block
IMPORT_KEY = "model_import"            # the run snapshot's record of the import
EXTRA_CONFIG = "import"                # master.yaml extra_configs key
DEFAULT_CONFIG = "config/import_config.yaml"
FORMATS = ("keras_h5", "torch_state_dict")
REFUSED_FORMATS = {"onnx": "ONNX is not supported (no official library turns an ONNX graph into a Keras or "
                           "PyTorch module; TraceFlow does not hand-port one)"}
FIT_ON = ("training", "profiling")
SETTING_KEYS = ("data", "trace_normalization", "fit_on", "sca", "seed", "perform_attack")
MODEL_KEYS = ("name", "file", "sha256", "source", "format", "model", *SETTING_KEYS)
ENVS = ("dev", "staging", "prod")
# The model_id of every imported run of a project: the orchestrator runs registered runs of one model_id at a time
# (MODEL_ID=imported selects them; make import passes it). Each run's architecture is its own model entry, and the
# released file's SHA-256 joins its hash, so one id for all imports never merges two of them.
IMPORT_MODEL_ID = "imported"

# An imported run trains nothing; these trainer keys are what TraceFlow's run plumbing reads (the planner,
# the data loaders, the test's output transformation). The optimizer is never built.
IMPORT_TRAINER: dict = {
    "epochs": 1,
    "batch_size": 256,
    "early_stopping": False,
    "precision": "fp32",
    "loss": {"type": "CrossEntropyLoss", "args": {"reduction": "mean", "from_logits": True}},
    "optimizer": {"type": "Adam", "learning_rate": 0.001},
    "evaluation": {"output_transformation": {"type": "Softmax", "args": {"dim": 1}}},
}

# Keras layer classes the conversion maps (convert_keras_layers); any other is refused by name.
KERAS_IMPORTABLE = (
    "InputLayer", "Conv1D", "BatchNormalization", "AveragePooling1D", "MaxPooling1D",
    "GlobalAveragePooling1D", "GlobalMaxPooling1D", "Flatten", "Dense", "Dropout", "Activation",
    "ZeroPadding1D",
)
UNSAFE_KERAS = ("Lambda",)             # Python code inside the file: never loaded


class ModelImportError(ConfigError):
    """An import cannot be carried out as configured, or its file is refused."""


# =============================================================================
# The import config
# =============================================================================
def _load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_block(project: str) -> tuple[dict, Path]:
    """master.yaml's entry of the project and its root folder."""
    from utils.config_loader import PROJECTS_ROOT, SCC_CONFIG_DIR

    master = _load_yaml(SCC_CONFIG_DIR / "master.yaml") or {}
    block = next((p for p in master.get("project") or [] if (p or {}).get("name") == project), None)
    if block is None:
        raise ModelImportError(f"No project named '{project}' in master.yaml")
    return block, (PROJECTS_ROOT / block["root"]).resolve()


def import_config_path(project: str, override: Optional[str] = None) -> Path:
    """The project's import config: ``override``, master.yaml extra_configs.import, else
    config/import_config.yaml; in the run's frozen copy of config/ when there is one."""
    from services.side_channel_core.utils.config_freeze import config_path, frozen_dir

    block, project_root = project_block(project)
    frozen = frozen_dir(project, project_root)
    rel = override or (block.get("extra_configs") or {}).get(EXTRA_CONFIG) or DEFAULT_CONFIG
    path = Path(rel)
    return config_path(path if path.is_absolute() else (project_root / path).resolve(), project_root, frozen)


def load_import_config(path: Path) -> dict:
    """model_import: of the import config, its keys checked (utils/config_schema.IMPORT)."""
    from services.side_channel_core.utils import config_schema

    path = Path(path)
    if not path.is_file():
        raise ModelImportError(
            f"Import config not found: {path}. Copy services/side_channel_core/config/blueprint/import_config.yaml "
            "(or use the panel: params.yaml -> model -> Import a Model)"
        )
    raw = _load_yaml(path) or {}
    if not isinstance(raw, Mapping) or CONFIG_KEY not in raw:
        raise ModelImportError(f"{path} must contain a top-level '{CONFIG_KEY}:' block")
    from utils.config_loader import drop_empty_keys

    raw = drop_empty_keys(dict(raw))                # an empty key is no setting, as in every config
    config_schema.check_keys(raw, config_schema.IMPORT, source=path.name)
    block = raw.get(CONFIG_KEY) or {}
    models = block.get("models") or []
    if not isinstance(models, list):
        raise ModelImportError(f"{path.name}: {CONFIG_KEY}.models must be a list of imports")
    return {"defaults": dict(block.get("defaults") or {}), "models": [dict(m or {}) for m in models]}


def _merged(defaults: Mapping[str, Any], own: Mapping[str, Any]) -> dict:
    """One import's settings: its own over the defaults (sca and perform_attack merged key by key)."""
    out = {k: copy.deepcopy(v) for k, v in defaults.items() if k in SETTING_KEYS}
    for key in SETTING_KEYS:
        if key not in own:
            continue
        if key in ("sca", "perform_attack") and isinstance(out.get(key), Mapping) and isinstance(own[key], Mapping):
            merged = copy.deepcopy(dict(out[key]))
            for sub, value in own[key].items():
                if isinstance(merged.get(sub), Mapping) and isinstance(value, Mapping):
                    merged[sub] = {**merged[sub], **copy.deepcopy(dict(value))}
                else:
                    merged[sub] = copy.deepcopy(value)
            out[key] = merged
        else:
            out[key] = copy.deepcopy(own[key])
    return out


def _file_pairs(own: Mapping[str, Any], at: str) -> list[tuple[str, Optional[str]]]:
    """The released files of one import with their SHA-256, each file beside its own: `file` (+ `sha256`)
    for one, `files: [{file, sha256}, ...]` for several (one run each, in order)."""
    if "files" in own and "file" in own:
        raise ModelImportError(f"{at}: give file (one model) or files (several), not both")
    if isinstance(own.get("file"), list) or isinstance(own.get("sha256"), list):
        raise ModelImportError(
            f"{at}: several files are listed as files: [{{file: <path>, sha256: <hex>}}, ...], each file beside its "
            "own SHA-256 (not a file list and a separate sha256 list)"
        )
    if "files" not in own:
        file = own.get("file")
        if file in (None, ""):
            raise ModelImportError(f"{at}.file is required: the released model file (a path in the project folder)")
        pin = own.get("sha256")
        return [(str(file), str(pin).strip().lower() if pin not in (None, "") else None)]
    items = own.get("files")
    if not isinstance(items, list) or not items:
        raise ModelImportError(f"{at}.files must be a list of {{file: <path>, sha256: <hex>}}")
    pairs = []
    for i, item in enumerate(items):
        if not isinstance(item, Mapping) or item.get("file") in (None, ""):
            raise ModelImportError(f"{at}.files[{i}] must be {{file: <path>, sha256: <hex>}} (sha256 optional), got {item!r}")
        unknown = sorted(set(item) - {"file", "sha256"})
        if unknown:
            raise ModelImportError(f"{at}.files[{i}]: unknown key(s) {unknown}; a file takes file and sha256")
        pin = item.get("sha256")
        pairs.append((str(item["file"]), str(pin).strip().lower() if pin not in (None, "") else None))
    return pairs


def expand_imports(config: Mapping[str, Any], *, where: str = CONFIG_KEY) -> list[dict]:
    """Every run the config makes: one per file (files: one run per file, each with its own SHA-256), with its settings."""
    runs: list[dict] = []
    for index, own in enumerate(config["models"]):
        at = f"{where}.models[{index}]"
        pairs = _file_pairs(own, at)
        source = str(own.get("source") or "").strip()
        if not source:
            raise ModelImportError(f"{at}.source is required: the citation, or the repository URL @ commit, of the release")
        name = str(own.get("name") or "").strip()
        for i, (file, pin) in enumerate(pairs):
            several = "files" in own
            runs.append({
                "where": f"{at}.files[{i}]" if several else at,
                "name": (f"{name}_{i}" if name and several else name),
                "file": file,
                "sha256": pin,
                "source": source,
                "format": str(own.get("format") or "").strip().lower() or None,
                "model": copy.deepcopy(own.get("model")),
                "settings": _merged(config["defaults"], own),
            })
    names = [r["name"] for r in runs if r["name"]]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ModelImportError(f"{where}: the run name(s) {repeated} are given twice")
    return runs


# =============================================================================
# The released file
# =============================================================================
def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_sha256(path: Path, pinned: Optional[str], *, where: str) -> str:
    """The file's SHA-256; refused when a pinned (published) one is given and differs."""
    digest = file_sha256(path)
    if pinned and str(pinned).strip().lower() != digest:
        raise ModelImportError(f"{where}: {Path(path).name} has SHA-256 {digest}, not the pinned {pinned}: the file "
                               "is not the one released")
    return digest


def released_path(project_root: Path, file: str, *, where: str) -> Path:
    """The released file, which must sit inside the project folder (the container sees projects/ only)."""
    raw = Path(file)
    path = (raw if raw.is_absolute() else Path(project_root) / raw).resolve()
    try:
        rel = path.relative_to(Path(project_root).resolve())
    except ValueError:
        raise ModelImportError(
            f"{where}: {file} is outside the project folder {project_root}. The run reads the file inside the "
            "container, which sees projects/ only: copy it into the project (e.g. released/) and give that path"
        ) from None
    if not path.is_file():
        raise ModelImportError(f"{where}: {path} does not exist")
    return rel


def detect_format(path: Path, declared: Optional[str], *, where: str) -> str:
    """keras_h5 (an HDF5 file with a model_config), torch_state_dict when declared; ONNX refused."""
    if declared in REFUSED_FORMATS or Path(path).suffix.lower().lstrip(".") in REFUSED_FORMATS:
        raise ModelImportError(f"{where}: {REFUSED_FORMATS['onnx']}")
    if declared and declared not in FORMATS:
        raise ModelImportError(f"{where}.format must be one of {list(FORMATS)}, got {declared!r}")
    import h5py

    is_h5 = h5py.is_hdf5(str(path))
    if declared == "torch_state_dict":
        if is_h5:
            raise ModelImportError(f"{where}: {path.name} is an HDF5 file, not a PyTorch state dict")
        return declared
    if not is_h5:
        raise ModelImportError(
            f"{where}: {path.name} is not an HDF5 file. A Keras model is read from its HDF5 file (.h5 / .hdf5); "
            "a PyTorch state dict needs format: torch_state_dict and its model entry"
        )
    return "keras_h5"


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def inspect_keras_h5(path: Path) -> dict:
    """What a Keras HDF5 file says about itself, read with h5py only (nothing is loaded or executed)."""
    import h5py

    with h5py.File(str(path), "r") as handle:
        attrs = dict(handle.attrs)
        if "model_config" not in attrs:
            raise ModelImportError(f"{path.name} holds no model_config: weights alone (save_weights) carry no network")
        config = json.loads(_text(attrs["model_config"]))
        weight_groups = list(handle["model_weights"].keys()) if "model_weights" in handle else []
    layers = config.get("config", {})
    layers = layers.get("layers", []) if isinstance(layers, Mapping) else layers
    if config.get("class_name") not in ("Sequential", "Model", "Functional"):
        raise ModelImportError(f"{path.name}: a {config.get('class_name')!r} model is not imported (Sequential or a "
                               "functional Model without branches)")
    names = [layer.get("config", {}).get("name") for layer in layers]
    for i, layer in enumerate(layers):                   # a functional model must be one chain
        inbound = layer.get("inbound_nodes") or []
        if config.get("class_name") == "Sequential" or i == 0:
            continue
        sources = [node[0] for group in inbound for node in group] if inbound and isinstance(inbound[0], list) else []
        if sources and sources != [names[i - 1]]:
            raise ModelImportError(f"{path.name}: layer {names[i]!r} takes {sources}, not the layer before it: only a "
                                   "chain of layers is imported")
    unsafe = [layer.get("class_name") for layer in layers if layer.get("class_name") in UNSAFE_KERAS]
    if unsafe:
        raise ModelImportError(f"{path.name}: refused, it holds a Lambda layer (Python code); TraceFlow loads no code "
                               "from a released file")
    unknown = sorted({str(layer.get("class_name")) for layer in layers} - set(KERAS_IMPORTABLE))
    if unknown:
        raise ModelImportError(f"{path.name}: layer class(es) {unknown} have no TraceFlow mapping (custom or not "
                               f"supported); importable: {', '.join(KERAS_IMPORTABLE)}")
    return {
        "keras_version": _text(attrs.get("keras_version", "")),
        "backend": _text(attrs.get("backend", "")),
        "class_name": config.get("class_name"),
        "layers": layers,
        "weight_groups": weight_groups,
    }


# =============================================================================
# Keras layer list -> TraceFlow layer list
# =============================================================================
def _one(value: Any) -> Any:
    return value[0] if isinstance(value, (list, tuple)) else value


def _batch_input_shape(layers: list) -> list:
    for layer in layers:
        config = layer.get("config") or {}
        shape = config.get("batch_input_shape") or config.get("batch_shape")     # Keras 2 | Keras 3
        if shape:
            return list(shape)
    raise ModelImportError("the released model states no input shape (batch_input_shape)")


def _pooled(length: int, size: int, stride: int, padding: str) -> int:
    if padding == "same":
        return -(-length // stride)
    return (length - size) // stride + 1


def convert_keras_layers(info: Mapping[str, Any], *, name: str = "released") -> dict:
    """The released Keras layer list as TraceFlow's neutral layer list (deep_learning.framework: keras).

    Returns {layers, input: [channels, length], outputs, output_activation, data_format, weighted}: weighted
    lists every layer with weights in order ({keras, index, kind, flatten: [channels, length] or None}), which
    the weight copy follows (utils/model_import_weights.py). A final softmax is the test's output
    transformation (TraceFlow models end in logits), so the last Dense is converted without it.
    """
    layers = [layer for layer in info["layers"] if layer.get("class_name") != "InputLayer"]
    shape = _batch_input_shape(info["layers"])
    if len(shape) != 3:
        raise ModelImportError(f"{name}: input shape {shape} is not (batch, length, channels): only 1-D traces are imported")
    data_format = "channels_last"
    for layer in layers:
        fmt = (layer.get("config") or {}).get("data_format")
        if fmt:
            data_format = fmt
            break
    channels, length = (int(shape[2]), int(shape[1])) if data_format == "channels_last" else (int(shape[1]), int(shape[2]))
    out: list[dict] = []
    weighted: list[dict] = []
    flat: Optional[list[int]] = None        # (channels, length) before the Flatten, until a Dense consumed it
    features: Optional[int] = None          # the width once the trace is flattened
    final_activation = None
    for i, layer in enumerate(layers):
        kind, c = layer["class_name"], dict(layer.get("config") or {})
        at = f"{name}: layer {c.get('name')!r} ({kind})"
        if c.get("data_format") not in (None, data_format):
            raise ModelImportError(f"{at}: mixes data formats")
        activation = c.get("activation")
        if kind == "Conv1D":
            if features is not None:
                raise ModelImportError(f"{at}: a convolution after the trace is flattened")
            padding = str(c.get("padding", "valid"))
            if padding not in ("same", "valid"):
                raise ModelImportError(f"{at}: padding {padding!r} has no TraceFlow mapping (same or valid)")
            kernel, stride, dilation = int(_one(c["kernel_size"])), int(_one(c.get("strides", 1))), int(_one(c.get("dilation_rate", 1)))
            if padding == "same" and stride != 1:
                raise ModelImportError(f"{at}: padding same with stride {stride} has no TraceFlow mapping")
            entry = {"in_channels": channels, "out_channels": int(c["filters"]), "kernel_size": kernel, "stride": stride,
                     "padding": padding if padding == "same" else 0, "dilation": dilation,
                     "bias": bool(c.get("use_bias", True))}
            if int(c.get("groups", 1) or 1) != 1:
                entry["groups"] = int(c["groups"])
            if activation not in (None, "linear"):
                entry["activation"] = activation
            out.append({"Conv1d": entry})
            weighted.append({"keras": c.get("name"), "index": len(out) - 1, "kind": "conv", "flatten": None})
            channels = int(c["filters"])
            length = length if padding == "same" else (length - dilation * (kernel - 1) - 1) // stride + 1
        elif kind == "BatchNormalization":
            if not bool(c.get("center", True)):
                raise ModelImportError(f"{at}: center false (no beta) has no TraceFlow mapping")
            axis = _one(c.get("axis", -1))
            expected = (-1, 2) if (features is None and data_format == "channels_last") else ((1,) if features is None else (-1, 1))
            if axis not in expected:
                raise ModelImportError(f"{at}: normalises axis {axis}, not the channel axis")
            entry = {"num_features": features if features is not None else channels,
                     "momentum": round(1.0 - float(c.get("momentum", 0.99)), 12), "eps": float(c.get("epsilon", 1e-3))}
            if not bool(c.get("scale", True)):
                entry["scale"] = False
            out.append({"BatchNorm1d": entry})
            weighted.append({"keras": c.get("name"), "index": len(out) - 1, "kind": "batchnorm", "flatten": None,
                             "scale": bool(c.get("scale", True))})
        elif kind in ("AveragePooling1D", "MaxPooling1D"):
            if features is not None:
                raise ModelImportError(f"{at}: pooling after the trace is flattened")
            size = int(_one(c.get("pool_size", 2)))
            stride = int(_one(c.get("strides") or size))
            padding = str(c.get("padding", "valid"))
            if padding != "valid":
                raise ModelImportError(f"{at}: pooling padding {padding!r} has no TraceFlow mapping (valid)")
            out.append({"Pooling": {"dim": 1, "type": "avg" if kind.startswith("Average") else "max",
                                    "kernel_size": size, "stride": stride}})
            length = _pooled(length, size, stride, padding)
        elif kind in ("GlobalAveragePooling1D", "GlobalMaxPooling1D"):
            out.append({"Pooling": {"dim": 1, "type": "global_avg" if "Average" in kind else "global_max"}})
            features, length = channels, 1
        elif kind == "Flatten":
            flat = [channels, length] if data_format == "channels_last" and channels > 1 and length > 1 else None
            features = channels * length
            out.append({"Flatten": {}})
        elif kind == "Dense":
            if features is None:
                raise ModelImportError(f"{at}: a Dense layer on the unflattened trace acts on its last axis, which "
                                       "differs between the two layouts")
            units = int(c["units"])
            entry = {"in_features": features, "out_features": units, "bias": bool(c.get("use_bias", True))}
            last = i == len(layers) - 1 or all(l["class_name"] in ("Activation", "Dropout") for l in layers[i + 1:])
            if activation == "softmax" and not last:
                raise ModelImportError(f"{at}: softmax before the last layer")
            if activation not in (None, "linear", "softmax"):
                entry["activation"] = activation
            if last:
                final_activation = activation or "linear"
            out.append({"Linear": entry})
            weighted.append({"keras": c.get("name"), "index": len(out) - 1, "kind": "dense", "flatten": flat})
            flat, features = None, units
        elif kind == "Dropout":
            out.append({"Dropout": {"p": float(c.get("rate", 0.5))}})
        elif kind == "ZeroPadding1D":
            pad = c.get("padding", 1)
            pad = [int(pad), int(pad)] if isinstance(pad, int) else [int(v) for v in pad]
            out.append({"ZeroPad1d": {"padding": pad}})
            length += sum(pad)
        elif kind == "Activation":
            if not out:
                raise ModelImportError(f"{at}: an activation before any layer")
            previous = next(iter(out[-1].values()))
            if activation == "softmax":
                if i != len(layers) - 1:
                    raise ModelImportError(f"{at}: softmax before the last layer")
                final_activation = "softmax"
            elif "activation" in previous or not any(k in out[-1] for k in ("Conv1d", "Linear")):
                raise ModelImportError(f"{at}: an activation layer that follows no Conv1D or Dense of its own")
            else:
                previous["activation"] = activation
        else:                                               # inspect_keras_h5 refused it already
            raise ModelImportError(f"{at}: no TraceFlow mapping")
    if features is None or not weighted or weighted[-1]["kind"] != "dense":
        raise ModelImportError(f"{name}: the model ends in no Dense layer on the flattened trace (the class scores)")
    shape_in = [int(shape[2]), int(shape[1])] if data_format == "channels_last" else [int(shape[1]), int(shape[2])]
    return {"layers": out, "input": shape_in, "outputs": features, "output_activation": final_activation or "linear",
            "data_format": data_format, "weighted": weighted}


def leakage_suggestion(outputs: Optional[int]) -> Optional[str]:
    """The leakage model an output width suggests (the import names it; never set from this)."""
    return {256: "ID", 9: "HW", 2: "MSB"}.get(int(outputs)) if outputs else None


# =============================================================================
# The run's config
# =============================================================================
def _input_length(data: Mapping[str, Any]) -> Optional[int]:
    window = data.get("target_range")
    if isinstance(window, (list, tuple)) and len(window) >= 2:
        return int(window[1]) - int(window[0])
    return None


def model_entry(run: Mapping[str, Any], conversion: Optional[Mapping[str, Any]], model_id: str) -> dict:
    """The run's one model entry: the converted layer list, or (torch_state_dict) the import's own entry."""
    if conversion is not None:
        layers = conversion["layers"]
        architecture = "CNN" if any("Conv1d" in entry for entry in layers) else "MLP"
        return {"model_id": model_id, "architecture": architecture,
                "architecture_args": {"strict_graph_args": True, "layers": copy.deepcopy(layers)}}
    entry = run.get("model")
    if not isinstance(entry, Mapping) or not entry:
        raise ModelImportError(f"{run['where']}: format torch_state_dict needs model: the TraceFlow model entry of "
                               "the network (as a params.yaml model), since a state dict holds weights only")
    entry = copy.deepcopy(dict(entry))
    entry["model_id"] = model_id
    return entry


def compose_run_cfg(project: str, run: Mapping[str, Any], *, record: Mapping[str, Any],
                    conversion: Optional[Mapping[str, Any]], env: str) -> dict:
    """The imported run's config, as a planned run's: the project's sections of params.yaml that are not the
    training recipe, the chosen data entry with the import's preprocessing, the converted model, the import's
    sca, seed and attack, IMPORT_TRAINER, no cross-validation, the final (only) checkpoint tested."""
    from services.side_channel_core.src.sca.dataset_types import check_run_settings
    from services.side_channel_core.src.sca.test_request import _check_attack_block
    from utils.cli_helpers import select_data_entry
    from utils.config_loader import get_project_config

    settings = run["settings"]
    at = run["where"]
    cfg = get_project_config(project_name=project)
    cfg.pop(EXTRA_CONFIG, None)                          # the import config itself is no run setting
    dataset = settings.get("data")
    if not dataset:
        raise ModelImportError(f"{at}: data is required: the name of one of the project's data entries "
                               f"({', '.join(map(str, (cfg.get('data') or {}).keys()))})")
    try:
        select_data_entry(cfg, str(dataset), "make import")
    except ValueError as exc:
        raise ModelImportError(f"{at}: {exc}") from None
    data = cfg["data"]
    if settings.get("trace_normalization") is not None:
        data["trace_normalization"] = copy.deepcopy(settings["trace_normalization"])
    fitted = str((data.get("trace_normalization") or {}).get("mode") or "none").strip().lower() == "fitted"
    fit_on = settings.get("fit_on")
    if fit_on is not None and not fitted:
        raise ModelImportError(f"{at}.fit_on applies to a fitted trace_normalization only (mode: fitted); the "
                               f"entry's is {(data.get('trace_normalization') or {}).get('mode', 'none')!r}")
    fit_on = str(fit_on or "training")
    if fit_on not in FIT_ON:
        raise ModelImportError(f"{at}.fit_on must be one of {list(FIT_ON)}, got {fit_on!r}")
    # The training split serves the scaler only (nothing trains): kept for fit_on training, dropped when the
    # scaler fits every profiling trace (before any split, as Zaid's and Wouters' scripts) or nothing is fitted.
    if not fitted or fit_on == "profiling":
        data.pop("profiling_split", None)
    sca = dict(cfg.get("sca") or {})
    own_sca = dict(settings.get("sca") or {})
    if not own_sca.get("leakage_model"):
        hint = leakage_suggestion((conversion or {}).get("outputs"))
        raise ModelImportError(
            f"{at}: sca.leakage_model is required (the labels the released model predicts)"
            + (f"; its {conversion['outputs']} outputs suggest {hint}" if hint else "")
        )
    sca.update(own_sca)
    cfg["sca"] = sca
    if "seed" not in settings:
        raise ModelImportError(f"{at}: seed is required: it seeds the attack orders and tie-breaks of the test "
                               "(a number), or null for an unseeded test")
    cfg["seed"] = settings["seed"]
    attack = copy.deepcopy(cfg.get("perform_attack") or {})
    own_attack = _check_attack_block(settings.get("perform_attack"), where=f"{at}.perform_attack")
    if "test" in own_attack:
        attack["test"] = {**dict(attack.get("test") or {}), **own_attack["test"]}
    for key in ("attack_step", "ge_thresholds"):
        if key in own_attack:
            attack[key] = own_attack[key]
    # Nothing trains, so nothing is validated: the validation split is not attacked (attack_times 0).
    attack["validation"] = {**dict(attack.get("validation") or {}), "attack_times": 0}
    cfg["perform_attack"] = attack
    evaluation = copy.deepcopy(cfg.get("evaluation") or {})
    evaluation["validation"] = {**dict(evaluation.get("validation") or {}), "checkpointing": False}
    evaluation["test"] = {**dict(evaluation.get("test") or {}), "load_model": "final_epoch"}
    cfg["evaluation"] = evaluation
    cfg["cross_validation"] = {"enabled": False}
    cfg["trainer"] = copy.deepcopy(IMPORT_TRAINER)
    for key in ("auto_search", "auto_mlops"):           # an import is no search
        if isinstance(cfg.get(key), Mapping):
            cfg[key] = {**cfg[key], "enabled": False}
    model_id = IMPORT_MODEL_ID
    if any(isinstance(m, Mapping) and str(m.get("model_id")) == IMPORT_MODEL_ID for m in cfg.get("model") or []):
        raise ModelImportError(f"params.yaml has a model named {IMPORT_MODEL_ID!r}, the model_id of every imported "
                               "run: rename it")
    if record["format"] == "keras_h5":
        cfg["deep_learning"] = {"framework": "keras"}
        length = _input_length(data)
        if length is not None and conversion is not None and conversion["input"][1] != length:
            raise ModelImportError(
                f"{at}: the released model takes traces of {conversion['input'][1]} samples, the data entry "
                f"{dataset!r} gives {length} (its target_range): choose the entry its traces came from"
            )
    else:
        framework = (cfg.get("deep_learning") or {}).get("framework")
        cfg["deep_learning"] = {"framework": "pytorch"} if framework in (None, "pytorch") else cfg["deep_learning"]
    cfg["model"] = [model_entry(run, conversion, model_id)]
    cfg["model_id"] = model_id
    cfg[IMPORT_KEY] = copy.deepcopy(dict(record))
    cfg["env"] = env
    check_run_settings(cfg)
    return cfg


def resolve_run(cfg: dict, project_root: Path, *, env: str, name: Optional[str], reserved: set) -> tuple[dict, str, str]:
    """The composed config made concrete as the planner makes a run (one sweep point, the model contract,
    the identity): (run_cfg, model_name, model_hash). A list left in a section the import takes from
    params.yaml would be a sweep: refused."""
    from utils.sweep_engine import _scc_identity, _sweep_dimensions, determine_run_identity, generate_param_combinations

    def dotted(path: Any) -> str:                        # a linked group is a tuple of paths
        parts = path if path and isinstance(path[0], tuple) else (path,)
        return " + ".join(".".join(map(str, p)) for p in parts)

    swept = [dotted(path) for path, values in _sweep_dimensions(cfg) if len(values) > 1]
    if swept:
        raise ModelImportError(
            f"params.yaml sweeps {swept}: an import is one run per file. Set one value for it in import_config.yaml "
            "(sca, seed, perform_attack) or in params.yaml"
        )
    combos = list(generate_param_combinations(copy.deepcopy(cfg)))
    if len(combos) != 1:
        raise ModelImportError(f"the import's config makes {len(combos)} runs, not one (a list left in params.yaml)")
    base = copy.deepcopy(cfg)
    base["project"].setdefault("root", str(project_root))
    run_cfg, _name, _hash = _scc_identity(combos[0], base, Path(project_root), str(cfg["model_id"]), env, reserved=reserved)
    run_cfg[IMPORT_KEY] = copy.deepcopy(cfg[IMPORT_KEY])
    run_cfg.pop("model_name", None)
    run_cfg.pop("model_hash", None)
    model_name, model_hash = determine_run_identity(run_cfg, name or None, persist=False, reserved=reserved)
    return run_cfg, model_name, model_hash


def prepare(project: str, *, env: str, names: Iterable[str] = (), config: Optional[str] = None) -> list[dict]:
    """Every selected import checked and composed, nothing written: [{run, cfg, model_name, model_hash, info}]."""
    _block, project_root = project_block(project)
    path = import_config_path(project, config)
    runs = expand_imports(load_import_config(path), where=f"{path.name}: {CONFIG_KEY}")
    wanted = [n for n in (str(x).strip() for x in names) if n]
    out: list[dict] = []
    reserved: set = set()
    for run in runs:
        rel = released_path(project_root, run["file"], where=run["where"])
        absolute = project_root / rel
        fmt = detect_format(absolute, run["format"], where=run["where"])
        digest = verified_sha256(absolute, run["sha256"], where=run["where"])
        info, conversion = {}, None
        if fmt == "keras_h5":
            info = inspect_keras_h5(absolute)
            conversion = convert_keras_layers(info, name=str(rel))
        record = {
            "file": str(rel), "sha256": digest, "format": fmt, "source": run["source"],
            **({"keras_version": info["keras_version"], "backend": info["backend"],
                "output_activation": conversion["output_activation"], "data_format": conversion["data_format"],
                "weighted": conversion["weighted"]} if conversion else {}),
        }
        cfg = compose_run_cfg(project, run, record=record, conversion=conversion, env=env)
        run_cfg, model_name, model_hash = resolve_run(cfg, project_root, env=env, name=run["name"], reserved=reserved)
        reserved.add(model_name)
        if wanted and model_name not in wanted and model_hash not in wanted and run["name"] not in wanted:
            continue
        out.append({"run": run, "cfg": run_cfg, "model_name": model_name, "model_hash": model_hash, "record": record})
    if wanted:
        found = {x for r in out for x in (r["model_name"], r["model_hash"], r["run"]["name"])}
        missing = [n for n in wanted if n not in found]
        if missing:
            raise ModelImportError(f"MODEL={','.join(missing)}: no import of {path.name} has that name")
    return out


def register(project: str, *, env: str, names: Iterable[str] = (), force: bool = False,
             config: Optional[str] = None) -> list[str]:
    """Enter every selected import as a run (snapshot, run_registry.json, model_name_map.json), as make
    reproduce enters a shipped snapshot; the names still to import (DONE ones only with force)."""
    from utils.batch import add_batch, current_batch, utc_stamp
    from utils.data_reproducibility import write_params_snapshot
    from utils.model_naming import _load_mapping, _save_mapping
    from utils.path_utils import resolve_path
    from utils.sweep_engine import load_registry, save_registry
    from utils.reproduce import IMPORTED_KEY

    if env not in ENVS:
        raise ModelImportError(f"env must be one of {ENVS}, got {env!r}")
    _block, project_root = project_block(project)
    planned = prepare(project, env=env, names=names, config=config)
    if not planned:
        raise ModelImportError("import_config.yaml lists no model to import")
    registry = load_registry(project_root)
    name_map = _load_mapping(project_root)
    taken = {v: k for k, v in name_map.items()}
    todo: list[str] = []
    for item in planned:
        model_name, model_hash = item["model_name"], item["model_hash"]
        if taken.get(model_name) not in (None, model_hash):
            raise ModelImportError(f"{item['run']['where']}: the name {model_name!r} is another run's here")
        entry = registry.get(model_hash)
        status = str((entry or {}).get("status") or "").strip().lower()
        if entry is not None and status == "done" and not force:
            continue
        cfg = copy.deepcopy(item["cfg"])
        cfg["model_name"], cfg["model_hash"] = model_name, model_hash
        params_out = Path(resolve_path(cfg["paths"]["params_out"], cfg, model_name=model_name))
        write_params_snapshot(cfg, params_out)
        if entry is None:
            entry = {
                "model_hash": model_hash, "model_name": model_name, "model_id": cfg.get("model_id", ""),
                "env": env, "status": "NEW", "leakage_model": (cfg.get("sca") or {}).get("leakage_model", ""),
                "path": f"{project_root.name}/artifacts/{env}/generated_models/{model_name}",
                "selection_source": "model_import", IMPORTED_KEY: True,
                IMPORT_KEY: {k: item["record"][k] for k in ("file", "sha256", "format", "source")},
            }
            batch = current_batch()
            if batch:
                add_batch(entry, *batch)
            add_batch(entry, "import", f"import-{utc_stamp()}")
            registry[model_hash] = entry
        name_map[model_hash] = model_name
        todo.append(model_name)
    save_registry(registry, project_root)
    _save_mapping(project_root, name_map)
    return todo


def is_imported(cfg: Mapping[str, Any]) -> bool:
    """A run made by a model import (its snapshot carries the model_import record)."""
    return isinstance(cfg, Mapping) and isinstance(cfg.get(IMPORT_KEY), Mapping)


def describe(path: Path) -> dict:
    """What a released file says about itself, for the panel's Import a Model: format, SHA-256, layers,
    input, outputs and the leakage model they suggest, or why it is refused."""
    path = Path(path)
    out: dict[str, Any] = {"file": str(path)}
    try:
        out["sha256"] = file_sha256(path)
        fmt = detect_format(path, None, where=path.name)
        info = inspect_keras_h5(path)
        conversion = convert_keras_layers(info, name=path.name)
        out.update(format=fmt, keras_version=info["keras_version"], backend=info["backend"],
                   layers=conversion["layers"], input=conversion["input"], outputs=conversion["outputs"],
                   output_activation=conversion["output_activation"],
                   leakage_suggestion=leakage_suggestion(conversion["outputs"]))
    except (ModelImportError, OSError, ValueError, KeyError) as exc:
        out["error"] = str(exc)
    return out


# =============================================================================
# Command line (make import)
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m utils.model_import", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="enter the imports as runs; prints the names to run, last line")
    reg.add_argument("--project", required=True)
    reg.add_argument("--env", default="dev")
    reg.add_argument("--model", default="", help="comma-separated run names or hashes (default: every import)")
    reg.add_argument("--config", default=None, help="another import config, relative to the project root")
    reg.add_argument("--force", action="store_true", help="import DONE runs again")
    ins = sub.add_parser("describe", help="what a released file says about itself (JSON)")
    ins.add_argument("file")
    args = parser.parse_args(argv)
    if args.command == "describe":
        print(json.dumps(describe(Path(args.file))))
        return 0
    from utils.reproduce import running_experiments

    busy = running_experiments(Path(__file__).resolve().parents[3])
    if busy:
        print("An experiment is running on this machine; let it finish first (make import starts the stack, which "
              "stops it):\n  " + "\n  ".join(busy), file=sys.stderr)
        return 3
    try:
        names = register(args.project, env=args.env, names=[n for n in args.model.split(",") if n.strip()],
                         force=args.force, config=args.config)
    except ConfigError as exc:
        print(f"make import: {exc}", file=sys.stderr)
        return 2
    print(f"to import: {len(names)} run(s)")
    print(",".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
