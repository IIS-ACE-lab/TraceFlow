# services/side_channel_core/src/core/framework_switch.py
"""
deep_learning.framework: which library a run's deep-learning functions come from.

params.yaml names every deep-learning function once, in neutral words: the
layers (Linear, Conv1d, Pooling, Flatten, Dropout, BatchNorm1d), the
activations (relu, selu, hard_sigmoid, ...), the loss (CrossEntropyLoss), the
optimizer (RMSprop, Adam, ...), the scheduler and the initialisers. This
module decides which library implements them. Nothing is re-implemented
here: every function is the library's own.

    framework: pytorch   torch.nn layers and activations, torch.nn.init
                         initialisers, torch.nn.CrossEntropyLoss / NLLLoss,
                         torch.optim optimizers, torch.optim.lr_scheduler
                         schedulers, torch.nn.utils.clip_grad_norm_.
    framework: keras     Keras 3 (keras 3.12.4): keras.layers, keras.activations,
                         keras.initializers, keras.losses.CategoricalCrossentropy,
                         keras.optimizers, keras.optimizers.schedules, and the
                         optimizers' own global_clipnorm. Keras runs on its
                         PyTorch backend (KERAS_BACKEND=torch), the backend Keras
                         ships for training inside a PyTorch program: the Keras
                         model is a torch.nn.Module, its gradients come from
                         torch autograd, and keras.optimizers update the same
                         parameter tensors. So data loading, metrics, attacks,
                         cross-validation, snapshots and the model hash are the
                         same for both frameworks.

What framework: keras is and is not
    The definitions are Keras 3's: its formulas, defaults and update rules
    (hard_sigmoid = relu6(x + 3) / 6, RMSprop with epsilon 1e-7 inside the
    square root, glorot_uniform kernels with zero biases, BatchNormalization
    momentum 0.99 and epsilon 1e-3, ...). The arithmetic is PyTorch's kernels
    on the run's device, as for any Keras model trained on the torch backend;
    a TensorFlow-backend run computes the same functions with TensorFlow's
    kernels, which round differently in the last bits. Keras 3 is the only
    Keras version available here: the ASCAD paper's Keras 2.1.1 (TensorFlow
    1.4, Python <= 3.6) cannot be installed next to this system, so its
    definitions where Keras 3 changed them (hard_sigmoid 0.2 x + 0.5,
    RMSprop epsilon 1e-8 outside the root, lecun_normal without the
    truncation correction) are not available.

Configuration (params.yaml)
    deep_learning:
      framework: keras            # pytorch (default) | keras; a list sweeps it
      pytorch:                    # read when framework is pytorch
        weight_init: xavier_uniform      # a torch.nn.init function, or default
        weight_init_args: {gain: 1.0}    # its arguments
        bias_init: zeros                 # a torch.nn.init function, or default
        bias_init_args: {}
      keras:                      # read when framework is keras
        weight_init: glorot_uniform      # a keras.initializers name
        weight_init_args: {}
        bias_init: zeros
        bias_init_args: {}
        optimizer_args:                  # keras.optimizers arguments, by optimizer type
          RMSprop: {rho: 0.9, epsilon: 1.0e-7, momentum: 0.0, centered: false}

    Initialiser names are the framework's own: torch.nn.init has
    xavier_uniform, kaiming_normal, ... and no lecun_*; keras.initializers
    has glorot_uniform, he_normal, lecun_normal, ... "default" under pytorch
    keeps the layer's own reset_parameters() (kaiming_uniform with a =
    sqrt(5) weights, uniform bias). No seed argument: the initialisers draw
    from the run's seed.

    A model block may override the initialisers for itself with
    architecture_args.weight_init / bias_init, either one name of the active
    framework or one entry per framework:
        weight_init:
          pytorch: {name: kaiming_normal, args: {nonlinearity: linear}}
          keras: {name: lecun_normal}

    Optimizer arguments live with the framework whose names they use:
    deep_learning.<framework>.optimizer_args maps an optimizer type
    (trainer.optimizer[].type, matched case-insensitively) to that class's own
    arguments (torch.optim RMSprop alpha/eps, keras.optimizers RMSprop
    rho/epsilon, ...). trainer.optimizer[] holds type, learning_rate,
    weight_decay and scheduler only; wire_optimizer_args() copies the selected
    framework's arguments into the run's trainer.optimizer.args when the sweep
    selects an optimizer type (a type without an entry gets {}: the library's
    defaults). A list inside optimizer_args sweeps as a trainer argument would
    (betas, etas and step_sizes stay pairs). trainer.optimizer.scheduler.args
    keep the framework's own names in the trainer.

Models under framework: keras
    Layer lists map to keras.layers: Linear -> Dense, Conv1d -> Conv1D
    (channels_first, padding 'same'/'valid'), Pooling max/avg ->
    MaxPooling1D/AveragePooling1D, Pooling stride -> Cropping1D +
    MaxPooling1D(pool_size=1) (footnote 24 p. 186; Keras has no stride-pooling
    layer), Pooling global_avg -> GlobalAveragePooling1D, Flatten, Dropout,
    BatchNorm1d -> BatchNormalization (scale: false -> scale=False; momentum and eps
    in PyTorch's meaning, Keras momentum = 1 - momentum), ZeroPad1d ->
    ZeroPadding1D. Graph models (nodes/outputs, e.g. ASCAD_ResNet50 and
    ASCAD_InceptionV3) become a Keras functional model; Add and Concatenate
    merge branches. The named PyTorch classes (architecture: ResNet50_1D, ...)
    are PyTorch-only. Every layer has a name fixed by its position or node, so
    state_dict keys are the same whenever a model is built.

Model hash
    The resolved block (framework and its initialisers) joins the hash only
    when deep_learning is present, so runs made before it keep their identity.
    optimizer_args is not hashed from deep_learning: the wired values join
    through trainer.optimizer.args, where they were before, so moving them
    left every hash unchanged.
    Without the block a run is pytorch with PyTorch's default initialisation;
    a model-level weight_init then needs the block (its meaning depends on the
    framework).

Author: Shraddha Mishra
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAMEWORKS = ("pytorch", "keras")
DEFAULT_FRAMEWORK = "pytorch"
FRAMEWORK_BLOCKS = {"pytorch": "pytorch", "keras": "keras"}
_INIT_KEYS = ("weight_init", "weight_init_args", "bias_init", "bias_init_args")
OPTIMIZER_ARGS_KEY = "optimizer_args"
# deep_learning.<framework>.grad_clip_norm: the framework's gradient clipping, written in params.yaml under the
# block it belongs to; utils/config_loader.canonical_grad_clip folds the chosen framework's value into
# trainer.grad_clip_norm, which gradient_clipper and build_optimizer read, and takes the key out of the blocks.
# A block read before the loader (a test on the raw file) may still hold it.
GRAD_CLIP_KEY = "grad_clip_norm"
_BLOCK_DEFAULTS = {
    "pytorch": {"weight_init": "default", "bias_init": "default"},
    "keras": {"weight_init": "glorot_uniform", "bias_init": "zeros"},   # keras.layers.Dense / Conv1D defaults
}

# torch.nn.init functions by their own names (without the trailing underscore).
TORCH_INITIALIZERS = {
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform_,
    "normal": nn.init.normal_,
    "trunc_normal": nn.init.trunc_normal_,
    "orthogonal": nn.init.orthogonal_,
    "constant": nn.init.constant_,
    "zeros": nn.init.zeros_,
    "ones": nn.init.ones_,
}

# Layers whose weight/bias the initialisers set (PyTorch framework).
_KERNEL_LAYERS = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    nn.Linear,
)


# ── Keras import ────────────────────────────────────────────────────────────

def keras_module():
    """The keras package on its torch backend; imported on first use."""
    current = os.environ.setdefault("KERAS_BACKEND", "torch")
    if current != "torch":
        raise RuntimeError(
            f"deep_learning.framework: keras trains on Keras' torch backend, but "
            f"KERAS_BACKEND={current!r} is set in the environment. Unset it or set "
            "KERAS_BACKEND=torch."
        )
    import keras

    backend = keras.backend.backend()
    if backend != "torch":
        raise RuntimeError(
            f"keras was imported with the {backend!r} backend before TraceFlow "
            "selected the torch backend; start the run in a fresh process"
        )
    return keras


@contextlib.contextmanager
def _keras_device(device: torch.device | str):
    keras = keras_module()
    with keras.device(str(device)):
        yield


# ── deep_learning block ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class InitSpec:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}


@dataclass(frozen=True)
class FrameworkSpec:
    framework: str
    weight_init: InitSpec
    bias_init: InitSpec
    declared: bool

    @property
    def is_keras(self) -> bool:
        return self.framework == "keras"

    @property
    def block(self) -> str:
        return FRAMEWORK_BLOCKS[self.framework]

    def hash_payload(self) -> dict[str, Any] | None:
        """What joins the model hash: the framework and its initialisers, None without the block."""
        if not self.declared:
            return None
        return {
            "framework": self.framework,
            self.block: {
                "weight_init": self.weight_init.payload(),
                "bias_init": self.bias_init.payload(),
            },
        }


def _scalar(value: Any, field_name: str) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{field_name} must be resolved to one value before the run, got {value!r}")
        return value[0]
    return value


def _init_name(raw: Any, default: str, field_name: str) -> str:
    raw = _scalar(raw, field_name)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    if not isinstance(raw, str):
        raise TypeError(f"{field_name} must be a name, got {raw!r}")
    return raw.strip()


def _init_args(raw: Any, field_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {raw!r}")
    if "seed" in raw:
        raise ValueError(f"{field_name}.seed is not accepted: initialisers draw from the run's seed")
    return {str(k): _scalar(v, f"{field_name}.{k}") for k, v in raw.items()}


def _validate_init(framework: str, spec: InitSpec, field_name: str) -> InitSpec:
    if framework == "pytorch":
        if spec.name == "default":
            if spec.args:
                raise ValueError(f"{field_name}: 'default' takes no arguments")
            return spec
        fn = TORCH_INITIALIZERS.get(spec.name)
        if fn is None:
            raise ValueError(
                f"{field_name}={spec.name!r} is not a torch.nn.init function. Under "
                f"framework: pytorch use one of {['default', *TORCH_INITIALIZERS]} "
                "(PyTorch has no lecun_*; its SELU documentation recommends "
                "kaiming_normal with nonlinearity: linear)"
            )
        allowed = set(inspect.signature(fn).parameters) - {"tensor", "generator"}
        unknown = sorted(set(spec.args) - allowed)
        if unknown:
            raise ValueError(f"{field_name}: torch.nn.init.{spec.name}_ has no argument(s) {unknown}; allowed {sorted(allowed)}")
        return spec
    keras = keras_module()
    try:
        initializer = keras.initializers.get(spec.name)
    except ValueError as exc:
        raise ValueError(f"{field_name}={spec.name!r} is not a keras.initializers name: {exc}") from exc
    allowed = set(inspect.signature(type(initializer).__init__).parameters) - {"self", "seed"}
    unknown = sorted(set(spec.args) - allowed)
    if unknown:
        raise ValueError(f"{field_name}: keras.initializers.{type(initializer).__name__} has no argument(s) {unknown}; allowed {sorted(allowed)}")
    return spec


def resolve_framework(cfg: Mapping[str, Any] | None, *, validate: bool = True) -> FrameworkSpec:
    """The run's framework and its initialisers, from cfg['deep_learning'].

    validate=False skips the check of the initialiser names against the
    library (no keras import); the model hash uses it.
    """
    cfg = cfg if isinstance(cfg, Mapping) else {}
    if "framework" in cfg:
        raise ValueError("Move the top-level 'framework' key to deep_learning.framework")
    block = cfg.get("deep_learning")
    if block is None:
        return FrameworkSpec(DEFAULT_FRAMEWORK, InitSpec("default"), InitSpec("default"), declared=False)
    if not isinstance(block, Mapping):
        raise TypeError("deep_learning must be a mapping")
    unknown = sorted(set(block) - {"framework", *FRAMEWORK_BLOCKS.values()})
    if unknown:
        raise ValueError(f"Unknown deep_learning key(s) {unknown}; allowed: framework, pytorch, keras")
    raw = _scalar(block.get("framework"), "deep_learning.framework")
    framework = DEFAULT_FRAMEWORK if raw is None else str(raw).strip().lower()
    if framework not in FRAMEWORKS:
        raise ValueError(f"deep_learning.framework must be one of {FRAMEWORKS}, got {raw!r}")
    name = FRAMEWORK_BLOCKS[framework]
    settings = block.get(name) or {}
    if not isinstance(settings, Mapping):
        raise TypeError(f"deep_learning.{name} must be a mapping")
    unknown = sorted(set(settings) - {*_INIT_KEYS, OPTIMIZER_ARGS_KEY, GRAD_CLIP_KEY})
    if unknown:
        raise ValueError(f"Unknown deep_learning.{name} key(s) {unknown}; allowed: {[*_INIT_KEYS, OPTIMIZER_ARGS_KEY, GRAD_CLIP_KEY]}")
    defaults = _BLOCK_DEFAULTS[framework]
    weight = InitSpec(
        _init_name(settings.get("weight_init"), defaults["weight_init"], f"deep_learning.{name}.weight_init"),
        _init_args(settings.get("weight_init_args"), f"deep_learning.{name}.weight_init_args"),
    )
    bias = InitSpec(
        _init_name(settings.get("bias_init"), defaults["bias_init"], f"deep_learning.{name}.bias_init"),
        _init_args(settings.get("bias_init_args"), f"deep_learning.{name}.bias_init_args"),
    )
    if validate:
        weight = _validate_init(framework, weight, f"deep_learning.{name}.weight_init")
        bias = _validate_init(framework, bias, f"deep_learning.{name}.bias_init")
    return FrameworkSpec(framework, weight, bias, declared=True)


def optimizer_args_table(cfg: Mapping[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    """deep_learning.<framework>.optimizer_args of the selected framework, keyed by lower-case type.

    None when the config has no deep_learning block or the selected block no
    optimizer_args: trainer.optimizer.args are then used as written.
    """
    cfg = cfg if isinstance(cfg, Mapping) else {}
    block = cfg.get("deep_learning")
    if not isinstance(block, Mapping):
        return None
    raw = _scalar(block.get("framework"), "deep_learning.framework")
    framework = DEFAULT_FRAMEWORK if raw is None else str(raw).strip().lower()
    if framework not in FRAMEWORKS:
        raise ValueError(f"deep_learning.framework must be one of {FRAMEWORKS}, got {raw!r}")
    name = FRAMEWORK_BLOCKS[framework]
    table = (block.get(name) or {}).get(OPTIMIZER_ARGS_KEY)
    if table is None:
        return None
    where = f"deep_learning.{name}.{OPTIMIZER_ARGS_KEY}"
    if not isinstance(table, Mapping):
        raise TypeError(f"{where} must map an optimizer type to its arguments, got {table!r}")
    out: dict[str, dict[str, Any]] = {}
    for kind, args in table.items():
        key = str(kind).strip().lower()
        if key in out:
            raise ValueError(f"{where}: optimizer type {kind!r} is given twice")
        if args is not None and not isinstance(args, Mapping):
            raise TypeError(f"{where}.{kind} must be a mapping of arguments, got {args!r}")
        out[key] = dict(args or {})
    return out


def wire_optimizer_args(entry: Mapping[str, Any], optimizer_type: str,
                        table: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    """trainer.optimizer entry (without its type) with args from the framework's optimizer_args.

    table None (no optimizer_args): the entry as written. Otherwise
    trainer.optimizer.args is not accepted, except the wired arguments
    themselves (a params_out snapshot records both; a swept argument there
    holds one of the listed values).
    """
    entry = dict(entry)
    if table is None:
        return entry
    wired = dict(table.get(str(optimizer_type).strip().lower(), {}))

    def selected_from(value: Any, option: Any) -> bool:   # a snapshot holds one choice of a sweep list
        return value == option or (isinstance(option, list) and any(value == o for o in option))

    given = entry.get("args")
    if "args" in entry and not (
        isinstance(given, Mapping) and set(given) == set(wired)
        and all(selected_from(given[k], wired[k]) for k in wired)
    ):
        raise ValueError(
            f"trainer.optimizer ({optimizer_type}) has args {entry['args']!r}; optimizer arguments are "
            f"set in deep_learning.<framework>.{OPTIMIZER_ARGS_KEY}.{optimizer_type} (here {wired!r}). "
            "Move them there and remove trainer.optimizer.args."
        )
    entry["args"] = wired
    return entry


# Model-level weight_init values of configs written before deep_learning
# existed (every run snapshot made before 2026-10-10), mapped to the torch.nn.init
# functions with exactly the same definition, so those runs rebuild with their
# recorded initialisation and keep their hash. The Keras truncated-normal modes
# (glorot_normal, he_normal, lecun_normal, lecun_normal_keras) and random_normal
# have no torch.nn.init equivalent and are rejected.
LEGACY_WEIGHT_INIT = {
    "kaiming_uniform": (InitSpec("default"), InitSpec("default")),          # "PyTorch's default, untouched"
    "glorot_uniform": (InitSpec("xavier_uniform", {"gain": 1.0}), InitSpec("zeros")),
    "he_uniform": (InitSpec("kaiming_uniform", {"nonlinearity": "relu"}), InitSpec("zeros")),
    "lecun_uniform": (InitSpec("kaiming_uniform", {"nonlinearity": "linear"}), InitSpec("zeros")),
}


def model_init_specs(spec: FrameworkSpec, arch_args: Mapping[str, Any], where: str) -> tuple[InitSpec, InitSpec]:
    """The weight and bias initialisers of one model: its own override, else the framework block's."""
    if not spec.declared and arch_args.get("bias_init") in (None, ""):
        legacy = arch_args.get("weight_init")
        if isinstance(legacy, str) and legacy.strip().lower() in LEGACY_WEIGHT_INIT:
            return LEGACY_WEIGHT_INIT[legacy.strip().lower()]
    out = []
    for key, fallback in (("weight_init", spec.weight_init), ("bias_init", spec.bias_init)):
        raw = arch_args.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            out.append(fallback)
            continue
        if not spec.declared:
            raise ValueError(
                f"{where}: architecture_args.{key}={raw!r} needs a deep_learning block (its "
                "meaning depends on deep_learning.framework). Without the block only the "
                f"pre-2026-10-10 names {sorted(LEGACY_WEIGHT_INIT)} are read (as the torch.nn.init "
                "functions of the same definition); add deep_learning: {framework: pytorch}."
            )
        if isinstance(raw, Mapping) and set(raw) <= set(FRAMEWORK_BLOCKS.values()):
            raw = raw.get(spec.block)
            if raw is None:
                out.append(fallback)
                continue
        field_name = f"{where} architecture_args.{key}"
        if isinstance(raw, Mapping):
            unknown = sorted(set(raw) - {"name", "args"})
            if unknown:
                raise ValueError(f"{field_name}: unknown key(s) {unknown}; use name and args")
            init = InitSpec(_init_name(raw.get("name"), "", field_name), _init_args(raw.get("args"), f"{field_name}.args"))
        else:
            init = InitSpec(_init_name(raw, "", field_name))
        if not init.name:
            raise ValueError(f"{field_name} names no initialiser")
        out.append(_validate_init(spec.framework, init, field_name))
    return out[0], out[1]


# A layer's own initialiser (owner 2026-10-29): weight_init / bias_init in one Conv/Linear layer's params, in the forms
# of architecture_args', for that layer over the model's. Keras gives a layer its code names no kernel_initializer
# glorot_uniform: Zaid's and Wouters' output layers (their saved models), whose other layers are he_uniform.
LAYER_INIT_KEYS = ("weight_init", "bias_init")
LAYER_INIT_ATTR = "traceflow_init"           # a torch layer's (weight, bias) override, read by apply_torch_init


def layer_init_overrides(spec: FrameworkSpec, params: dict[str, Any], where: str) -> tuple[InitSpec | None, InitSpec | None]:
    """A layer's own (weight, bias) initialisers, popped from its params; None where it sets none (the model's apply)."""
    own = {k: params.pop(k) for k in LAYER_INIT_KEYS if k in params}
    own = {k: v for k, v in own.items() if not (v is None or (isinstance(v, str) and not v.strip()))}
    if not own:
        return None, None
    if not spec.declared:
        raise ValueError(f"{where}: a layer's {', '.join(sorted(own))} needs a deep_learning block (its meaning "
                         "depends on deep_learning.framework); add deep_learning: {framework: pytorch}")
    weight, bias = model_init_specs(spec, own, where)
    return (weight if "weight_init" in own else None), (bias if "bias_init" in own else None)


# ── PyTorch initialisation ──────────────────────────────────────────────────

def has_lazy_parameters(model: nn.Module) -> bool:
    for module in model.modules():
        check = getattr(module, "has_uninitialized_params", None)
        if callable(check) and check():
            return True
    return False


def materialize_lazy_modules(model: nn.Module, example_input: torch.Tensor) -> None:
    """One side-effect-free forward pass so lazy layers acquire shapes."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example_input)
    finally:
        model.train(was_training)
    if has_lazy_parameters(model):
        raise RuntimeError(f"Lazy parameters remain after the example pass with input {tuple(example_input.shape)}")


def apply_torch_init(model: nn.Module, weight: InitSpec, bias: InitSpec, *, example_input: torch.Tensor | None) -> int:
    """Apply torch.nn.init functions to every Conv/Linear weight and bias (a layer's own, LAYER_INIT_ATTR, over the
    model's); returns the layer count."""
    own = {id(m): getattr(m, LAYER_INIT_ATTR) for m in model.modules() if getattr(m, LAYER_INIT_ATTR, None)}
    if weight.name == "default" and bias.name == "default" and not own:
        return 0
    if has_lazy_parameters(model):
        if example_input is None:
            raise ValueError("Lazy layers need an example input before initialisation")
        materialize_lazy_modules(model, example_input)
    touched = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, _KERNEL_LAYERS):
                continue
            own_weight, own_bias = own.get(id(module), (None, None))
            w, b = own_weight or weight, own_bias or bias
            if w.name != "default":
                TORCH_INITIALIZERS[w.name](module.weight, **w.args)
            if b.name != "default" and module.bias is not None:
                TORCH_INITIALIZERS[b.name](module.bias, **b.args)
            touched += 1
    if touched == 0:
        raise ValueError(f"weight_init={weight.name!r} found no Conv/Linear layer to initialise")
    return touched


# ── Keras model ─────────────────────────────────────────────────────────────

def _torch_seed() -> int:
    """A seed for one Keras random object, drawn from the run's torch generator."""
    return int(torch.randint(0, 2**31 - 1, (), dtype=torch.int64).item())


def _keras_initializer(init: InitSpec):
    keras = keras_module()
    cls = type(keras.initializers.get(init.name))
    if "seed" in inspect.signature(cls.__init__).parameters:
        return cls(**init.args, seed=_torch_seed())
    return cls(**init.args)


def _keras_activation(raw: Any, where: str, layer_name: str):
    """keras.layers.Activation for a configured activation (a name, or {name, args})."""
    keras = keras_module()
    if isinstance(raw, str):
        name, args = raw.strip().lower(), {}
    elif isinstance(raw, Mapping):
        name, args = str(raw.get("name") or "").strip().lower(), dict(raw.get("args") or {})
    else:
        raise ValueError(f"{where}: invalid activation {raw!r}")
    if name in {"", "none", "identity"}:
        name = "linear"
    try:
        fn = keras.activations.get(name)
    except ValueError as exc:
        raise ValueError(f"{where}: Keras has no activation {name!r} (deep_learning.framework: keras): {exc}") from exc
    if args:
        allowed = set(inspect.signature(fn).parameters) - {"x"}
        unknown = sorted(set(args) - allowed)
        if unknown:
            raise ValueError(f"{where}: keras.activations.{name} has no argument(s) {unknown}; allowed {sorted(allowed)}")
        fn = functools.partial(fn, **args)
    return keras.layers.Activation(fn, name=layer_name)


def _only(params: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"{where}: argument(s) {unknown} are not supported under deep_learning.framework: keras; allowed {sorted(allowed)}")


def _keras_padding(padding: Any, kernel_size: int, stride: int, dilation: int, where: str) -> str:
    if isinstance(padding, str) and padding in {"same", "valid"}:
        return padding
    if isinstance(padding, (list, tuple)) and len(padding) == 1:
        padding = padding[0]
    if padding == 0:
        return "valid"
    if stride == 1 and kernel_size % 2 == 1 and padding == dilation * (kernel_size - 1) // 2:
        return "same"
    raise ValueError(
        f"{where}: padding={padding!r} has no Keras Conv1D equivalent (Keras pads 'valid' or "
        "'same'); use padding 0, 'same', or (kernel_size - 1) / 2 with stride 1"
    )


# The layers _keras_layer maps under deep_learning.framework: keras, and the Pooling types it builds (the
# panel offers these alone then; tests/unit/test_framework_switch.py builds each one).
KERAS_LAYERS = ("Linear", "Conv1d", "Pooling", "Flatten", "Dropout", "BatchNorm1d", "ZeroPad1d")
KERAS_POOLING = ("max", "avg", "stride", "global_avg", "global_max")


def _keras_layer(name: str, params: dict[str, Any], *, weight: InitSpec, bias: InitSpec, where: str, layer_name: str):
    """The keras.layers object for one neutral layer entry, plus the input width it must receive.

    layer_name is fixed by the layer's position, so the variable paths (and the
    state_dict keys) of a model are the same every time it is built, in any process.
    """
    keras = keras_module()
    L = keras.layers
    named = {"name": layer_name}
    if name == "Linear":
        _only(params, {"in_features", "out_features", "bias"}, where)
        return L.Dense(
            int(params["out_features"]),
            use_bias=bool(params.get("bias", True)),
            kernel_initializer=_keras_initializer(weight),
            bias_initializer=_keras_initializer(bias),
            **named,
        ), params.get("in_features")
    if name == "Conv1d":
        _only(params, {"in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"}, where)
        kernel, stride, dilation = (int(_scalar(params.get(k, d), k)) for k, d in (("kernel_size", 1), ("stride", 1), ("dilation", 1)))
        return L.Conv1D(
            int(params["out_channels"]),
            kernel,
            strides=stride,
            padding=_keras_padding(params.get("padding", 0), kernel, stride, dilation, where),
            dilation_rate=dilation,
            groups=int(params.get("groups", 1)),
            use_bias=bool(params.get("bias", True)),
            data_format="channels_first",
            kernel_initializer=_keras_initializer(weight),
            bias_initializer=_keras_initializer(bias),
            **named,
        ), params.get("in_channels")
    if name == "Pooling":
        _only(params, {"dim", "type", "kernel_size", "stride", "padding", "count_include_pad"}, where)
        if int(params.get("dim", 1)) != 1:
            raise ValueError(f"{where}: Keras pooling is supported for dim 1")
        kind = str(params.get("type", "max")).lower()
        if kind in ("global_avg", "global_max"):
            cls = L.GlobalAveragePooling1D if kind == "global_avg" else L.GlobalMaxPooling1D
            return cls(data_format="channels_first", **named), None
        size = int(params["kernel_size"])
        stride = int(params.get("stride") or size)
        padding = params.get("padding", 0)
        if kind == "stride":
            if padding:
                raise ValueError(f"{where}: stride pooling takes no padding")
            # Footnote 24 p. 186: the first value of each window, on the window grid of
            # max/avg pooling (whole windows only, 175 -> 87). Keras has no stride-pooling
            # layer; this is Keras layers only: crop the last kernel_size - 1 samples
            # (Cropping1D works on the steps axis, hence the Permutes), then
            # MaxPooling1D(pool_size=1, strides=stride) keeps positions 0, s, 2s, ...
            return keras.Sequential([
                L.Permute((2, 1)),
                L.Cropping1D((0, size - 1)),
                L.Permute((2, 1)),
                L.MaxPooling1D(pool_size=1, strides=stride, padding="valid", data_format="channels_first"),
            ], name=layer_name), None
        cls = {"max": L.MaxPooling1D, "avg": L.AveragePooling1D}.get(kind)
        if cls is None:
            raise ValueError(f"{where}: Pooling type {kind!r} has no Keras layer; Keras offers {', '.join(KERAS_POOLING)}")
        if padding in (0, None, "valid"):
            keras_padding = "valid"
        elif (stride == 1 and size % 2 == 1 and padding == (size - 1) // 2
              and (kind == "max" or params.get("count_include_pad") is False)):
            # Keras 'same' = symmetric padding at stride 1; the average leaves the
            # padding out of the mean (PyTorch count_include_pad: false).
            keras_padding = "same"
        else:
            raise ValueError(
                f"{where}: pooling padding={padding!r} has no Keras equivalent; Keras pads 'valid' or "
                "'same' (stride 1, padding (kernel_size - 1) / 2, count_include_pad: false for avg)"
            )
        return cls(pool_size=size, strides=stride, padding=keras_padding, data_format="channels_first", **named), None
    if name == "Flatten":
        _only(params, set(), where)
        return L.Flatten(**named), None
    if name == "Dropout":
        _only(params, {"p"}, where)
        return L.Dropout(float(params.get("p", 0.5)), seed=_torch_seed(), **named), None
    if name == "BatchNorm1d":
        # One meaning in both frameworks, torch.nn.BatchNorm1d's: momentum is the weight of
        # the new batch (running = (1 - m) * running + m * batch) and eps the epsilon.
        # Keras's momentum weighs the old value (moving = m * moving + (1 - m) * batch), so
        # it gets 1 - m. Unset, each framework keeps its own default (Keras 0.99 = 0.01 here).
        _only(params, {"num_features", "momentum", "eps", "scale"}, where)
        kwargs = {}
        if "momentum" in params:
            kwargs["momentum"] = 1.0 - float(params["momentum"])
        if "eps" in params:
            kwargs["epsilon"] = float(params["eps"])
        return L.BatchNormalization(axis=1, scale=bool(params.get("scale", True)), **kwargs, **named), params.get("num_features")
    if name == "ZeroPad1d":
        _only(params, {"padding"}, where)
        padding = params["padding"]
        padding = tuple(int(v) for v in padding) if isinstance(padding, (list, tuple)) else int(padding)
        return L.ZeroPadding1D(padding, data_format="channels_first", **named), None
    raise ValueError(
        f"{where}: layer {name!r} has no Keras mapping under deep_learning.framework: keras; "
        f"supported: {', '.join(KERAS_LAYERS)} (Pooling: {', '.join(KERAS_POOLING)}), and in graph models Add and Concatenate"
    )


class KerasModel(nn.Module):
    """A Keras 3 model (torch backend) inside the PyTorch training system.

    forward() calls the Keras model with training=self.training, under the
    input's device, so model.train()/model.eval() and model.to() work as for
    any torch module. parameters() and state_dict() are the Keras variables.
    """

    framework = "keras"

    def __init__(self, keras_model):
        super().__init__()
        self.keras_model = keras_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _keras_device(x.device):
            return self.keras_model(x, training=self.training)

    @property
    def trainable_variables(self):
        return self.keras_model.trainable_variables


def _keras_layer_init(spec: FrameworkSpec | None, name: str, params: dict, weight: InitSpec, bias: InitSpec,
                      where: str) -> tuple[InitSpec, InitSpec]:
    """One Keras layer's (weight, bias) initialisers: its own weight_init / bias_init over the model's."""
    if not any(k in params for k in LAYER_INIT_KEYS):
        return weight, bias
    if spec is None:
        raise ValueError(f"{where}: a layer's weight_init / bias_init needs the run's deep_learning block")
    own_weight, own_bias = layer_init_overrides(spec, params, where)
    if (own_weight or own_bias) and name not in ("Linear", "Conv1d"):
        raise ValueError(f"{where}: weight_init / bias_init are read on Linear and Conv1d layers only")
    return own_weight or weight, own_bias or bias


def build_keras_model(layer_defs: list, *, input_shape: tuple[int, ...], weight: InitSpec, bias: InitSpec,
                      resolve_out_features, where: str, spec: FrameworkSpec | None = None) -> KerasModel:
    """keras.Sequential from the neutral layer list (Linear, Conv1d, ... with activation)."""
    keras = keras_module()
    # Keras numbers what TraceFlow does not name (the SeedGenerator of Dropout:
    # seed_generator, seed_generator_1, ...) by a process-wide counter. The CV
    # training process builds 10 folds and the refit; the test process builds one
    # model. Reset the counters, so each model's state_dict keys are the same
    # wherever it is built and the refit's weights load into the test model.
    keras.backend.clear_session()
    layers: list[Any] = [keras.Input(shape=input_shape, name="trace")]
    expected: list[tuple[Any, int, str]] = []
    for idx, layer_def in enumerate(layer_defs):
        if not isinstance(layer_def, Mapping) or len(layer_def) != 1:
            raise ValueError(f"{where}: layer at index {idx} must be a single-key dictionary")
        name, raw = next(iter(layer_def.items()))
        params = dict(raw or {})
        activation = params.pop("activation", None)
        if name == "Linear" and "out_features" in params:
            params["out_features"] = resolve_out_features(params["out_features"], idx)
        w, b = _keras_layer_init(spec, name, params, weight, bias, f"{where} layer {idx} ({name})")
        layer, width = _keras_layer(
            name, params, weight=w, bias=b, where=f"{where} layer {idx} ({name})",
            layer_name=f"layer{idx}_{name.lower()}",
        )
        layers.append(layer)
        if width is not None:
            expected.append((layer, int(width), f"layer {idx} ({name})"))
        if activation is not None:
            layers.append(_keras_activation(activation, f"{where} layer {idx}", f"layer{idx}_activation"))
    model = keras.Sequential(layers, name="traceflow")
    _check_widths(expected, where)
    return KerasModel(model)


def _check_widths(expected: list, where: str) -> None:
    """The neutral in_features / in_channels / num_features must match what Keras inferred."""
    for layer, width, label in expected:
        incoming = layer.input.shape
        actual = incoming[1] if label.endswith(("(Conv1d)", "(BatchNorm1d)")) else incoming[-1]
        if int(actual) != width:
            raise ValueError(f"{where} {label}: configured input width {width}, but the previous layer gives {int(actual)}")


def build_keras_graph(nodes: list, outputs: list, *, input_shape: tuple[int, ...], weight: InitSpec,
                      bias: InitSpec, resolve_out_features, where: str, spec: FrameworkSpec | None = None) -> KerasModel:
    """keras.Model (functional) from a neutral graph: nodes {name, type, inputs, params}, outputs.

    Each node is the layer _keras_layer maps (named after the node, so state_dict
    keys are fixed); Add and Concatenate (channel axis) take several inputs. A
    node's params.activation adds keras.layers.Activation after it.
    """
    keras = keras_module()
    keras.backend.clear_session()          # fixed state_dict keys wherever it is built (build_keras_model)
    tensors: dict[str, Any] = {"input": keras.Input(shape=input_shape, name="trace")}
    expected: list[tuple[Any, int, str]] = []
    for idx, node in enumerate(nodes):
        name = str(node.get("name") or "").strip()
        kind = str(node.get("type") or "").strip()
        inputs = node.get("inputs", ["input"])
        inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        missing = [i for i in inputs if i not in tensors]
        if not name or not kind or missing or name in tensors:
            raise ValueError(f"{where}: graph node {idx} ({name!r}, {kind!r}) is malformed, a duplicate or has unknown inputs {missing}")
        params = dict(node.get("params") or {})
        activation = params.pop("activation", None)
        label = f"{where} graph node {name!r} ({kind})"
        if kind in ("Add", "Concatenate"):
            _only(params, set(), label)
            if len(inputs) < 2:
                raise ValueError(f"{label}: needs at least two inputs")
            layer = keras.layers.Add(name=name) if kind == "Add" else keras.layers.Concatenate(axis=1, name=name)
            out = layer([tensors[i] for i in inputs])
        else:
            if len(inputs) != 1:
                raise ValueError(f"{label}: takes one input, got {inputs}")
            if kind == "Linear" and "out_features" in params:
                params["out_features"] = resolve_out_features(params["out_features"], f"graph:{name}")
            w, b = _keras_layer_init(spec, kind, params, weight, bias, label)
            layer, width = _keras_layer(kind, params, weight=w, bias=b, where=label, layer_name=name)
            out = layer(tensors[inputs[0]])
            if width is not None:
                expected.append((layer, int(width), f"graph node {name!r} ({kind})"))
        if activation is not None:
            out = _keras_activation(activation, label, f"{name}_activation")(out)
        tensors[name] = out
    unknown = [o for o in outputs if o not in tensors]
    if unknown:
        raise ValueError(f"{where}: graph outputs {unknown} are not nodes")
    result = [tensors[o] for o in outputs]
    model = keras.Model(tensors["input"], result[0] if len(result) == 1 else result, name="traceflow")
    _check_widths(expected, where)
    return KerasModel(model)


# ── Loss ────────────────────────────────────────────────────────────────────

class KerasCrossEntropy(nn.Module):
    """keras.losses.CategoricalCrossentropy on integer targets; the network outputs logits.

    from_logits=True (default): the cross-entropy from the logits (log_softmax), no clipping.
    from_logits=False: keras.ops.softmax of the logits, then Keras's cross-entropy of those
    probabilities, which divides by their sum and clips them to [epsilon, 1 - epsilon]
    (keras.config.epsilon(), 1e-7) before the logarithm. That is Keras 2.1.1's
    categorical_crossentropy on a softmax output layer (the paper's models, Sect. 2.4.3
    p. 168, softmax s in Eqs. 9-10): a trace whose true class falls
    below epsilon adds a constant and no gradient. On the torch backend Keras 3 takes
    the probabilities as given (only its TensorFlow backend swaps in the logits).
    """

    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0, from_logits: bool = True):
        super().__init__()
        keras = keras_module()
        self.reduction = reduction
        self.from_logits = bool(from_logits)
        self.fn = keras.losses.CategoricalCrossentropy(
            from_logits=self.from_logits,
            label_smoothing=float(label_smoothing),
            reduction={"mean": "sum_over_batch_size", "sum": "sum"}[reduction],
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        keras = keras_module()
        with _keras_device(logits.device):
            output = logits if self.from_logits else keras.ops.softmax(logits, axis=-1)
            return self.fn(F.one_hot(targets.long(), logits.shape[1]).to(logits.dtype), output)


def keras_cross_entropy_per_trace(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """keras.losses.categorical_crossentropy(one_hot, logits, from_logits=True): one value per trace."""
    keras = keras_module()
    with _keras_device(logits.device):
        return keras.losses.categorical_crossentropy(
            F.one_hot(targets.long(), logits.shape[1]).to(logits.dtype), logits, from_logits=True,
        )


def keras_cross_entropy_from_probabilities(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """keras.losses.categorical_crossentropy of probabilities (from_logits=False), per trace."""
    keras = keras_module()
    with _keras_device(probabilities.device):
        return keras.losses.categorical_crossentropy(
            F.one_hot(targets.long(), probabilities.shape[1]).to(probabilities.dtype), probabilities, from_logits=False,
        )


# trainer.loss.type under framework: keras (the panel offers only these then); NLLLoss is PyTorch's alone
KERAS_LOSSES = ("CrossEntropyLoss",)


def build_loss(cfg: Mapping[str, Any]) -> nn.Module:
    """trainer.loss under the run's framework."""
    from utils.loss_util import build_loss_from_config, resolve_loss_config

    spec = resolve_framework(cfg)
    trainer = cfg["trainer"]
    if not spec.is_keras:
        return build_loss_from_config(trainer)
    name, args = resolve_loss_config(trainer)
    if name not in KERAS_LOSSES:
        raise ValueError(f"trainer.loss.type={name!r} has no Keras counterpart; under framework: keras use {' | '.join(KERAS_LOSSES)}")
    return KerasCrossEntropy(reduction=args.get("reduction", "mean"), label_smoothing=args.get("label_smoothing", 0.0),
                             from_logits=args.get("from_logits", True))


def cross_entropy_per_trace(cfg: Mapping[str, Any], *, logits: torch.Tensor | None = None,
                            probabilities: torch.Tensor | None = None, targets: torch.Tensor) -> torch.Tensor | None:
    """The cross_entropy metric as the run's framework computes it; None means PyTorch's (the caller's own path)."""
    if not resolve_framework(cfg).is_keras:
        return None
    loss_args = ((cfg.get("trainer") or {}).get("loss") or {}).get("args") or {}
    if logits is not None and loss_args.get("from_logits", True):
        return keras_cross_entropy_per_trace(logits, targets)
    if probabilities is None:                  # from_logits: false: as the loss, from the softmax probabilities
        keras = keras_module()
        with _keras_device(logits.device):
            probabilities = keras.ops.softmax(logits, axis=-1)
    return keras_cross_entropy_from_probabilities(probabilities, targets)


# ── Optimizer, scheduler, gradient clipping ─────────────────────────────────

class KerasOptimizer:
    """A keras.optimizers object driven like a torch optimizer by the training loop.

    step() hands the autograd gradients to keras_optimizer.apply(), as Keras'
    own torch-backend train_step does. param_groups[0]['lr'] shows the rate of
    the last step for logging.
    """

    def __init__(self, keras_optimizer, variables):
        self.keras_optimizer = keras_optimizer
        self.variables = list(variables)
        self.param_groups = [{"params": [v.value for v in self.variables], "lr": float(self._lr())}]
        self.state: dict = {}

    def _lr(self):
        return float(keras_module().ops.convert_to_numpy(self.keras_optimizer.learning_rate))

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.param_groups[0]["params"]:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Keras optimizers take no closure")
        params = self.param_groups[0]["params"]
        self.param_groups[0]["lr"] = self._lr()
        device = params[0].device if params else "cpu"
        with torch.no_grad(), _keras_device(device):
            self.keras_optimizer.apply([p.grad for p in params], self.variables)

    def state_dict(self) -> dict:
        return {"keras_optimizer": self.keras_optimizer.get_config(), "iterations": int(self.keras_optimizer.iterations.numpy())}


def keras_schedules() -> list[str]:
    """The trainer.optimizer.scheduler.type values Keras takes: every keras.optimizers.schedules class (the panel's menu)."""
    keras = keras_module()
    base = keras.optimizers.schedules.LearningRateSchedule
    return sorted(n for n, c in vars(keras.optimizers.schedules).items() if inspect.isclass(c) and issubclass(c, base) and c is not base)


def _keras_schedule(scheduler: Mapping[str, Any] | None, learning_rate: float):
    if scheduler is None:
        return learning_rate
    keras = keras_module()
    name = scheduler["type"]
    cls = getattr(keras.optimizers.schedules, name, None)
    if name not in keras_schedules() and inspect.isclass(getattr(torch.optim.lr_scheduler, name, None)):
        raise ValueError(
            f"trainer.optimizer.scheduler.type={name!r} is not a keras.optimizers.schedules class: it is a "
            "torch.optim.lr_scheduler class, available under deep_learning.framework: pytorch "
            f"(keras.optimizers.schedules: {', '.join(keras_schedules())})"
        )
    if name not in keras_schedules() or cls is None:
        raise ValueError(
            f"trainer.optimizer.scheduler.type={name!r} is not a keras.optimizers.schedules class "
            "(ExponentialDecay, InverseTimeDecay, PiecewiseConstantDecay, PolynomialDecay, "
            "CosineDecay, CosineDecayRestarts). Keras schedules advance per optimizer step."
        )
    args = dict(scheduler.get("args") or {})
    if "initial_learning_rate" in inspect.signature(cls.__init__).parameters:
        if "initial_learning_rate" in args:
            raise ValueError("trainer.optimizer.scheduler.args.initial_learning_rate is trainer.optimizer.learning_rate; remove it")
        args["initial_learning_rate"] = learning_rate
    return cls(**args)


_KERAS_NOT_OPTIMIZERS = {"Optimizer", "LossScaleOptimizer"}     # the base class and a wrapper, not optimizers


def keras_activations(names) -> list[str]:
    """Of the activation names core.model builds, the ones keras.activations.get accepts (the panel's menus list
    the others disabled under deep_learning.framework: keras, saying they are PyTorch's)."""
    keras = keras_module()
    out = []
    for name in names:
        try:
            keras.activations.get("linear" if str(name).strip().lower() in {"", "none", "identity"} else str(name))   # as _keras_activation maps them
            out.append(str(name))
        except (ValueError, AttributeError, KeyError, TypeError):
            continue
    return out


def keras_optimizers() -> list[str]:
    """The trainer.optimizer.type values Keras takes: every keras.optimizers class (the panel's menu)."""
    keras = keras_module()
    return sorted(n for n, c in vars(keras.optimizers).items()
                  if inspect.isclass(c) and issubclass(c, keras.optimizers.Optimizer) and n not in _KERAS_NOT_OPTIMIZERS)


def resolve_keras_optimizer(trainer_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate trainer.optimizer for Keras; return the resolved settings."""
    keras = keras_module()
    optimizer = trainer_cfg.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise TypeError("trainer.optimizer must be a mapping")
    unknown = sorted(set(optimizer) - {"type", "learning_rate", "weight_decay", "args", "scheduler"})
    if unknown:
        raise ValueError(f"Unknown trainer.optimizer fields {unknown}")
    name = str(_scalar(optimizer.get("type"), "trainer.optimizer.type")).strip()
    cls = getattr(keras.optimizers, name, None)
    if not (inspect.isclass(cls) and issubclass(cls, keras.optimizers.Optimizer)) or name in _KERAS_NOT_OPTIMIZERS:
        raise ValueError(f"trainer.optimizer.type={name!r} is not a keras.optimizers class")
    learning_rate = float(_scalar(optimizer.get("learning_rate"), "trainer.optimizer.learning_rate"))
    weight_decay = float(_scalar(optimizer.get("weight_decay", 0.0) or 0.0, "trainer.optimizer.weight_decay"))
    if not (math.isfinite(learning_rate) and learning_rate > 0):
        raise ValueError("trainer.optimizer.learning_rate must be > 0")
    if not (math.isfinite(weight_decay) and weight_decay >= 0):
        raise ValueError("trainer.optimizer.weight_decay must be >= 0")
    args = {str(k): _scalar(v, f"trainer.optimizer.args.{k}") for k, v in (optimizer.get("args") or {}).items()}
    allowed = set(inspect.signature(cls.__init__).parameters) - {
        "self", "learning_rate", "weight_decay", "name", "kwargs", "clipnorm", "clipvalue",
        "global_clipnorm", "use_ema", "ema_momentum", "ema_overwrite_frequency",
        "loss_scale_factor", "gradient_accumulation_steps",
    }
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ValueError(
            f"keras.optimizers.{name} has no argument(s) {unknown}; allowed {sorted(allowed)} "
            "(Keras names: rho is PyTorch's alpha, epsilon PyTorch's eps, beta_1/beta_2 PyTorch's betas)"
        )
    scheduler = optimizer.get("scheduler")
    if isinstance(scheduler, Mapping):
        kind = _scalar(scheduler.get("type", "none"), "trainer.optimizer.scheduler.type")
        if kind is None or str(kind).strip().lower() in {"", "none", "null", "off"}:
            scheduler = None
        else:
            if scheduler.get("interval") not in (None, "step"):
                raise ValueError("Keras schedules advance per optimizer step; remove scheduler.interval or set it to 'step'")
            scheduler = {"type": str(kind).strip(), "args": dict(scheduler.get("args") or {})}
    elif scheduler is not None:
        raise TypeError("trainer.optimizer.scheduler must be a mapping or null")
    clip = trainer_cfg.get("grad_clip_norm")
    return {
        "type": name, "class": cls, "learning_rate": learning_rate, "weight_decay": weight_decay,
        "args": args, "scheduler": scheduler, "global_clipnorm": None if clip is None else float(clip),
    }


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module, *, allow_closure_optimizer: bool = False):
    """(optimizer, scheduler_or_None, metadata) under the run's framework."""
    from utils.optimizer_util import build_optimizer_from_config

    spec = resolve_framework(cfg)
    if not spec.is_keras:
        return build_optimizer_from_config(cfg["trainer"], model.parameters(), allow_closure_optimizer=allow_closure_optimizer)
    if getattr(model, "framework", None) != "keras":
        raise TypeError("deep_learning.framework: keras needs the Keras model built by core.model.build_model")
    resolved = resolve_keras_optimizer(cfg["trainer"])
    params = [v.value for v in model.trainable_variables]
    device = params[0].device if params else "cpu"
    with _keras_device(device):
        keras_optimizer = resolved["class"](
            learning_rate=_keras_schedule(resolved["scheduler"], resolved["learning_rate"]),
            weight_decay=resolved["weight_decay"] or None,
            global_clipnorm=resolved["global_clipnorm"],
            **resolved["args"],
        )
    metadata = {
        "framework": "keras",
        "type": resolved["type"],
        "learning_rate": resolved["learning_rate"],
        "weight_decay": resolved["weight_decay"],
        "requires_closure": False,
        "args": dict(resolved["args"]),
        "global_clipnorm": resolved["global_clipnorm"],
        "scheduler": None if resolved["scheduler"] is None else {**resolved["scheduler"], "interval": "step"},
    }
    # The schedule lives inside the Keras optimizer (it reads optimizer.iterations),
    # so the training loop receives no separate scheduler.
    return KerasOptimizer(keras_optimizer, model.trainable_variables), None, metadata


def gradient_clipper(cfg: Mapping[str, Any]):
    """The loop's clipping call: torch.nn.utils.clip_grad_norm_ (pytorch); None for keras,
    whose optimizer clips with its own global_clipnorm."""
    clip = cfg["trainer"].get("grad_clip_norm")
    if clip is None or resolve_framework(cfg).is_keras:
        return None
    clip = float(clip)
    return lambda model: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)


def keras_input_shape(layer_defs: Iterable[Mapping[str, Any]], length: int) -> tuple[int, int]:
    """(channels, length) of one trace: the first Conv1d's in_channels, else 1 (traces are (N, 1, L))."""
    for layer_def in layer_defs:
        if isinstance(layer_def, Mapping) and "type" in layer_def:          # graph node
            name, params = layer_def.get("type"), layer_def.get("params")
        elif isinstance(layer_def, Mapping) and len(layer_def) == 1:        # layer list entry
            name, params = next(iter(layer_def.items()))
        else:
            continue
        if name == "Conv1d":
            return int((params or {}).get("in_channels", 1)), int(length)
    return 1, int(length)


__all__ = [
    "DEFAULT_FRAMEWORK", "FRAMEWORKS", "FRAMEWORK_BLOCKS", "FrameworkSpec", "InitSpec",
    "KERAS_LAYERS", "KERAS_LOSSES", "KERAS_POOLING", "KerasCrossEntropy", "KerasModel", "KerasOptimizer", "TORCH_INITIALIZERS",
    "apply_torch_init", "build_keras_graph", "build_keras_model", "build_loss", "build_optimizer",
    "cross_entropy_per_trace", "gradient_clipper", "has_lazy_parameters",
    "keras_input_shape", "keras_module", "keras_optimizers", "keras_schedules", "materialize_lazy_modules",
    "model_init_specs", "resolve_framework", "resolve_keras_optimizer",
]
