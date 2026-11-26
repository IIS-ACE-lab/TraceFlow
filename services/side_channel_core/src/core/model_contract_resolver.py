# services/side_channel_core/src/core/model_contract_resolver.py
"""
model_contract_resolver.py
==========================

Single source of truth for resolving the model/search contract before training.

Rules
-----
- model_id selects exactly one model block.
- trainer.optimizer.learning_rate is the only official learning-rate key.
- search_space is AutoSearch-only input, never train.py runtime input.
- train.py must receive concrete non-empty architecture_args.layers.
- SCC/manual mode uses layers directly and ignores search_space.
- AutoSearch mode uses search_space to materialize layers and ignores existing 
layers.
- A named architecture (e.g. ResNet50_1D) has no layers; its num_classes follows
  the leakage model.
- architecture_args.weight_init / bias_init (optional) must name an
  initialiser of deep_learning.framework (core/framework_switch.py).
- The automatic head resolver:

    ID  -> out_features: 256
    HW  -> out_features: 9
    MSB -> out_features: 2

Author: Shraddha Mishra
"""

from __future__ import annotations

import random
import math
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple
from services.side_channel_core.src.sca.leakage_models import (
    get_num_classes,
    resolve_runtime_leakage_model,
)
from services.side_channel_core.src.sca.key_enumeration import (
    KeyEnumerationError,
    assert_enumeration_lifecycle_supported,
)
from services.side_channel_core.utils.sca_sweep import (
    assert_concrete_sca_runtime,
)
Json = Dict[str, Any]


class ModelContractError(ValueError):
    """Raised when the model/search contract is invalid."""


def resolve_model_contract(
    cfg: Json,
    *,
    mode: str,
    model_id: Optional[str] = None,
    sampled: Optional[Mapping[str, Any]] = None,
    trial: Any = None,
) -> Json:
    """
    Resolve cfg into train.py-safe runtime config.

    Parameters
    ----------
    cfg:
        Full merged runtime config.
    mode:
        "auto_search" or "scc".
    model_id:
        Optional explicit model selector.
    sampled:
        Optional sampled params from AutoSearch/Optuna.
    trial:
        Optional Optuna trial-like object.

    Returns
    -------
    dict
        Deep-copied, resolved config safe for hashing and train.py.

    Hard guarantee
    --------------
    Returned cfg has:
      - cfg["model_id"]
      - exactly one selected model active
      - selected model architecture_args.layers is non-empty
      - selected model architecture_args.search_space removed
      - trainer.optimizer.learning_rate normalized
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")

    out = deepcopy(cfg)
    mode = _normalize_mode(mode)

    sampled = dict(sampled or {})

    _apply_sampled_top_level(out, sampled)
    #
    # Runtime model resolution must never operate on an unresolved SCA
    # search specification.
    #
    sca_cfg = out.get("sca")

    if sca_cfg is not None:
        if not isinstance(
            sca_cfg,
            dict,
        ):
            raise ModelContractError(
                "cfg['sca'] must be a dict"
            )

        try:
            assert_concrete_sca_runtime(
                sca_cfg
            )
        except Exception as exc:
            raise ModelContractError(
                "resolve_model_contract() received an unresolved SCA "
                "configuration. SCA sweep/search expansion must occur "
                "before model-contract resolution."
            ) from exc

        # Fail before training, not after the lifecycle has finished.
        try:
            assert_enumeration_lifecycle_supported(
                out
            )
        except KeyEnumerationError as exc:
            raise ModelContractError(
                str(exc)
            ) from exc
    _normalize_trainer_lr(out)
    selected_model_id = _resolve_model_id(out, model_id=model_id, sampled=sampled)
    selected = _select_model(out, selected_model_id)

    if mode == "auto_search":
        _resolve_auto_search_mode(out, selected, selected_model_id, sampled=sampled, trial=trial)
    elif mode == "scc":
        _resolve_scc_mode(out, selected, selected_model_id)
    else:
        raise ModelContractError(f"unsupported resolver mode: {mode!r}")

    _strip_search_spaces(out)
    _validate_runtime_layers(out, selected_model_id)

    out["model_id"] = selected_model_id
    out.setdefault("runtime_contract", {})
    out["runtime_contract"]["mode"] = mode
    out["runtime_contract"]["model_id"] = selected_model_id
    out["runtime_contract"]["resolved"] = True
    out["runtime_contract"]["lr_key"] = "trainer.optimizer.learning_rate"

    return out


def _normalize_mode(mode: str) -> str:
    value = str(mode or "").strip().lower()
    aliases = {
        "manual": "scc",
        "sweep": "scc",
        "grid": "scc",
        "auto": "auto_search",
        "search": "auto_search",
        "optuna": "auto_search",
        "optuna_lite": "auto_search",
    }
    return aliases.get(value, value)


def _apply_sampled_top_level(
    out: Json,
    sampled: Mapping[str, Any],
) -> None:
    """
    Apply generic sampled non-architecture values.

    SCA is deliberately NOT applied here.

    SCA search has conditional semantics and is compiled by sca_sweep.py.
    AutoSearch must inject the selected concrete SCA contract into cfg["sca"]
    before calling resolve_model_contract().

    This resolver therefore receives SCA as runtime configuration, not as a
    generic sampled overlay.

    Important:
    - model_id is handled separately;
    - trainer.optimizer.learning_rate is normalized later;
    - model_search is recipe input only;
    - sampled["sca"] is forbidden.
    """
    if "sca" in sampled:
        raise ModelContractError(
            "sampled['sca'] must not be passed directly to "
            "resolve_model_contract(). SCA must first be compiled into one "
            "concrete conditional runtime contract and injected into cfg['sca']."
        )

    trainer_updates = sampled.get(
        "trainer"
    )

    if isinstance(
        trainer_updates,
        dict,
    ):
        out.setdefault(
            "trainer",
            {},
        )

        if not isinstance(
            out["trainer"],
            dict,
        ):
            raise ModelContractError(
                "cfg['trainer'] must be a dict"
            )

        out["trainer"].update(
            deepcopy(
                trainer_updates
            )
        )

    if "learning_rate" in sampled or "lr" in sampled:
        raise ModelContractError(
            "Flat sampled learning rates are unsupported; use sampled.trainer.optimizer.learning_rate"
        )


def _resolve_model_id(
    cfg: Json,
    *,
    model_id: Optional[str],
    sampled: Mapping[str, Any],
) -> str:
    candidate = (
        model_id
        or sampled.get("model_id")
        or cfg.get("model_id")
    )

    models = cfg.get("model")
    if not isinstance(models, list):
        raise ModelContractError("cfg['model'] must be a list of model blocks")

    available = [
        str(m.get("model_id")).strip()
        for m in models
        if isinstance(m, dict) and str(m.get("model_id", "")).strip()
    ]

    if candidate:
        candidate = str(candidate).strip()
        if candidate not in available:
            raise ModelContractError(
                f"model_id={candidate!r} not found. Available model_id values: {available}"
            )
        return candidate

    if len(available) == 1:
        return available[0]

    raise ModelContractError(
        f"model_id is required because multiple model entries exist: {available}"
    )


def _select_model(cfg: Json, model_id: str) -> Json:
    models = cfg.get("model")
    if not isinstance(models, list):
        raise ModelContractError("cfg['model'] must be a list")

    selected = None
    for model in models:
        if isinstance(model, dict) and str(model.get("model_id", "")).strip() == model_id:
            selected = model
            break

    if selected is None:
        raise ModelContractError(f"selected model_id={model_id!r} not found")

    # Runtime config should carry only selected model to avoid CNN1/MLP1 mixing.
    cfg["model"] = [deepcopy(selected)]
    return cfg["model"][0]


def _resolve_auto_search_mode(
    cfg: Json,
    selected: Json,
    model_id: str,
    *,
    sampled: Mapping[str, Any],
    trial: Any,
) -> None:
    auto_search = cfg.get("auto_search") or {}
    if isinstance(auto_search, dict) and not bool(auto_search.get("enabled", False)):
        raise ModelContractError(
            "mode='auto_search' requested but cfg['auto_search']['enabled'] is not true"
        )

    arch_args = selected.setdefault("architecture_args", {})
    if not isinstance(arch_args, dict):
        raise ModelContractError(f"model_id={model_id}: architecture_args must be a dict")

    search_space = arch_args.get("search_space") or {}
    if not isinstance(search_space, dict) or not bool(search_space.get("enabled", False)):
        raise ModelContractError(
            f"model_id={model_id}: auto_search requires architecture_args.search_space.enabled=true"
        )

    architecture = str(selected.get("architecture", "")).strip().upper()
    clean_space = {k: deepcopy(v) for k, v in search_space.items() if k != "enabled"}

    # If OptunaLite sampled nested model_search, merge that selected block on top.
    sampled_model_search = sampled.get("model_search")
    if isinstance(sampled_model_search, dict):
        selected_sample_space = sampled_model_search.get(model_id)
        if isinstance(selected_sample_space, dict):
            clean_space.update(deepcopy(selected_sample_space))

    if "learning_rate" in clean_space or "lr" in clean_space:
        raise ModelContractError(
            "Move architecture search_space learning_rate/lr choices to trainer.optimizer.learning_rate"
        )

    if architecture == "MLP":
        layers = _materialize_mlp(cfg, clean_space, trial)
    elif architecture == "CNN":
        layers = _materialize_cnn(cfg, clean_space, trial)
    else:
        raise ModelContractError(
            f"model_id={model_id}: search_space materialization is not supported for architecture={architecture!r}"
        )

    if not layers:
        raise ModelContractError(f"model_id={model_id}: auto_search materialized empty layers")

    arch_args["layers"] = layers
    #This makes the final head automatic:
    _resolve_final_classifier_head(cfg, selected=selected, model_id=model_id) #
    
    cfg.setdefault("trial_materialization", {})
    cfg["trial_materialization"] = {
        "model_id": model_id,
        "architecture": architecture,
        "trainer.optimizer.learning_rate": cfg["trainer"]["optimizer"]["learning_rate"],
        "layers": deepcopy(layers),
    }


# Architectures written as a YAML layer list; anything else is a named class
# built from architecture_args (e.g. ResNet50_1D, InceptionV3_1D).
_LAYER_ARCHITECTURES = {"CNN", "MLP", "TRANSFORMER", "HYBRID"}


def is_named_architecture(selected: Json) -> bool:
    """True for a class-built model block (no YAML layer list)."""
    architecture = str(selected.get("architecture") or "").strip().upper()
    return bool(architecture) and architecture not in _LAYER_ARCHITECTURES


def _validate_weight_init(cfg: Json, selected: Json, model_id: str) -> None:
    """
    Check deep_learning and the model's weight_init / bias_init before any run is planned.

    The names must be the active framework's own (torch.nn.init or
    keras.initializers, core/framework_switch.py); build_model() applies the
    same check when it builds the model.
    """
    # framework_switch imports torch; a local import keeps this module importable without it.
    from services.side_channel_core.src.core.framework_switch import model_init_specs, resolve_framework

    arch_args = selected.get("architecture_args") or {}
    try:
        model_init_specs(resolve_framework(cfg), arch_args, f"model_id={model_id}")
    except (TypeError, ValueError) as exc:
        raise ModelContractError(str(exc)) from exc


def _resolve_named_architecture_head(cfg: Json, selected: Json, model_id: str) -> None:
    """
    Give a named architecture the class count of the active leakage model.

    The layer path resolves ``out_features: auto``; a named class takes the
    same decision through its ``num_classes`` constructor argument, so the two
    paths cannot disagree with sca.leakage_model.
    """
    arch_args = selected.setdefault("architecture_args", {})
    leakage_model = resolve_runtime_leakage_model(cfg, mutate=True)
    expected = int(get_num_classes(leakage_model))
    configured = arch_args.get("num_classes", "auto")

    if isinstance(configured, str) and configured.strip().lower() == "auto":
        arch_args["num_classes"] = expected
        return

    if int(configured) != expected:
        raise ModelContractError(
            f"model_id={model_id}: architecture_args.num_classes={configured} "
            f"does not match sca.leakage_model={leakage_model!r} ({expected} classes). "
            "Use num_classes: auto to follow the leakage model."
        )


def _resolve_scc_mode(cfg: Json, selected: Json, model_id: str) -> None:
    arch_args = selected.setdefault("architecture_args", {})
    if not isinstance(arch_args, dict):
        raise ModelContractError(f"model_id={model_id}: architecture_args must be a dict")

    if is_named_architecture(selected):
        _resolve_named_architecture_head(cfg, selected=selected, model_id=model_id)
        return

    if _graph_nodes(arch_args) is not None:
        _resolve_graph_classifier_head(cfg, selected=selected, model_id=model_id)
        return

    layers = arch_args.get("layers")
    if not isinstance(layers, list) or len(layers) == 0:
        raise ModelContractError(
            f"model_id={model_id}: SCC/manual mode requires non-empty architecture_args.layers "
            "(or a graph: architecture_args.nodes and outputs)"
        )

    _resolve_final_classifier_head(cfg, selected=selected, model_id=model_id)


def _graph_nodes(arch_args: Json) -> list | None:
    """The nodes of a graph model (architecture_args.nodes and outputs), else None."""
    nodes, outputs = arch_args.get("nodes"), arch_args.get("outputs")
    if nodes is None and outputs is None:
        return None
    if not isinstance(nodes, list) or not nodes or not isinstance(outputs, list) or not outputs:
        raise ModelContractError("a graph model needs non-empty architecture_args.nodes and outputs lists")
    return nodes


def _graph_head(arch_args: Json, model_id: str) -> dict:
    """The params of the graph's classifier node: the single output, a Linear without activation."""
    nodes = _graph_nodes(arch_args)
    outputs = arch_args["outputs"]
    if len(outputs) != 1:
        raise ModelContractError(f"model_id={model_id}: an SCA graph model has one output, the classifier; got {outputs}")
    by_name = {str(n.get("name")): n for n in nodes if isinstance(n, dict)}
    head = by_name.get(str(outputs[0]))
    if head is None or str(head.get("type")) not in _CLASSIFIER_LAYER_NAMES:
        raise ModelContractError(
            f"model_id={model_id}: the graph output {outputs[0]!r} must be a Linear/LazyLinear node "
            "(the classifier head, out_features: auto)"
        )
    params = head.setdefault("params", {})
    activation = params.get("activation")
    if activation is not None and str(activation).strip().lower() not in {"none", "identity", "linear", ""}:
        raise ModelContractError("Final classifier activation must be identity/none; training requires raw logits")
    return params


def _resolve_graph_classifier_head(cfg: Json, *, selected: Json, model_id: str) -> None:
    """Graph models: the output Linear's out_features follows the leakage model, as for layer lists."""
    params = _graph_head(selected.get("architecture_args") or {}, model_id)
    leakage_model = resolve_runtime_leakage_model(cfg, mutate=True)
    expected_classes = int(get_num_classes(leakage_model))
    current = params.get("out_features")
    if current is None:
        raise ModelContractError(f"model_id={model_id}: the graph's classifier node is missing out_features; use auto")
    if _is_auto_head_value(current) or _is_classifier_sized_value(current):
        params["out_features"] = expected_classes
    else:
        raise ModelContractError(
            f"model_id={model_id}: suspicious classifier out_features={current!r}; expected auto or one of "
            f"{{256, 9, 2}} (leakage_model={leakage_model} requires {expected_classes})"
        )


def _strip_search_spaces(cfg: Json) -> None:
    """
    Remove search_space before train.py and hashing.

    This prevents AutoSearch recipe fields from polluting runtime identity.
    """
    models = cfg.get("model")
    if not isinstance(models, list):
        return

    for model in models:
        if not isinstance(model, dict):
            continue
        arch_args = model.get("architecture_args")
        if isinstance(arch_args, dict):
            arch_args.pop("search_space", None)


def _validate_runtime_layers(cfg: Json, model_id: str) -> None:
    """
    This makes the resolver fail before training, instead of letting a bad model reach the epoch loop.

    Your epoch loop also has assert_head_sanity() as a second safety layer, which checks the actual PyTorch output shape against leakage model. That is good, but this resolver patch catches the problem earlier at YAML/model-contract level.
    """
    selected = _select_model(cfg, model_id)
    arch_args = selected.get("architecture_args") or {}
    layers = arch_args.get("layers")
    _validate_weight_init(cfg, selected, model_id)

    if is_named_architecture(selected):
        # The head of a named architecture is its num_classes argument.
        _resolve_named_architecture_head(cfg, selected=selected, model_id=model_id)
        return

    if _graph_nodes(arch_args) is not None:
        params = _graph_head(arch_args, model_id)
        expected_classes = int(get_num_classes(resolve_runtime_leakage_model(cfg, mutate=True)))
        if _is_auto_head_value(params.get("out_features")) or int(params.get("out_features")) != expected_classes:
            raise ModelContractError(
                f"FATAL: model_id={model_id} graph classifier out_features={params.get('out_features')!r}, "
                f"but the leakage model requires {expected_classes}. Classifier head resolution failed."
            )
        return

    if not isinstance(layers, list) or len(layers) == 0:
        raise ModelContractError(
            f"FATAL: model_id={model_id} resolved to empty layers. "
            "train.py must never receive layers: []"
        )

    # Final safety check: no unresolved auto head reaches build_model().
    leakage_model = resolve_runtime_leakage_model(cfg, mutate=True)
    expected_classes = int(get_num_classes(leakage_model))

    final_idx = None
    final_name = None
    final_args = None

    for idx in range(len(layers) - 1, -1, -1):
        name, args = _layer_name_and_args(layers[idx], model_id=model_id, index=idx)
        if name in _CLASSIFIER_LAYER_NAMES:
            final_idx = idx
            final_name = name
            final_args = args
            break

    if final_idx is None or final_args is None:
        raise ModelContractError(
            f"FATAL: model_id={model_id} has no final Linear/LazyLinear classifier head."
        )

    activation = final_args.get("activation")
    if activation is not None and str(activation).strip().lower() not in {"none", "identity", "linear", ""}:
        raise ModelContractError("Final classifier activation must be identity/none; training requires raw logits")
    for idx in range(final_idx + 1, len(layers)):
        name, _ = _layer_name_and_args(layers[idx], model_id=model_id, index=idx)
        if name != "Identity":
            raise ModelContractError(
                f"Layer {name} follows the classifier head; keep raw logits and configure trainer.evaluation.output_transformation"
            )

    out_features = final_args.get("out_features", None)

    if _is_auto_head_value(out_features):
        raise ModelContractError(
            f"FATAL: model_id={model_id} still has unresolved classifier out_features={out_features!r}. "
            "resolve_model_contract() must convert it before train.py/build_model()."
        )

    try:
        actual = int(out_features)
    except Exception as e:
        raise ModelContractError(
            f"FATAL: model_id={model_id} final {final_name}.out_features must be int after resolution, "
            f"got {out_features!r}"
        ) from e

    if actual != expected_classes:
        raise ModelContractError(
            f"FATAL: model_id={model_id} final {final_name}.out_features={actual}, "
            f"but leakage_model={leakage_model} requires {expected_classes}. "
            "Classifier head resolution failed."
        )


def _normalize_trainer_lr(cfg: Json) -> None:
    trainer = cfg.get("trainer")
    if not isinstance(trainer, dict):
        raise ModelContractError("cfg['trainer'] must be a dict")
    obsolete = {"lr", "learning_rate", "weight_decay", "optimizer_args", "loss_reduction"}.intersection(trainer)
    if obsolete:
        raise ModelContractError(f"Obsolete trainer fields: {sorted(obsolete)}; use nested optimizer/loss settings")
    optimizer = trainer.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ModelContractError("trainer.optimizer must be a resolved mapping")
    value = optimizer.get("learning_rate")
    if isinstance(value, (bool, list, tuple)) or value is None:
        raise ModelContractError("trainer.optimizer.learning_rate must be one scalar selected before resolution")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelContractError("trainer.optimizer.learning_rate must be numeric") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ModelContractError("trainer.optimizer.learning_rate must be finite and positive")
    optimizer["learning_rate"] = rate


def _input_length(cfg: Json) -> int:
    data = cfg.get("data")

    if isinstance(data, dict):
        if "target_range" in data:
            target_range = data.get("target_range")
        else:
            # dict-of-datasets fallback
            first = next(iter(data.values()), {})
            target_range = first.get("target_range") if isinstance(first, dict) else None
    elif isinstance(data, list) and data:
        first = data[0]
        target_range = first.get("target_range") if isinstance(first, dict) else None
    else:
        target_range = None

    if isinstance(target_range, list) and len(target_range) >= 2:
        return int(target_range[1] - target_range[0])

    return 700


def _num_classes(cfg: Json) -> int:
    """
    Resolve the active runtime leakage model and return its classifier size.

    ID  -> 256
    HW  -> 9
    MSB -> 2

    This must be the only class-count source used by model materialization.
    """
    leakage_model = resolve_runtime_leakage_model(cfg, mutate=True)
    return int(get_num_classes(leakage_model))

_CLASSIFIER_LAYER_NAMES = {"Linear", "LazyLinear"}
_VALID_AUTO_HEAD_VALUES = {"auto", "num_classes", "__num_classes__"}


def _is_auto_head_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _VALID_AUTO_HEAD_VALUES


def _is_classifier_sized_value(value: Any) -> bool:
    """
    True for known SCA classifier sizes.

    These are safe to rewrite because they are the output-space sizes for
    supported leakage models, not arbitrary hidden-layer widths.
    """
    try:
        return int(value) in {2, 9, 256}
    except Exception:
        return False


def _layer_name_and_args(layer: Any, *, model_id: str, index: int) -> Tuple[str, Json]:
    """
    Validate and unpack one YAML layer entry:
      - Linear: { ... }
      - LazyLinear: { ... }
    """
    if not isinstance(layer, dict) or len(layer) != 1:
        raise ModelContractError(
            f"model_id={model_id}: layer[{index}] must be a single-key dict, got {layer!r}"
        )

    name, args = next(iter(layer.items()))
    name = str(name).strip()

    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ModelContractError(
            f"model_id={model_id}: layer[{index}] args for {name} must be a dict, got {type(args)}"
        )

    return name, args


def _resolve_final_classifier_head(
    cfg: Json,
    *,
    selected: Json,
    model_id: str,
) -> None:
    """
    Make the final Linear/LazyLinear classifier head automatic.

    Contract:
      - The final classifier layer must be Linear or LazyLinear.
      - Its out_features is forced to get_num_classes(active leakage_model).
      - Supports:
          out_features: auto
          out_features: num_classes
          out_features: __num_classes__
          out_features: 256 / 9 / 2
      - Hard-fails strange final classifier sizes to prevent silent scientific bugs.
    """
    arch_args = selected.get("architecture_args") or {}
    layers = arch_args.get("layers")

    if not isinstance(layers, list) or not layers:
        raise ModelContractError(
            f"model_id={model_id}: cannot resolve classifier head because layers are empty"
        )

    leakage_model = resolve_runtime_leakage_model(cfg, mutate=True)
    expected_classes = int(get_num_classes(leakage_model))

    # Find the last Linear/LazyLinear layer. This is the classifier head.
    final_idx = None
    final_name = None
    final_args = None

    for idx in range(len(layers) - 1, -1, -1):
        name, args = _layer_name_and_args(layers[idx], model_id=model_id, index=idx)
        if name in _CLASSIFIER_LAYER_NAMES:
            final_idx = idx
            final_name = name
            final_args = args
            break

    if final_idx is None or final_name is None or final_args is None:
        raise ModelContractError(
            f"model_id={model_id}: no Linear/LazyLinear classifier head found. "
            "Add a final Linear/LazyLinear layer with out_features: auto."
        )

    current = final_args.get("out_features", None)

    if current is None:
        raise ModelContractError(
            f"model_id={model_id}: final {final_name} layer is missing out_features. "
            "Use out_features: auto."
        )

    if _is_auto_head_value(current):
        final_args["out_features"] = expected_classes
    elif _is_classifier_sized_value(current):
        # Production behavior: make leakage_model authoritative.
        # Example: YAML says 256 but leakage_model=HW => rewrite to 9.
        final_args["out_features"] = expected_classes
    else:
        raise ModelContractError(
            f"model_id={model_id}: suspicious final classifier out_features={current!r}. "
            f"Expected auto or one of known SCA class counts {{256, 9, 2}}. "
            f"Active leakage_model={leakage_model} requires {expected_classes}. "
            "Refusing to guess because this may be a non-standard classifier head."
        )

    # Write back into the original single-key layer dict.
    layers[final_idx] = {final_name: final_args}

    selected.setdefault("architecture_args", {})
    selected["architecture_args"]["layers"] = layers

    cfg.setdefault("runtime_contract", {})
    cfg["runtime_contract"]["leakage_model"] = leakage_model
    cfg["runtime_contract"]["num_classes"] = expected_classes
    cfg["runtime_contract"]["classifier_head"] = {
        "layer_index": int(final_idx),
        "layer_type": final_name,
        "out_features": expected_classes,
    }

def _choice(trial: Any, name: str, values: Any) -> Any:
    # Already-sampled concrete value
    if not isinstance(values, list):
        return deepcopy(values)

    if len(values) == 0:
        raise ModelContractError(f"search_space.{name} must be a non-empty list")

    if trial is None:
        return deepcopy(random.choice(values))

    if hasattr(trial, "suggest_categorical"):
        return trial.suggest_categorical(name, values)

    return deepcopy(random.choice(values))


def _range_choice(trial: Any, name: str, values: Any) -> int:
    # Already-sampled concrete value
    if not isinstance(values, list):
        return int(values)

    if len(values) == 0:
        raise ModelContractError(f"search_space.{name} must be a non-empty list")

    if len(values) == 2 and all(isinstance(x, int) for x in values):
        lo, hi = int(values[0]), int(values[1])
        if trial is None:
            return random.randint(lo, hi)
        if hasattr(trial, "suggest_int"):
            return int(trial.suggest_int(name, lo, hi))

    return int(_choice(trial, name, values))

def _require(space: Json, key: str) -> Any:
    if key not in space:
        raise ModelContractError(f"missing required search_space field: {key}")
    return space[key]


def _materialize_mlp(cfg: Json, space: Json, trial: Any) -> List[Json]:
    input_len = _input_length(cfg)
    classes = _num_classes(cfg)

    n_layers = _range_choice(trial, "mlp_n_dense_layers", _require(space, "n_dense_layers"))
    units = _choice(trial, "mlp_dense_units", _require(space, "dense_units"))
    activation = _choice(trial, "mlp_activation", _require(space, "activation"))

    layers: List[Json] = []

    # Input arrives as [N, 1, 700]. MLP needs [N, 700].
    layers.append({"Flatten": {}})

    in_features = int(input_len)

    for _ in range(int(n_layers)):
        layers.append(
            {
                "Linear": {
                    "in_features": int(in_features),
                    "out_features": int(units),
                    "activation": activation,
                }
            }
        )
        in_features = int(units)

    layers.append(
        {
            "Linear": {
                "in_features": int(in_features),
                "out_features": int(classes),
            }
        }
    )

    return layers


def _materialize_cnn(cfg: Json, space: Json, trial: Any) -> List[Json]:
    classes = _num_classes(cfg)

    n_conv = _range_choice(trial, "cnn_n_conv_layers", _require(space, "n_conv_layers"))
    activation = _choice(trial, "cnn_activation", _require(space, "activation"))

    layers: List[Json] = []
    in_channels = 1

    for i in range(int(n_conv)):
        filters = _choice(trial, f"cnn_conv_{i}_filters", _require(space, "conv_filters"))
        kernel = _choice(trial, f"cnn_conv_{i}_kernel_size", _require(space, "conv_kernel_size"))
        pool_type = _choice(trial, f"cnn_pool_{i}_type", _require(space, "pooling_type"))
        pool_size = _choice(trial, f"cnn_pool_{i}_size", _require(space, "pooling_size"))
        pool_stride = _choice(trial, f"cnn_pool_{i}_stride", _require(space, "pooling_stride"))

        layers.append(
            {
                "Conv1d": {
                    "in_channels": int(in_channels),
                    "out_channels": int(filters),
                    "kernel_size": int(kernel),
                    "stride": 1,
                    "padding": int(kernel) // 2,
                    "activation": activation,
                }
            }
        )

        layers.append(
            {
                "Pooling": {
                    "dim": 1,
                    "type": pool_type,
                    "kernel_size": int(pool_size),
                    "stride": int(pool_stride),
                }
            }
        )

        in_channels = int(filters)

    layers.append({"Flatten": {}})

    n_dense = _range_choice(trial, "cnn_n_dense_layers", _require(space, "n_dense_layers"))
    dense_units = _choice(trial, "cnn_dense_units", _require(space, "dense_units"))

    for i in range(int(n_dense)):
        if i == 0:
            layers.append(
                {
                    "LazyLinear": {
                        "out_features": int(dense_units),
                        "activation": activation,
                    }
                }
            )
        else:
            layers.append(
                {
                    "Linear": {
                        "in_features": int(dense_units),
                        "out_features": int(dense_units),
                        "activation": activation,
                    }
                }
            )

    layers.append(
        {
            "Linear": {
                "in_features": int(dense_units),
                "out_features": int(classes),
            }
        }
    )

    return layers
