#services/side_channel_core/utils/model_preview.py
"""
model_preview.py
================

Resolved model preview/export utilities.

Purpose
-------
The model is created dynamically before training.

In AutoSearch mode:
    architecture_args.search_space -> resolve_model_contract() -> concrete layers

In SCC/manual mode:
    architecture_args.layers -> resolve_model_contract() -> validated concrete layers

This module does NOT materialize the model.
This module does NOT sample the search space.
This module does NOT build the PyTorch module.

It only records and displays the already-resolved train.py-safe runtime model
after resolve_model_contract(...) and before build_model(...).

Output location
---------------
Given params.yaml:

    paths:
      model_out: "artifacts/{{env}}/generated_models/{{model_name}}/model"

This module saves:

    artifacts/<env>/generated_models/<model_name>/model/resolved_model.yaml
    artifacts/<env>/generated_models/<model_name>/model/resolved_model.json

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from services.side_channel_core.src.core.model_contract_resolver import is_named_architecture
from services.side_channel_core.utils.seal import seal_text


Json = Dict[str, Any]


class ModelPreviewError(RuntimeError):
    """Raised when resolved model preview/export fails."""


# =============================================================================
# Path handling
# =============================================================================

def _render_template_path(
    path_template: str,
    *,
    cfg: Mapping[str, Any],
    env: str,
    model_name: str,
) -> Path:
    """
    Render common PROJECT FREEDOM placeholders.

    Supported:
        {{env}}, {env}
        {{model_name}}, {model_name}
        {{model_hash}}, {model_hash}
    """
    if not path_template or not str(path_template).strip():
        raise ModelPreviewError("cfg['paths']['data'] is empty or missing.")

    rendered = str(path_template)

    replacements = {
        "{{env}}": str(env),
        "{env}": str(env),
        "{{model_name}}": str(model_name),
        "{model_name}": str(model_name),
        "{{model_hash}}": str(cfg.get("model_hash", "")),
        "{model_hash}": str(cfg.get("model_hash", "")),
    }

    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    p = Path(rendered).expanduser()

    if p.is_absolute():
        return p.resolve()

    project = cfg.get("project") or {}
    project_root = project.get("root")

    if project_root:
        return (Path(str(project_root)).expanduser().resolve() / p).resolve()

    return p.resolve()

def get_resolved_model_output_dir(
    cfg: Mapping[str, Any],
    *,
    env: str,
    model_name: str,
) -> Path:
    """
    Return the output directory for resolved_model.yaml/json.

    The resolved dynamic model should be saved beside model artifacts, under:

        paths:
          model_out: "artifacts/{{env}}/generated_models/{{model_name}}/model"

    Output:

        artifacts/<env>/generated_models/<model_name>/model/resolved_model.yaml
        artifacts/<env>/generated_models/<model_name>/model/resolved_model.json
    """
    paths = cfg.get("paths") or {}
    if not isinstance(paths, Mapping):
        raise ModelPreviewError("cfg['paths'] must be a mapping.")

    model_out_path = paths.get("model_out")
    if not model_out_path:
        raise ModelPreviewError(
            "cfg['paths']['model_out'] is required to save resolved_model.yaml. "
            "Expected example: artifacts/{{env}}/generated_models/{{model_name}}/model"
        )

    output_dir = _render_template_path(
        str(model_out_path),
        cfg=cfg,
        env=env,
        model_name=model_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Contract extraction
# =============================================================================

def _selected_model(cfg: Mapping[str, Any]) -> Json:
    """
    Return the selected single model.

    This expects cfg to have already passed through resolve_model_contract(),
    which reduces cfg["model"] to exactly one selected model block.
    """
    models = cfg.get("model")

    if not isinstance(models, list) or not models:
        raise ModelPreviewError(
            "Resolved cfg must contain non-empty cfg['model'] list. "
            "Call resolve_model_contract(...) before model preview."
        )

    if len(models) != 1:
        raise ModelPreviewError(
            "Resolved cfg must contain exactly one selected model before training. "
            f"Got {len(models)} entries. Call resolve_model_contract(...) first."
        )

    selected = models[0]
    if not isinstance(selected, dict):
        raise ModelPreviewError("Selected model entry must be a dict.")

    return selected


def _get_layers_from_resolved_cfg(cfg: Mapping[str, Any], selected: Mapping[str, Any]) -> List[Json]:
    """
    Extract resolved concrete layers.

    Priority:
    1. selected["architecture_args"]["layers"]
    2. cfg["trial_materialization"]["layers"]

    Both should match in AutoSearch mode. A named architecture (e.g.
    ResNet50_1D) is built from constructor arguments and has no layers.
    """
    arch_args = selected.get("architecture_args") or {}

    if not isinstance(arch_args, Mapping):
        raise ModelPreviewError("selected_model['architecture_args'] must be a mapping.")

    layers = arch_args.get("layers")

    if isinstance(layers, list) and layers:
        return deepcopy(layers)

    trial_materialization = cfg.get("trial_materialization") or {}
    if isinstance(trial_materialization, Mapping):
        tm_layers = trial_materialization.get("layers")
        if isinstance(tm_layers, list) and tm_layers:
            return deepcopy(tm_layers)

    if is_named_architecture(selected):
        return []

    raise ModelPreviewError(
        "Resolved model has no concrete layers. "
        "Expected selected_model.architecture_args.layers after resolve_model_contract(...)."
    )


def _clean_architecture_args(selected: Mapping[str, Any], layers: List[Json]) -> Json:
    """
    Return architecture_args without raw search_space.

    search_space is recipe input only. It must not be treated as the runtime model.
    """
    arch_args = selected.get("architecture_args") or {}
    if not isinstance(arch_args, Mapping):
        raise ModelPreviewError("architecture_args must be a mapping.")

    clean = {
        key: deepcopy(value)
        for key, value in arch_args.items()
        if key != "search_space"
    }
    if layers or not is_named_architecture(selected):
        clean["layers"] = deepcopy(layers)
    return clean


def _safe_trainer_payload(cfg: Mapping[str, Any]) -> Json:
    """
    Extract trainer fields useful for audit/debugging.
    """
    trainer = cfg.get("trainer") or {}
    if not isinstance(trainer, Mapping):
        raise ModelPreviewError("cfg['trainer'] must be a mapping.")

    preferred_keys = [
        "learning_rate",
        "optimizer",
        "loss",
        "loss_reduction",
        "epochs",
        "batch_size",
        "weight_decay",
        "momentum",
        "scheduler",
        "grad_clip_norm",
        "early_stopping",
        "patience",
        "min_delta",
    ]

    return {
        key: deepcopy(trainer[key])
        for key in preferred_keys
        if key in trainer
    }


def _safe_data_payload(cfg: Mapping[str, Any]) -> Json:
    """
    Keep lightweight data-shape details that explain why the dynamic model
    resolved the way it did, for example target_range/input length.
    """
    data = cfg.get("data")
    if isinstance(data, Mapping):
        keep_keys = [
            "target_range",
            "trace_len",
            "input_length",
            "num_classes",
            "database",
            "train_path",
            "val_path",
            "test_path",
        ]
        return {
            key: deepcopy(data[key])
            for key in keep_keys
            if key in data
        }

    return {}


def _safe_sca_payload(cfg: Mapping[str, Any]) -> Json:
    """
    Keep SCA fields that affect dynamic output classes.
    Example:
        leakage_model=HW -> 9 classes
        leakage_model=ID -> 256 classes
    """
    sca = cfg.get("sca") or {}
    if not isinstance(sca, Mapping):
        return {}

    keep_keys = [
        "leakage_model",
        "target_byte",
        "attack_byte",
        "dataset_type",
    ]

    return {
        key: deepcopy(sca[key])
        for key in keep_keys
        if key in sca
    }


def extract_resolved_model_payload(
    cfg: Mapping[str, Any],
    *,
    env: str,
    model_name: str,
) -> Json:
    """
    Create the payload saved to resolved_model.yaml/json.

    Important:
    This function assumes cfg is already resolved by resolve_model_contract().

    Schema principle:
    - resolved_model.architecture_args.layers is the canonical concrete model.
    - trial_materialization stores provenance, but does not duplicate layers.
    """
    selected = _selected_model(cfg)
    layers = _get_layers_from_resolved_cfg(cfg, selected)
    clean_arch_args = _clean_architecture_args(selected, layers)

    runtime_contract = cfg.get("runtime_contract") or {}
    if not isinstance(runtime_contract, Mapping):
        runtime_contract = {}

    trial_materialization = cfg.get("trial_materialization") or {}
    if not isinstance(trial_materialization, Mapping):
        trial_materialization = {}

    runtime = cfg.get("runtime") or {}
    if not isinstance(runtime, Mapping):
        runtime = {}

    # Keep trial provenance, but avoid duplicating full layer stack.
    trial_materialization_summary = {
        key: deepcopy(value)
        for key, value in trial_materialization.items()
        if key != "layers"
    }

    resolved_model = {
        "model_id": deepcopy(cfg.get("model_id") or selected.get("model_id")),
        "architecture": deepcopy(selected.get("architecture")),
        "architecture_args": clean_arch_args,
    }

    payload: Json = {
        "schema_version": "resolved_model.v1",
        "description": (
            "Concrete train.py-safe model resolved immediately before training. "
            "This is the dynamic model produced after search_space/SCC resolution."
        ),
        "env": str(env),
        "model_name": str(model_name),
        "model_hash": deepcopy(cfg.get("model_hash")),

        "runtime": {
            "execution_source": deepcopy(runtime.get("execution_source")),
            "driver": deepcopy(runtime.get("driver")),
            "selection_source": deepcopy(runtime.get("selection_source")),
            "trial_id": deepcopy(runtime.get("trial_id")),
            "trial_index": deepcopy(runtime.get("trial_index")),
            "trial_total": deepcopy(runtime.get("trial_total")),
            "sampled_key": deepcopy(runtime.get("sampled_key")),
            "dataset": deepcopy(runtime.get("dataset")),
        },

        "resolver": {
            "resolved": bool(runtime_contract.get("resolved", False)),
            "mode": deepcopy(runtime_contract.get("mode")),
            "lr_key": deepcopy(runtime_contract.get("lr_key")),
            "model_id": deepcopy(runtime_contract.get("model_id")),
        },

        "trainer": _safe_trainer_payload(cfg),
        "sca": _safe_sca_payload(cfg),
        "data": _safe_data_payload(cfg),

        "trial_materialization": trial_materialization_summary,
        "resolved_model": resolved_model,
    }

    return payload

def _find_model_entry(cfg: dict, model_id: str | None = None) -> dict:
    """
    Find the selected model entry from cfg["model"].

    If model_id is provided, select that model.
    If only one model exists, select it automatically.
    """
    models = cfg.get("model")

    if not isinstance(models, list) or not models:
        raise ValueError("cfg['model'] must be a non-empty list before training.")

    if model_id:
        wanted = str(model_id).strip()
        for model in models:
            if isinstance(model, dict) and str(model.get("model_id", "")).strip() == wanted:
                return model

        available = [
            str(m.get("model_id", ""))
            for m in models
            if isinstance(m, dict)
        ]
        raise ValueError(f"model_id={wanted!r} not found. Available: {available}")

    if len(models) == 1:
        if not isinstance(models[0], dict):
            raise ValueError("Only model entry must be a dict.")
        return models[0]

    available = [
        str(m.get("model_id", ""))
        for m in models
        if isinstance(m, dict)
    ]
    raise ValueError(
        f"model_id is required because multiple model entries exist: {available}"
    )


def _model_has_enabled_search_space(model_entry: dict) -> bool:
    """
    Return True only when the selected model itself has enabled search_space.

    Global cfg['auto_search']['enabled'] is not enough.
    """
    arch_args = model_entry.get("architecture_args") or {}
    if not isinstance(arch_args, dict):
        return False

    search_space = arch_args.get("search_space") or {}
    if not isinstance(search_space, dict):
        return False

    return bool(search_space.get("enabled", False))


def _model_has_concrete_layers(model_entry: dict) -> bool:
    """
    Return True when the selected model already has concrete runtime layers.
    """
    arch_args = model_entry.get("architecture_args") or {}
    if not isinstance(arch_args, dict):
        return False

    layers = arch_args.get("layers")
    return isinstance(layers, list) and len(layers) > 0


# =============================================================================
# Save/export
# =============================================================================

def save_resolved_model_preview(
    cfg: Mapping[str, Any],
    *,
    env: str,
    model_name: str,
    logger: Optional[Any] = None,
    save_json: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """
    Save resolved_model.yaml and optionally resolved_model.json.

    Must be called after resolve_model_contract(...) and before build_model(...).
    """
    output_dir = get_resolved_model_output_dir(cfg, env=env, model_name=model_name)
    payload = extract_resolved_model_payload(cfg, env=env, model_name=model_name)

    yaml_path = output_dir / "resolved_model.yaml"
    json_path: Optional[Path] = output_dir / "resolved_model.json" if save_json else None

    try:
        yaml_path.write_text(
            seal_text(yaml_path, yaml.safe_dump(
                payload,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )),
            encoding="utf-8",
        )

        if json_path is not None:
            json_path.write_text(
                seal_text(json_path, json.dumps(payload, indent=2, ensure_ascii=False)),
                encoding="utf-8",
            )

    except Exception as exc:
        raise ModelPreviewError(
            f"Failed to save resolved model preview under {output_dir}: {exc}"
        ) from exc

    if logger:
        logger.info("Resolved model YAML saved at: %s", str(yaml_path))
        if json_path is not None:
            logger.info("Resolved model JSON saved at: %s", str(json_path))
    else:
        print(f"Resolved model YAML saved at: {yaml_path}")
        if json_path is not None:
            print(f"Resolved model JSON saved at: {json_path}")

    return yaml_path, json_path


# =============================================================================
# Terminal preview
# =============================================================================

def _format_layer(layer: Mapping[str, Any], index: int) -> str:
    """
    Convert a dynamic YAML layer definition into one compact terminal line.
    """
    if not isinstance(layer, Mapping) or len(layer) != 1:
        return f"[{index:02d}] INVALID_LAYER {layer!r}"

    name, params = next(iter(layer.items()))
    params = params or {}

    if not isinstance(params, Mapping) or not params:
        return f"[{index:02d}] {name}"

    preferred_keys = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "dim",
        "type",
        "pool_type",
        "in_features",
        "out_features",
        "activation",
        "dropout",
        "p",
        "output_size",
    ]

    parts: List[str] = []

    for key in preferred_keys:
        if key in params:
            parts.append(f"{key}={params[key]}")

    for key, value in params.items():
        if key not in preferred_keys:
            parts.append(f"{key}={value}")

    return f"[{index:02d}] {str(name):<16} {' '.join(parts)}".rstrip()


def _resolved_scalar(value: Any) -> Any:
    """Unwrap a sweep list that has been resolved to a single value."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _mapping_or_first(value: Any) -> Mapping[str, Any]:
    """Return a mapping, taking the first entry of a list of alternatives."""
    if isinstance(value, list) and value:
        value = value[0]
    return value if isinstance(value, Mapping) else {}


def _experiment_lines(cfg: Mapping[str, Any], selected: Mapping[str, Any]) -> List[str]:
    """The training, evaluation and attack settings this run will use."""
    trainer = cfg.get("trainer") if isinstance(cfg.get("trainer"), Mapping) else {}
    optimizer = _mapping_or_first(trainer.get("optimizer"))
    scheduler = _mapping_or_first(optimizer.get("scheduler"))
    loss = _mapping_or_first(trainer.get("loss"))
    loss_args = loss.get("args") if isinstance(loss.get("args"), Mapping) else {}
    evaluation = trainer.get("evaluation") if isinstance(trainer.get("evaluation"), Mapping) else {}
    transform = _mapping_or_first(evaluation.get("output_transformation"))
    sca = cfg.get("sca") if isinstance(cfg.get("sca"), Mapping) else {}
    cv = cfg.get("cross_validation") if isinstance(cfg.get("cross_validation"), Mapping) else {}
    attack = cfg.get("perform_attack") if isinstance(cfg.get("perform_attack"), Mapping) else {}
    eval_policy = cfg.get("evaluation") if isinstance(cfg.get("evaluation"), Mapping) else {}
    arch_args = selected.get("architecture_args") if isinstance(selected.get("architecture_args"), Mapping) else {}

    scheduler_type = _resolved_scalar(scheduler.get("type", "none"))
    scheduler_text = str(scheduler_type)
    if str(scheduler_type).strip().lower() not in {"", "none", "null", "off"}:
        scheduler_text += f" args={dict(scheduler.get('args') or {})}"
    early_stopping = _resolved_scalar(trainer.get("early_stopping", False))
    early_text = str(early_stopping)
    if early_stopping:
        early_text += f" (patience={trainer.get('patience')}, min_delta={trainer.get('min_delta')})"
    validation_policy = (eval_policy.get("validation") or {}) if isinstance(eval_policy.get("validation"), Mapping) else {}
    test_policy = (eval_policy.get("test") or {}) if isinstance(eval_policy.get("test"), Mapping) else {}

    from services.side_channel_core.src.core.framework_switch import resolve_framework

    framework = resolve_framework(cfg, validate=False)
    init_text = "torch defaults (no deep_learning block)"
    if framework.declared:
        init_text = (
            f"deep_learning.{framework.block}: {framework.weight_init.name} {dict(framework.weight_init.args) or ''}"
            f" / bias {framework.bias_init.name} {dict(framework.bias_init.args) or ''}"
        )
    for key in ("weight_init", "bias_init"):
        if arch_args.get(key) not in (None, ""):
            init_text += f"; model {key}={arch_args.get(key)}"
    for idx, layer in enumerate(arch_args.get("layers") or []):        # a layer's own initialiser (layer_init_overrides)
        if isinstance(layer, Mapping) and len(layer) == 1:
            name, params = next(iter(layer.items()))
            own = {k: v for k, v in (params or {}).items() if k in ("weight_init", "bias_init") and v not in (None, "")}
            if own:
                init_text += f"; layer {idx} ({name}) " + " ".join(f"{k}={v}" for k, v in own.items())

    return [
        f"framework       : {framework.framework}"
        + (" (Keras 3, torch backend)" if framework.is_keras else " (torch.nn / torch.optim)"),
        f"seed            : {cfg.get('seed')}",
        f"epochs          : {_resolved_scalar(trainer.get('epochs'))}",
        f"batch_size      : {_resolved_scalar(trainer.get('batch_size'))}",
        f"optimizer       : {_resolved_scalar(optimizer.get('type'))} args={dict(optimizer.get('args') or {})}",
        f"learning_rate   : {_resolved_scalar(optimizer.get('learning_rate', trainer.get('learning_rate')))}",
        f"weight_decay    : {_resolved_scalar(optimizer.get('weight_decay', trainer.get('weight_decay')))}",
        f"scheduler       : {scheduler_text}",
        f"loss            : {_resolved_scalar(loss.get('type'))} reduction={loss_args.get('reduction', 'mean')}"
        f" label_smoothing={loss_args.get('label_smoothing', 0.0)}",
        "cross_entropy   : metric computed as "
        + ("keras.losses.categorical_crossentropy of the logits, like the loss"
           if framework.is_keras
           else "PyTorch cross-entropy of the logits"),
        f"grad_clip_norm  : {trainer.get('grad_clip_norm')}",
        f"early_stopping  : {early_text}",
        f"monitor_metric  : {_resolved_scalar(trainer.get('monitor_metric'))}",
        f"output_transform: {_resolved_scalar(transform.get('type'))}",
        f"initialisation  : {init_text}",
        f"checkpointing   : validation={validation_policy.get('checkpointing')} test.load_model={test_policy.get('load_model')}",
        f"leakage_model   : {_resolved_scalar(sca.get('leakage_model'))}  attack_byte={_resolved_scalar(sca.get('attack_byte'))}"
        f"  variant={sca.get('variant')}",
        f"cross_validation: enabled={cv.get('enabled', False)} n_folds={cv.get('n_folds')} strategy={cv.get('strategy')}"
        f" shuffle={cv.get('shuffle')}",
        f"perform_attack  : validation={attack.get('validation')} test={attack.get('test')}"
        f" step={attack.get('attack_step')} ge_thresholds={attack.get('ge_thresholds')}",
    ]


def build_resolved_model_preview_text(cfg: Mapping[str, Any]) -> str:
    """
    Build terminal preview text without printing it.

    Useful for:
    - logger output
    - dashboard logs
    - tests
    """
    selected = _selected_model(cfg)
    layers = _get_layers_from_resolved_cfg(cfg, selected)

    runtime_contract = cfg.get("runtime_contract") or {}
    trial_materialization = cfg.get("trial_materialization") or {}

    model_id = cfg.get("model_id") or selected.get("model_id")
    architecture = selected.get("architecture")
    mode = runtime_contract.get("mode", "unknown") if isinstance(runtime_contract, Mapping) else "unknown"

    lines: List[str] = []
    lines.append("")
    lines.append("=" * 88)
    lines.append("RESOLVED DYNAMIC MODEL BEFORE TRAINING")
    lines.append("=" * 88)
    lines.append(f"model_id        : {model_id}")
    lines.append(f"architecture    : {architecture}")
    lines.append(f"resolver_mode   : {mode}")
    lines.append(f"layers          : {len(layers) if layers else 'named architecture'}")
    lines.extend(_experiment_lines(cfg, selected))

    if not layers:
        lines.append(f"source          : {architecture} class, built from architecture_args")
    elif isinstance(trial_materialization, Mapping) and trial_materialization:
        lines.append("source          : trial_materialization / resolved architecture_args.layers")
    else:
        lines.append("source          : resolved architecture_args.layers")

    lines.append("-" * 88)

    for index, layer in enumerate(layers):
        lines.append(_format_layer(layer, index))

    if not layers:
        arch_args = selected.get("architecture_args") or {}
        for key, value in arch_args.items():
            lines.append(f"{key:<16}: {value}")

    lines.append("=" * 88)
    lines.append("")

    return "\n".join(lines)


def print_resolved_model_preview(
    cfg: Mapping[str, Any],
    *,
    logger: Optional[Any] = None,
) -> None:
    """
    Print/log the resolved dynamic model before training starts.
    """
    text = build_resolved_model_preview_text(cfg)

    if logger:
        logger.info("%s", text)
    else:
        print(text)
