# services/visualization/vscode_monitor/choices.py
"""
The options of the config keys that take one of a known set, for the panel's menus.

    python -m services.visualization.vscode_monitor.choices static
    python -m services.visualization.vscode_monitor.choices refs --root ROOT

``static`` prints ``{"rules": {file: [rule, ...]}, "graph": {...}, "errors": [...]}``. A rule is
``{"p": "jobs.*.type", "o": [...], "free": false, "by": {...}}``: ``p`` a key path of
that file (``*`` one list index or mapping key), ``o`` the options, ``free`` true where
the code accepts names beyond the list (a torch class, a font), so the panel suggests
and still lets one type; ``by`` (optional) narrows the options by a sibling key of the
item at depth ``at`` (graph jobs' processes by their source). Every list is read from
the code that checks the key (the constant named beside it), never retyped here: a
module that cannot be imported leaves its keys as plain text boxes and is named in
``errors``. Booleans need no rule: the panel offers true / false for any boolean value.

``graph`` is what the graph_config.yaml forms draw beyond a menu (graph_forms): the plot
types each metric takes, the legend handle shapes, every style.axes key, the text rule
targets and fields, each export format's settings, with graph_manager's defaults.

A rule with ``ref`` instead of ``o`` takes its options from ``refs`` (``runs``, ``datasets``,
``model_ids``, ``queues``, or ``projects``: the project names).

``refs`` prints the project-defined names, per project: data entry names, model ids,
finished runs, queue keys, ENVs. They change with the files, so they are read on every
layout, cheaply (no torch).

Author: Shraddha Mishra
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

Rule = dict[str, Any]


def _names(values: Iterable[Any]) -> list[str]:
    return sorted({str(v) for v in values}, key=lambda v: (v.lower(), v))


def _ordered(values: Iterable[Any]) -> list[str]:
    return [str(v) for v in values]


# ── the option lists, each from its own module ───────────────────────────────

def _by_framework(pytorch: list[str], keras: list[str]) -> dict[str, Any]:
    """A menu that follows deep_learning.framework (menus.js reads the key from the file's root). The other
    framework's options stay in the menu, disabled, marked "available in <framework>" (others: true)."""
    from core.framework_switch import DEFAULT_FRAMEWORK

    return {"at": 0, "key": "deep_learning.framework", "default": DEFAULT_FRAMEWORK, "map": {"pytorch": pytorch, "keras": keras},
            "others": True}


def _keras_activations(names: list[str]) -> list[str]:
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")
    from core.framework_switch import keras_activations

    return keras_activations(names)


def _keras_optimizers() -> list[str]:
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")      # TraceFlow's Keras runs on the torch backend
    from core.framework_switch import keras_optimizers

    return keras_optimizers()


def _keras_schedules() -> list[str]:
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")
    from core.framework_switch import keras_schedules

    return keras_schedules()


def _metric_spellings(valid: Any) -> list[str]:
    """What metrics_to_log takes: every metric sca.metrics computes, a fit time as the file spells it
    (fit_time_{s}, mean_fit_time_{per_epoch}, ...; canonical_fit_time_name reads both)."""
    from sca.metrics import canonical_fit_time_name

    out = []
    for name in _names(valid):
        if canonical_fit_time_name(name) is not None:
            stem, _, unit = name.rpartition("_") if not name.endswith("per_epoch") else (name[:-len("_per_epoch")], "_", "per_epoch")
            name = f"{stem}_{{{unit}}}"
        if name not in out:
            out.append(name)
    return out


def _params() -> list[Rule]:
    from core.framework_switch import FRAMEWORKS, KERAS_LOSSES, TORCH_INITIALIZERS
    from sca.ascad_data_loader import _FITTED_STEPS, _SUPPORTED_NORMALIZATION_MODES
    from sca.dataset_types import TYPES
    from sca.key_enumeration import SUPPORTED_ENUMERATION_MODES
    from sca.leakage_models import _VALID_LEAKAGE_MODELS
    from sca.perform_attack import _ATTACK_DISTRIBUTIONS
    from core.model_contract_resolver import _LAYER_ARCHITECTURES
    from utils.checkpoint_policy import SUPPORTED_TEST_MODEL_POLICIES
    from utils.cross_validation import SUPPORTED_STRATEGIES
    from utils.loss_util import LOSS_REGISTRY
    from utils.optimizer_util import OPTIMIZER_REGISTRY, _SCHEDULER_INTERVALS
    from utils.output_transformation_util import OUTPUT_TRANSFORM_REGISTRY
    from utils.precision import PRECISIONS
    from utils.sca_sweep import SUPPORTED_CANDIDATE_RANKING_MODES
    from sca.metrics import VALID_METRIC_PROCESSES

    inits = ["default", *_names(TORCH_INITIALIZERS)]
    rules = [
        {"p": "deep_learning.framework", "o": _names(FRAMEWORKS)},
        {"p": "deep_learning.pytorch.weight_init", "o": inits},
        {"p": "deep_learning.pytorch.bias_init", "o": inits},
        {"p": "sca.leakage_model", "o": _names(_VALID_LEAKAGE_MODELS)},
        # refs: the bytes the project's data entries allow; the other bytes listed disabled, naming the entries that
        # define none (owner 2026-10-23: only a byte some entry defines a target for is choosable)
        {"p": "sca.attack_byte", "ref": "attack_bytes", "others_ref": "attack_bytes_others"},
        {"p": "sca.candidate_ranking.mode", "o": _names(SUPPORTED_CANDIDATE_RANKING_MODES),
         "l": {"all": "all (rank all 256 key candidates)", "none": "none (rank no candidates)",
               "top_k": "top_k (rank the k best candidates)"}},
        {"p": "sca.key_enumeration.mode", "o": _names(SUPPORTED_ENUMERATION_MODES)},
        {"p": "evaluation.test.load_model", "o": _names(SUPPORTED_TEST_MODEL_POLICIES)},
        {"p": "cross_validation.strategy", "o": _names(SUPPORTED_STRATEGIES)},
        {"p": "perform_attack.*.attack_distribution", "o": _names(_ATTACK_DISTRIBUTIONS)},
        {"p": "data.*.type", "o": _ordered(TYPES)},
        {"p": "data.*.trace_normalization.mode", "o": _names(_SUPPORTED_NORMALIZATION_MODES)},
        {"p": "data.*.trace_normalization.steps.*", "o": _ordered(_FITTED_STEPS)},
        {"p": "model.*.architecture", "o": _names(_LAYER_ARCHITECTURES), "free": True},
        # by the run's framework: the activations Keras builds; the others listed disabled (owner 2026-10-23: anything
        # framework-specific stays visible, unavailable, saying which framework has it)
        {"p": "model.*.architecture_args.layers.*.*.activation", "o": _activations(), "free": True,
         "by": _by_framework(_activations(), _keras_activations(_activations()))},
        {"p": "trainer.precision", "o": _names(PRECISIONS)},
        # by the run's framework: under keras only the losses it has (framework_switch.build_loss)
        {"p": "trainer.loss.type", "o": _names(LOSS_REGISTRY), "by": _by_framework(_names(LOSS_REGISTRY), list(KERAS_LOSSES))},
        # by the run's framework: torch.optim classes (optimizer_util), keras.optimizers classes (framework_switch)
        {"p": "trainer.optimizer.type", "o": _names(OPTIMIZER_REGISTRY), "by": _by_framework(_names(OPTIMIZER_REGISTRY), _keras_optimizers())},
        {"p": "trainer.optimizer.*.type", "o": _names(OPTIMIZER_REGISTRY), "by": _by_framework(_names(OPTIMIZER_REGISTRY), _keras_optimizers())},
        # by the run's framework: torch.optim.lr_scheduler classes (optimizer_util), keras.optimizers.schedules (framework_switch)
        {"p": "trainer.optimizer.scheduler.type", "o": ["none", *_schedulers()], "by": _by_framework(["none", *_schedulers()], ["none", *_keras_schedules()])},
        {"p": "trainer.optimizer.*.scheduler.type", "o": ["none", *_schedulers()], "by": _by_framework(["none", *_schedulers()], ["none", *_keras_schedules()])},
        {"p": "trainer.optimizer.scheduler.interval", "o": _names(_SCHEDULER_INTERVALS)},
        {"p": "trainer.optimizer.*.scheduler.interval", "o": _names(_SCHEDULER_INTERVALS)},
        {"p": "trainer.evaluation.output_transformation.type", "o": _names(OUTPUT_TRANSFORM_REGISTRY)},
        {"p": "metrics_to_log.*", "o": _metric_spellings(VALID_METRIC_PROCESSES)},
        {"p": "trainer.monitor_metric", "o": ["none", *_names(VALID_METRIC_PROCESSES)], "free": True},
        {"p": "trainer.early_stopping.monitor_metric", "o": ["none", *_names(VALID_METRIC_PROCESSES)], "free": True},
    ]
    return rules


# The true / false keys of the project files, by file: a key the file holds empty (a new project) or lacks is a
# true / false menu too (a value already there is a boolean: the panel offers true / false for any boolean).
# From the readers' _strict_bool / bool(...) of each key (2026-10-22); a new boolean key goes here.
BOOLEAN_KEYS = {
    "params.yaml": (
        "dataloader.pin_memory", "dataloader.persistent_workers", "dataloader.drop_last_train",
        "sca.key_enumeration.enabled", "cross_validation.refit.enabled",
        "sca.key_enumeration.stop_on_success", "evaluation.validation.checkpointing", "evaluation.validation.judge_log",
        "cross_validation.enabled", "cross_validation.shuffle", "cross_validation.report_aggregate",
        "perform_attack.enabled", "perform_attack.validation.shuffle", "perform_attack.test.shuffle",
        "key_recovery_reporting.enabled", "key_recovery_reporting.log_validation", "key_recovery_reporting.log_test",
        "key_recovery_reporting.save_validation_reports", "key_recovery_reporting.save_test_report",
        "key_recovery_reporting.require_stable_recovery", "key_recovery_reporting.include_score_margin",
        "data.*.generate_data_out", "data.*.debug_h5_inspect", "data.*.show_contract", "data.*.require_contract",
        "metrics_cfg.preds_are_probs", "metrics_cfg.lm_enabled", "metrics_cfg.strict_prob_check",
        "model.*.architecture_args.strict_graph_args",
        "trainer.early_stopping", "trainer.early_stopping.enabled", "trainer.loss.args.from_logits",
    ),
    "model_judge.yaml": ("model_judge.filters.require_done",),
    # graph_config.yaml: the true / false keys graph_manager reads (graph_forms types them for a key the file lacks;
    # these give the same menu to a key the file holds empty)
    "graph_config.yaml": tuple(f"{section}.*.{key}" for section in ("jobs", "faceting") for key in (
        "enabled", "legend.enabled", "legend.frame", "export.save_raw_plot_data", "export.svg.sanitize", "export.svg.validate_xml",
        "text_rules.*.bold", "text_rules.*.italic", "style.text_rules.*.bold", "style.text_rules.*.italic",
        "style.grid.enabled", "style.std_band.enabled", "style.axes.x_grid", "style.axes.x_grid_first")),
    "test_config.yaml": ("test_request.perform_attack.test.shuffle", "test_request.models.*.perform_attack.test.shuffle"),
}


def _booleans(file: str) -> list[Rule]:
    return [{"p": p, "o": ["true", "false"]} for p in BOOLEAN_KEYS.get(file, ())]


def _params_text_sets() -> list[Rule]:
    """Keys whose set the code names where it checks them."""
    from core.framework_switch import TORCH_INITIALIZERS
    from core.model import POOLING_TYPES_1D
    from sca.key_recovery import DEFAULT_RECOVERY_RANK, KEY_DISPLAYS
    from utils.type_coerce import LOSS_REDUCTIONS

    inits = _names(TORCH_INITIALIZERS)
    layer = "model.*.architecture_args.layers.*"
    return [{"p": "key_recovery_reporting.key_display", "o": _ordered(KEY_DISPLAYS)},
            {"p": "key_recovery_reporting.recovery_rank", "o": [str(DEFAULT_RECOVERY_RANK)]},
            {"p": f"{layer}.Pooling.type", "o": _ordered(POOLING_TYPES_1D)},
            {"p": f"{layer}.Linear.out_features", "o": ["auto"], "free": True},
            {"p": "model.*.architecture_args.weight_init.pytorch.name", "o": inits},
            {"p": "model.*.architecture_args.bias_init.pytorch.name", "o": inits},
            {"p": "trainer.loss.args.reduction", "o": _names(LOSS_REDUCTIONS)},
            {"p": "device", "o": ["cuda", "cpu"], "free": True}]


def _schedulers() -> list[str]:
    import torch

    return _names(n for n, c in vars(torch.optim.lr_scheduler).items()
                  if isinstance(c, type) and n[:1].isupper() and hasattr(c, "step") and not n.startswith("_")
                  and n not in {"LRScheduler", "Optimizer", "Counter"})


def _keras_params() -> list[Rule]:
    """The Keras side: names its own registries accept (keras.initializers, keras.optimizers)."""
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")      # TraceFlow's Keras runs on the torch backend
    import keras

    inits = _names(n for n in vars(keras.initializers) if n[:1].islower() and not n.startswith("_")
                   and n not in {"get", "serialize", "deserialize"})
    return [{"p": "deep_learning.keras.weight_init", "o": inits, "free": True},
            {"p": "deep_learning.keras.bias_init", "o": inits, "free": True},
            {"p": "model.*.architecture_args.weight_init.keras.name", "o": inits, "free": True},
            {"p": "model.*.architecture_args.bias_init.keras.name", "o": inits, "free": True}]


def _fonts() -> list[str]:
    """The font families a figure can use: matplotlib's generic ones, the Latin Modern fonts graph_manager
    registers when it draws (plot_style.register_tex_fonts), then every font matplotlib finds installed."""
    from matplotlib import font_manager
    from services.graph_manager.plot_style import TEX_FONT_FAMILIES

    tex = _names(TEX_FONT_FAMILIES)
    return ["serif", "sans-serif", "monospace", *tex, *[f for f in _names(font_manager.get_font_names()) if f not in tex]]


def params_forms() -> dict[str, Any]:
    """params.yaml forms: which deep_learning block each framework reads, and, for every spelling
    metrics_to_log takes, the monitor names it gives a run (utils/monitor_metric.monitorable_metrics):
    the panel's trainer.monitor_metric menu is the monitors of the file's metrics_to_log."""
    from sca.dataset_types import TYPES
    from core.framework_switch import DEFAULT_FRAMEWORK, FRAMEWORK_BLOCKS
    from sca.metrics import METRIC_OPTIONS, VALID_METRIC_PROCESSES
    from utils.monitor_metric import VALIDATION_OBJECTIVE, monitorable_metrics

    spellings = set(VALID_METRIC_PROCESSES) | {"ge", "final_ge", "top_5_accuracy", "fit_time", "mean_fit_time"}
    spellings |= {f"{m}fit_time_{{{u}}}" for m in ("", "mean_") for u in ("s", "h", "per_epoch")}
    monitors = {}
    for name in sorted(spellings):
        try:
            got = [m for m in monitorable_metrics({"metrics_to_log": [name]}) if m != VALIDATION_OBJECTIVE]
        except Exception:      # noqa: BLE001 - a spelling the metrics module refuses gives no monitor
            got = []
        if got:
            monitors[name] = got
    return {"frameworks": {"blocks": dict(FRAMEWORK_BLOCKS), "default": DEFAULT_FRAMEWORK},
            "monitor": {"by_metric": monitors, "always": [VALIDATION_OBJECTIVE]},
            "data_template": data_template(),
            "normalization": normalization_forms(),
            "profiling_split": profiling_split_forms(),
            # each type's target bytes (dataset_types): the panel writes `type.attack_byte = {…}` beside a data entry's type
            "type_bytes": {name: (sorted(t.attack_bytes) if t.attack_bytes is not None else "any") for name, t in TYPES.items()},
            # metrics_to_log entries that take a value ({name: {option: value}}): asked when added
            "metric_options": {name: {"option": opt, "default": d} for name, (opt, d) in METRIC_OPTIONS.items()}}


BLUEPRINT = Path(__file__).resolve().parents[2] / "side_channel_core" / "config" / "blueprint"
# The params.yaml keys that join the run hash (utils/model_naming.make_experiment_hash; cross_validation only when
# enabled, dataloader.drop_last_train only when false): a key staged under one of them gives a new run of an already
# trained model a new hash.
HASHED = ("data", "model", "trainer", "seed", "device", "deep_learning", "sca.attack_byte", "sca.leakage_model",
          "metrics_to_log", "perform_attack", "evaluation", "metrics_cfg", "validation.rank_step",
          "validation.rank_eval_interval", "key_recovery_reporting.recovery_rank",
          "key_recovery_reporting.require_stable_recovery", "cross_validation.enabled", "cross_validation.n_folds",
          "cross_validation.strategy", "cross_validation.shuffle", "dataloader.drop_last_train")
# Keys inside a HASHED block that the hash leaves out: evaluation.validation.judge_log, a log line per epoch
# (utils/validation_judge.py; model_naming._hashed_evaluation)
# and data[].attack_trace_budget, read only when a trained run is tested (dataset_types.TEST_TIME_KEYS; owner 2026-10-27)
UNHASHED = ("evaluation.validation.judge_log", "data.*.attack_trace_budget")
# blueprint/templates.yaml: one item of each list a new project starts empty (or with one item), by the list's path
TEMPLATE_LISTS = {"data": "data", "model": "model", "trainer.optimizer": "optimizer",
                  "model_import.models": "import_model"}       # import_config.yaml: an import's keys = the defaults'



def normalization_forms() -> dict[str, Any]:
    """data[].trace_normalization: the keys each mode reads (sca/ascad_data_loader.NORMALIZATION_KEYS), what each
    starts as, and its hint: the panel shows the chosen mode's keys only, drafted when the entry lacks them."""
    from sca.ascad_data_loader import NORMALIZATION_KEYS

    return {"keys": {mode: list(keys) for mode, keys in NORMALIZATION_KEYS.items()},
            "defaults": {"steps": [], "feature_range": []},
            "hints": {"eps": "the least std (zscore) or range (minmax) accepted, e.g. 1.0e-8 (default)",
                      "mean": "the fixed mean subtracted from every sample", "std": "the fixed std every sample is divided by",
                      "min": "the fixed minimum of the raw range", "max": "the fixed maximum of the raw range",
                      "steps": "scikit-learn scalers chained in this order, fitted on the training traces: standard, minmax",
                      "feature_range": "the minmax step's range, e.g. 0, 1 (default) or -1, 1"},
            "note": {"fitted": "feature_range is read when steps include minmax"}}


def profiling_split_forms() -> dict[str, Any]:
    """data[].profiling_split: the arguments enabled: true reads, the values the panel stages for them when enabled is
    switched to true (sca/dataset_types.PROFILING_SPLIT_DEFAULTS), and each one's hint."""
    from sca.dataset_types import PROFILING_SPLIT_DEFAULTS

    return {"defaults": dict(PROFILING_SPLIT_DEFAULTS),
            "hints": {"test_size": "the held-out part: a fraction in (0, 1), rounded up, or a row count "
                                   "(default 0.1, a 90/10 split; scikit-learn's own default is 0.25)",
                      "shuffle": "true: a random part (scikit-learn's default); false: the first rows",
                      "random_state": "the shuffle's seed (Wouters et al. 0); empty (null): unseeded, a new cut every run, "
                                      "not reproducible (Zaid et al.'s random.shuffle)"}}


def data_template() -> dict[str, Any]:
    """The data entry of blueprint/templates.yaml: what Add a Dataset writes (a new project's data: starts empty)."""
    return dict(_templates().get("data") or {})


def _templates() -> dict[str, Any]:
    import yaml

    loaded = yaml.safe_load((BLUEPRINT / "templates.yaml").read_text(encoding="utf-8")) or {}
    return {k: v for k, v in loaded.items() if isinstance(v, dict)}


def _hints(text: str) -> dict[str, str]:
    """Each key's comment in a blueprint file (yamledit.comments)."""
    from services.visualization.vscode_monitor.yamledit import comments

    return comments(text)


def blueprint_forms() -> dict[str, Any]:
    """What a project's config files may hold, for the panel: each scaffolded blueprint file as data (every key
    with the value a new project starts with) and each key's hint (its comment there), and templates.yaml's
    items of the lists a project fills itself (data entries, models, optimizers). A key a project's file lacks is
    drawn from here with its value and its menu; the panel's Load Blueprint stages them."""
    from services.visualization.vscode_monitor import yamlspan
    from services.visualization.vscode_monitor.layout import SCAFFOLD_FILES

    files = {}
    for name in SCAFFOLD_FILES:
        if name == "graph_config.yaml":                  # its forms are graph_forms (graph_manager's own defaults)
            continue
        text = (BLUEPRINT / name).read_text(encoding="utf-8")
        files[name] = {"tree": yamlspan.load(text), "hints": _hints(text)}
    text = (BLUEPRINT / "templates.yaml").read_text(encoding="utf-8")
    return {"files": files, "templates": _templates(), "template_hints": _hints(text), "lists": TEMPLATE_LISTS,
            "hashed": list(HASHED), "unhashed": list(UNHASHED)}


_LAYER_GROUPS = (                       # the [+] menu of a model's layers, by what a layer does
    ("Convolution", lambda n: n.startswith(("Conv", "ConvTranspose"))),
    ("Pooling", lambda n: n in {"Pooling", "SPP"} or n.startswith("SpectralPool")),
    ("Dense", lambda n: n in {"Linear", "LazyLinear", "Embedding"}),
    ("Normalization", lambda n: "Norm" in n),
    ("Dropout", lambda n: "Dropout" in n),
    ("Shape", lambda n: n in {"Flatten", "View", "ZeroPad1d", "Upsample", "PixelShuffle"}),
    ("Recurrent", lambda n: n in {"RNN", "LSTM", "GRU"}),
    ("Attention", lambda n: "Attention" in n or n.startswith("Transformer")),
    ("Merge", lambda n: n in {"Add", "Concatenate", "Identity"}),
)
_SKIP_ARGS = {"self", "device", "dtype", "factory_kwargs"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value if isinstance(value, (bool, int, float, str)) or value is None else str(value)


def _torch_set(source_of: Any, name: str) -> list[str]:
    """A set literal torch writes in a constructor (valid_padding_modes = {...}): its options, else []."""
    import inspect
    import re

    try:
        m = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", inspect.getsource(source_of))
    except (OSError, TypeError):
        return []
    return sorted(re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))) if m else []


def _runs(factory: Any, name: str, **kwargs: Any) -> bool:
    """The layer builds with kwargs and runs forward on a tiny input of its dimensions (torch decides)."""
    import torch

    dims = 3 if name.endswith("3d") else 2 if name.endswith("2d") else 1
    try:
        factory(1, 1, 3, **kwargs)(torch.zeros(1, 1, *([8] * dims)))
        return True
    except Exception:  # noqa: BLE001 - refused: not an option of this layer
        return False


def _activations() -> list[str]:
    from core.model import _ACTIVATION_MAP

    return _names(n for n in _ACTIVATION_MAP if n and _activation_builds(n))


def _activation_builds(name: str) -> bool:
    """core.model builds the activation from its name alone, as the panel writes it (threshold needs arguments)."""
    from core.model import _activation_module

    try:
        _activation_module(name, "menu")
        return True
    except Exception:  # noqa: BLE001
        return False


def model_forms() -> dict[str, Any]:
    """params.yaml's model block: every layer core.model builds (_LAYER_MAP) by group, with each argument
    of its class (torch.nn's own signature: default, kind, options) and TraceFlow's `activation`; the
    layer kinds (CNN, MLP, ...) and the named models (a class without a layer list); under keras, the
    layers and Pooling types framework_switch maps."""
    import inspect

    import torch.nn as nn
    from core.framework_switch import KERAS_LAYERS, KERAS_POOLING
    from core.model import _ACTIVATION_MAP, _LAYER_MAP, LAYER_LIST_LIMITS, POOLING_TYPES_1D
    from core.model_contract_resolver import _LAYER_ARCHITECTURES

    activations = _names(n for n in _ACTIVATION_MAP if n and _activation_builds(n))
    conv_modes = _torch_set(nn.modules.conv._ConvNd.__init__, "valid_padding_modes")
    conv_strings = _torch_set(nn.modules.conv._ConvNd.__init__, "valid_padding_strings")
    layers, named = {}, []
    for name, factory in _LAYER_MAP.items():
        group = next((g for g, test in _LAYER_GROUPS if test(name)), None)
        if group is None:                                # a whole network (ResNet50_1D, ...): a named model
            named.append(name)
            continue
        try:
            sig = inspect.signature(factory.__init__ if inspect.isclass(factory) else factory)
        except (TypeError, ValueError):
            continue
        args = []
        for p in sig.parameters.values():
            if p.name in _SKIP_ARGS or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            d = None if p.default is p.empty else _jsonable(p.default)
            t = ("bool" if isinstance(d, bool) else "int" if isinstance(d, int) else "num" if isinstance(d, float)
                 else "list" if isinstance(d, list) else "text" if isinstance(d, str) else "any")
            spec = {"k": p.name, "t": t, "d": d, "required": p.default is p.empty}
            if name == "Pooling" and p.name == "type":
                spec.update(t="enum", o=list(POOLING_TYPES_1D))
            elif name.startswith("Conv") and p.name == "padding_mode" and conv_modes:
                spec.update(t="enum", o=[m for m in conv_modes if _runs(factory, name, padding_mode=m)])   # ConvTranspose: zeros
            elif name.startswith("Conv") and p.name == "padding" and conv_strings:
                spec.update(t="any", o=[s for s in conv_strings if _runs(factory, name, padding=s)])       # a number, or same / valid
            elif name in {"Linear", "LazyLinear"} and p.name == "out_features":
                spec.update(t="any", o=["auto"])               # auto: the leakage model's class count
            args.append(spec)
        args.append({"k": "activation", "t": "enum", "o": activations, "d": None, "required": False, "traceflow": True})
        layers[name] = {"group": group, "args": args}
    kind_of = {"Convolution": "CNN", "Attention": "TRANSFORMER", "Recurrent": "HYBRID"}
    # a kind is disabled while none of the layers that make it builds from a layer list (core.model's reasons)
    disabled = {}
    for group, kind in kind_of.items():
        members = [n for n, spec in layers.items() if spec["group"] == group]
        if members and all(n in LAYER_LIST_LIMITS for n in members):
            disabled[kind] = "none of its layers builds from a layer list on 1-D traces: " + "; ".join(
                f"{n}: {LAYER_LIST_LIMITS[n]}" for n in members)
    return {"groups": [g for g, _ in _LAYER_GROUPS], "layers": layers, "named": named,
            "kinds": sorted(_LAYER_ARCHITECTURES),
            "limits": dict(LAYER_LIST_LIMITS), "disabledKinds": disabled,
            "kerasKinds": ["CNN", "MLP"],            # Keras maps no attention or recurrent layer (framework_switch)
            # what a layer list makes (the panel shows it as the model's architecture once it has layers)
            "kindOf": kind_of,
            "keras": {"layers": list(KERAS_LAYERS), "pooling": list(KERAS_POOLING), "activations": _keras_activations(activations)}}


def graph_forms() -> dict[str, Any]:
    """The graph_config.yaml forms: every key graph_manager reads in the blocks the panel draws
    whole (export, legend, style.axes, text_rules), with its type, options and default."""
    from matplotlib.lines import Line2D
    from services.graph_manager import faceting as F
    from services.graph_manager import graph_job_validator as V
    from services.graph_manager import metric_router as R
    from services.graph_manager.plot_style import DEFAULT_STYLE
    from services.graph_manager.plotters import POSITION_MAP
    from services.visualization.utils import fig_theme
    from services.visualization.vscode_monitor import model_diagram as MD

    markers = _names(str(m) for m in Line2D.markers if isinstance(m, str) and m.strip() and m not in {"None", "none"})
    names = set(R.METRIC_ALIASES) | set(R.METRIC_ALIASES.values())
    for name in dir(R):
        value = getattr(R, name)
        if name.endswith("_METRICS") and isinstance(value, (set, frozenset, tuple, list)):
            names |= {str(v) for v in value}
    kinds = {n.lower(): R.metric_kind(n) for n in sorted(names) if R.metric_kind(n) != "scalar"}
    axes = DEFAULT_STYLE["axes"]
    scales = _names(V.VALID_SCALES)
    colors = ["black", "gray", *[f"C{i}" for i in range(10)]]
    fonts = _fonts()
    rule = [{"k": "text", "t": "text", "d": ""}, {"k": "bold", "t": "bool", "d": False},
            {"k": "italic", "t": "bool", "d": False}, {"k": "color", "t": "color", "o": colors, "d": "black"},
            {"k": "font.family", "t": "enum", "o": fonts, "d": "serif"}, {"k": "font.size", "t": "int", "d": 10}]
    first = ["title", "x_label", "y_label", "panel_label", "legend_text"]
    order = lambda targets: [t for t in first if t in targets] + _names(t for t in targets if t not in first)  # noqa: E731
    return {
        "plotTypes": {"table": {k: {c: list(v) for c, v in t.items()} for k, t in R.PLOT_TYPES.items()},
                      "order": ["auto", *R.XY_PLOT_TYPES, "strip", "bar", "barh", "heatmap"], "overlay": list(R.OVERLAY_JOB_TYPES),
                      "kinds": kinds, "curvePrefixes": ["sr_curve_"]},
        "handle": {"job": ["line", *markers], "facet": ["line", "patch", *markers]},
        "legend": {
            "job": [{"k": "enabled", "t": "bool", "d": True}, {"k": "position", "t": "enum", "o": list(POSITION_MAP), "d": "best"},
                    {"k": "handle", "t": "handle", "d": None}, {"k": "mode", "t": "enum", "o": _names(V.VALID_LEGEND_MODES), "d": "auto"}],
            "facet": [{"k": "enabled", "t": "bool", "d": True}, {"k": "position", "t": "enum", "o": _names(F.VALID_LEGEND_POSITIONS), "d": "bottom"},
                      {"k": "columns", "t": "int", "d": None}, {"k": "title", "t": "text", "d": ""}, {"k": "frame", "t": "bool", "d": False},
                      {"k": "handle", "t": "handle", "d": ["line", 1.5]}, {"k": "mode", "t": "enum", "o": _names(F.VALID_LEGEND_MODES), "d": "auto"}],
        },
        "axes": [{"k": "x_scale", "t": "enum", "o": scales, "d": axes["x_scale"]},
                 {"k": "y_scale", "t": "enum", "o": scales, "d": axes["y_scale"]},
                 {"k": "x_limits", "t": "pair"}, {"k": "y_limits", "t": "pair"},
                 {"k": "x_step", "t": "auto"}, {"k": "y_step", "t": "auto"}, {"k": "x_grid", "t": "auto"},
                 {"k": "x_grid_first", "t": "bool", "d": axes["x_grid_first"]},
                 {"k": "x_margin", "t": "num", "d": axes["x_margin"]}, {"k": "y_margin", "t": "num", "d": axes["y_margin"]}],
        # array_epoch.mode: graph_job_validator.normalize_array_epoch_config reads latest when it is not set
        "arrayEpoch": {"modes": ["latest", *_names(m for m in V.VALID_ARRAY_EPOCH_MODES if m != "latest")], "default": "latest"},
        "textRule": {"targets": {"job": order(V.VALID_TEXT_TARGETS), "facet": order(F.VALID_FACET_TEXT_TARGETS),
                                 "diagram": list(MD.TARGETS)},
                     "fields": rule, "removed": dict(V.REMOVED_TEXT_TARGETS), "textTargets": {"diagram": list(MD.TEXT_TARGETS)}},
        # the planner's Download (model_diagram.py): its styles and the Scaled Sparkline's choices
        "diagram": {"styles": list(MD.STYLES), "orientations": list(MD.ORIENTATIONS), "legend": list(MD.LEGEND_POSITIONS),
                    "themes": fig_theme.names(), "theme": MD.DEFAULT_THEME,       # services/visualization/utils/fig_theme.py
                    # each theme's page, ink, muted and layer group colours: the planner's window previews the theme
                    "palettes": {t: {k: v for k, v in fig_theme.resolve(t, MD.GROUP_SLOT).items() if k != "panel"}
                                 for t in fig_theme.names()}},
        "export": {
            "job": {"formats": _names(V.VALID_EXPORT_FORMATS), "default": ["svg"],
                    "common": [{"k": "margin", "t": "num", "d": 0.05}, {"k": "save_raw_plot_data", "t": "bool", "d": False}],
                    "settings": {"png": [{"k": "dpi", "t": "int", "d": 300}],
                                 "svg": [{"k": "sanitize", "t": "bool", "d": True}, {"k": "validate_xml", "t": "bool", "d": True},
                                         {"k": "on_error", "t": "enum", "o": _names(V.VALID_SVG_ON_ERROR), "d": "warn"}],
                                 "pdf": [], "eps": []},
                    "moved": {"dpi": "moved to png.dpi", "pdf": "removed: a PDF is vector, it has no settings"}},
            "facet": {"formats": _names(F.VALID_FACET_FORMATS), "default": ["svg"],
                      "common": [{"k": "margin", "t": "num", "d": 0.05}],
                      "settings": {"png": [{"k": "dpi", "t": "int", "d": 300}], "pdf": [],
                                   "svg": [{"k": "sanitize", "t": "bool", "d": True}, {"k": "validate_xml", "t": "bool", "d": True},
                                           {"k": "on_error", "t": "enum", "o": _names(V.VALID_SVG_ON_ERROR), "d": "warn"}]},
                      "moved": {"dpi": "moved to png.dpi"}},
        },
        # a job's attack budget and test (graph_manager/trace_budget.py): empty, the run's curves and its latest All test
        "jobKeys": [{"k": "trace_budget", "t": "int", "d": None},
                    {"k": "test", "t": "enum", "o": ["latest", "budget", "recorded"], "d": "latest"}],
        "panel": [{"k": "label", "t": "text", "d": ""},
                  {"k": "label_position", "t": "enum", "o": list(F.VALID_LABEL_POSITIONS), "d": "bottom"}],
        "moved": {"job": {}, "facet": {"style": F.MOVED_KEYS["style"]}, "facetLegend": {"style": F.MOVED_KEYS["legend.style"]}},
    }


def _graph() -> list[Rule]:
    from services.graph_manager import faceting as F
    from services.graph_manager import graph_job_validator as V
    from services.graph_manager.explain import AGGREGATES, METHODS
    from services.graph_manager.plotters import POSITION_MAP

    by_source = {source: _names(processes) for source, processes in V.VALID_SOURCES.items()}
    return [
        {"p": "jobs.*.type", "o": _names(V.VALID_JOB_TYPES)},
        {"p": "jobs.*.mode", "o": _names(V.VALID_JOB_MODES)},
        {"p": "jobs.*.plot_type", "o": _names(V.VALID_PLOT_TYPES)},
        {"p": "jobs.*.source", "o": _ordered(V.VALID_SOURCES)},
        {"p": "jobs.*.processes.*", "o": _names(V.VALID_PROCESSES),
         "by": {"at": 2, "key": "source", "default": "attack_metrics", "map": by_source}},
        {"p": "jobs.*.processes.*.combine.mode", "o": _names(V.VALID_COMBINE_MODES)},
        {"p": "jobs.*.legend.mode", "o": _names(V.VALID_LEGEND_MODES)},
        {"p": "jobs.*.legend.position", "o": _ordered(POSITION_MAP)},
        {"p": "jobs.*.export.formats.*", "o": _names(V.VALID_EXPORT_FORMATS)},
        {"p": "jobs.*.export.svg.on_error", "o": _names(V.VALID_SVG_ON_ERROR)},
        {"p": "jobs.*.style.axes.x_scale", "o": _names(V.VALID_SCALES)},
        {"p": "jobs.*.style.axes.y_scale", "o": _names(V.VALID_SCALES)},
        {"p": "jobs.*.style.figure.orientation", "o": _names(V.VALID_ORIENTATIONS)},
        {"p": "jobs.*.style.line.styles.*", "o": _ordered(V.LINE_STYLES)},
        {"p": "jobs.*.style.text_rules.*.target", "o": _names(V.VALID_TEXT_TARGETS)},
        {"p": "jobs.*.style.font.family", "o": _fonts()},
        {"p": "jobs.*.style.text_rules.*.font.family", "o": _fonts()},
        {"p": "jobs.*.array_epoch.mode", "o": _names(V.VALID_ARRAY_EPOCH_MODES)},
        {"p": "jobs.*.array_epoch.summary.type", "o": _names(V.VALID_ARRAY_EPOCH_SUMMARIES)},
        {"p": "jobs.*.relative.rule", "o": _names(V.RELATIVE_RULES)},
        # groups: the band a group of runs is shaded with, stated (no default; graph_job_validator.GROUP_BANDS)
        {"p": "jobs.*.groups.*.band", "o": list(V.GROUP_BANDS)},
        {"p": "jobs.*.method", "o": _ordered(METHODS)},
        {"p": "jobs.*.aggregate", "o": _ordered(AGGREGATES)},
        {"p": "faceting.*.panels.*.orientation", "o": _names(F.VALID_ORIENTATIONS)},
        {"p": "faceting.*.arrangement.mode", "o": _names(F.VALID_ARRANGEMENTS)},
        {"p": "faceting.*.legend.position", "o": _names(F.VALID_LEGEND_POSITIONS)},
        {"p": "faceting.*.legend.mode", "o": _names(F.VALID_LEGEND_MODES)},
        {"p": "faceting.*.panels.*.label_position", "o": list(F.VALID_LABEL_POSITIONS)},
        {"p": "faceting.*.export.formats.*", "o": _names(F.VALID_FACET_FORMATS)},
        {"p": "faceting.*.export.svg.on_error", "o": _names(V.VALID_SVG_ON_ERROR)},
        {"p": "faceting.*.text_rules.*.target", "o": _names(F.VALID_FACET_TEXT_TARGETS)},
        {"p": "faceting.*.text_rules.*.font.family", "o": _fonts()},
    ]


def _graph_markers() -> list[Rule]:
    from matplotlib.lines import Line2D
    from services.graph_manager import metric_router as R

    markers = _names(str(m) for m in Line2D.markers if isinstance(m, str) and m.strip() and m not in {"None", "none"})
    metrics = set()
    for name in dir(R):
        value = getattr(R, name)
        if name.endswith("_METRICS") and isinstance(value, (set, frozenset, tuple, list)):
            metrics |= {str(v) for v in value}
    return [{"p": "jobs.*.style.line.marker", "o": ["none", *markers], "free": True},
            {"p": "jobs.*.legend.handle.0", "o": ["line", *markers], "free": True},
            {"p": "jobs.*.metric", "o": _names(metrics), "free": True},
            {"p": "jobs.*.metric.*", "o": _names(metrics), "free": True}]


def _auto_search() -> list[Rule]:
    from services.mlops.core.optuna_lite import SEARCH_METHODS
    from services.mlops.schemas.search import _ALLOWED_ENGINES

    return [{"p": "search.engine", "o": _names(_ALLOWED_ENGINES)}, {"p": "search.method", "o": _names(SEARCH_METHODS)}]


def _recorded_metrics() -> list[str]:
    """The metric names a run's rows hold, for the judge's menus: every metrics_to_log name as recorded (its
    canonical column; a metric with a value: the column of its default, e.g. top_5_accuracy, traces_to_ge_1),
    and the key-recovery columns the judge reads beside them (model_judge/run_reader.py)."""
    from sca.metrics import METRIC_OPTIONS, VALID_METRIC_PROCESSES, requested_column

    names = []
    for name in _names(VALID_METRIC_PROCESSES):
        try:
            column = requested_column(name, {METRIC_OPTIONS[name][0]: METRIC_OPTIONS[name][1]}) if name in METRIC_OPTIONS else requested_column(name)
        except Exception:      # noqa: BLE001 - a name the column rule refuses is listed as spelt
            column = name
        if column not in names:
            names.append(str(column))
    # the columns the judge's run reader adds beside the metrics (services/model_judge/run_reader.py)
    for extra in ("key_rank_reported", "key_enumeration_position", "selected_epoch", "fit_time_s", "mean_fit_time_s"):
        if extra not in names:
            names.append(extra)
    return names


def _filter_judge_test() -> dict[str, list[Rule]]:
    from services.model_filter.model_filter import MATCH_MODES, QUANTIFIERS
    from services.model_judge.model_judge import PROCESSES
    from sca.perform_attack import _ATTACK_DISTRIBUTIONS
    from utils.reproduce import ENVS

    processes = _ordered(PROCESSES)
    return {
        "model_filter.yaml": [
            {"p": "model_filter.match", "o": _ordered(MATCH_MODES)},
            {"p": "model_filter.env", "o": [*_ordered(ENVS), "any"]},
            {"p": "model_filter.where.*.quantifier", "o": _ordered(QUANTIFIERS)},
        ],
        # the file's keys sit under model_judge: (the rules name whole paths); a metric is one the runs record:
        # the metrics_to_log names and the columns their options make (free: a judge may name another column)
        "model_judge.yaml": [
            {"p": "model_judge.ranking.primary.process", "o": processes},
            {"p": "model_judge.ranking.tie_breakers.*.process", "o": processes},
            {"p": "model_judge.report_metrics.*.process", "o": processes},
            {"p": "model_judge.filters.require_processes.*", "o": processes},
            {"p": "model_judge.ranking.primary.metric", "o": _recorded_metrics(), "free": True},
            {"p": "model_judge.ranking.tie_breakers.*.metric", "o": _recorded_metrics(), "free": True},
            {"p": "model_judge.report_metrics.*.metric", "o": _recorded_metrics(), "free": True},
        ],
        "test_config.yaml": [
            {"p": "test_request.perform_attack.test.attack_distribution", "o": _names(_ATTACK_DISTRIBUTIONS)},
            {"p": "test_request.models.*.perform_attack.test.attack_distribution", "o": _names(_ATTACK_DISTRIBUTIONS)},
        ],
    }


def _master() -> list[Rule]:
    """master.yaml (MASTER CONTROL): default_project one of the projects; a project's ClearML scope names the
    profile it uploads with (clearml_tracker: profiles[scope]): global or local, another profile name typed."""
    return [
        {"p": "default_project", "ref": "projects"},
        {"p": "project.*.visualization.clearml.scope", "o": ["global", "local"], "free": True},
        *_timezone_rules(),
    ]


def _timezone_rules() -> list[Rule]:
    """master.yaml timezone: current_country a menu of the countries with zones (pytz, as
    utils/timestamps.resolve_timezone reads them), current_location the cities of that country's zones
    (the city part of each zone, the name the resolver matches: Asia/Taipei -> Taipei)."""
    import pytz

    cities: dict[str, list[str]] = {}
    for code, zones in pytz.country_timezones.items():
        name = pytz.country_names.get(code)
        if not name:
            continue
        # pytz lists the country's principal zone first: the city the panel picks when the country changes
        found = list(dict.fromkeys(z.rsplit("/", 1)[-1].replace("_", " ") for z in zones))
        cities[name] = cities[code] = found                       # the file may name the country or its ISO code
    names = sorted(pytz.country_names[c] for c in pytz.country_timezones if c in pytz.country_names)
    return [
        {"p": "timezone.current_country", "o": names},
        {"p": "timezone.current_location", "o": sorted({c for v in cities.values() for c in v}),
         "by": {"at": 1, "key": "current_country", "map": cities}},
    ]


_SOURCES: list[tuple[str, Callable[[], Any]]] = [
    ("params.yaml", _params), ("params.yaml", _params_text_sets), ("params.yaml", _keras_params),
    ("graph_config.yaml", _graph), ("graph_config.yaml", _graph_markers), ("auto_search.yaml", _auto_search),
    ("master.yaml", _master),
]


def static() -> dict[str, Any]:
    rules: dict[str, list[Rule]] = {}
    errors = []
    for file, source in _SOURCES:
        try:
            rules.setdefault(file, []).extend(source())
        except Exception as exc:   # noqa: BLE001 - a module that does not import leaves its keys as text
            errors.append(f"{file} {source.__name__}: {type(exc).__name__}: {exc}")
    try:
        for file, extra in _filter_judge_test().items():
            rules.setdefault(file, []).extend(extra)
    except Exception as exc:       # noqa: BLE001
        errors.append(f"model_filter / model_judge / test_config: {type(exc).__name__}: {exc}")
    # Keys naming the project's own things: the panel fills "ref" from `refs` (finished runs may
    # also be given by hash, so typing stays allowed).
    rules.setdefault("graph_config.yaml", []).append({"p": "jobs.*.models.*", "ref": "runs", "free": True})
    rules.setdefault("test_config.yaml", []).extend([{"p": "test_request.model", "ref": "runs", "free": True},
                                                     {"p": "test_request.model.*", "ref": "runs", "free": True}])
    for file in BOOLEAN_KEYS:
        rules.setdefault(file, []).extend(_booleans(file))
    rules.setdefault("master.yaml", []).append({"p": "default_project", "ref": "projects"})
    # params2.yaml shares params.yaml's keys (an older schema: a key it lacks has no field to draw).
    rules["params2.yaml"] = list(rules.get("params.yaml", []))
    try:
        graph = graph_forms()
    except Exception as exc:       # noqa: BLE001 - without it the graph forms are plain text boxes
        graph = None
        errors.append(f"graph_config.yaml graph_forms: {type(exc).__name__}: {exc}")
    try:
        models = model_forms()
    except Exception as exc:       # noqa: BLE001 - without it a model's layers are plain forms
        models = None
        errors.append(f"params.yaml model_forms: {type(exc).__name__}: {exc}")
    try:
        params = params_forms()
    except Exception as exc:       # noqa: BLE001 - without it params.yaml shows every block and a free monitor box
        params = None
        errors.append(f"params.yaml params_forms: {type(exc).__name__}: {exc}")
    if params is not None:
        params["models"] = models
    try:
        blueprint = blueprint_forms()
    except Exception as exc:       # noqa: BLE001 - without it the forms show only the keys a file holds
        blueprint = None
        errors.append(f"blueprint_forms: {type(exc).__name__}: {exc}")
    return {"rules": rules, "graph": graph, "params": params, "blueprint": blueprint, "errors": errors}


# ── project names ────────────────────────────────────────────────────────────

def _attack_bytes(data: Any) -> list[str]:
    """sca.attack_byte's options: the bytes some data entry of the project allows (dataset_types)."""
    by_entry = _attack_bytes_by_entry(data)
    allowed = sorted(set().union(*(set(b) for b in by_entry.values()))) if by_entry else list(range(16))
    return [str(b) for b in allowed]


def _attack_bytes_by_entry(data: Any) -> dict[str, list[int]]:
    """Each data entry's allowed bytes (dataset_types.attack_bytes_allowed: its type's target bytes within its file's
    window): the panel says which entries a chosen byte serves, and which bytes no entry defines."""
    try:
        from sca.dataset_types import attack_bytes_allowed
    except Exception:  # noqa: BLE001 - no sca module on the path: every byte
        return {}
    entries = [d for d in (data if isinstance(data, list) else []) if isinstance(d, dict) and d.get("name")]
    return {str(d["name"]): [int(b) for b in attack_bytes_allowed(d)] for d in entries}


def _attack_bytes_others(data: Any) -> dict[str, str]:
    """The bytes no data entry defines a target for: listed in the menu, never choosable, with why."""
    by_entry = _attack_bytes_by_entry(data)
    if not by_entry:
        return {}
    allowed = set().union(*(set(b) for b in by_entry.values()))
    types = {}
    for d in (data if isinstance(data, list) else []):
        if isinstance(d, dict) and d.get("name"):
            types.setdefault(str(d.get("type") or "ASCADf"), []).append(str(d["name"]))
    where = ", ".join(f"{t} ({len(names)} entr{'y' if len(names) == 1 else 'ies'})" for t, names in types.items())
    return {str(b): f"not defined for any data entry of this project: {where}" for b in range(16) if b not in allowed}


def refs(root: Path) -> dict[str, Any]:
    """Per project: data entry names, model ids, finished runs, queue keys (no torch)."""
    import yaml

    out: dict[str, Any] = {}
    for config in sorted((root / "projects").glob("*/config")):
        project = config.parent
        def load(name: str) -> Any:
            try:
                return yaml.safe_load((config / name).read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                return {}
        params = load("params.yaml")
        data = params.get("data") or []
        try:
            done = yaml.safe_load((project / "artifacts" / "done_model_names.yaml").read_text(encoding="utf-8")) or []
        except (OSError, yaml.YAMLError):
            done = []
        out[project.name] = {
            "attack_bytes": _attack_bytes(data),
            "attack_bytes_others": _attack_bytes_others(data),
            "attack_bytes_by_entry": _attack_bytes_by_entry(data),
            "datasets": [str(d.get("name")) for d in data if isinstance(d, dict) and d.get("name")],
            "model_ids": [str(m.get("model_id")) for m in params.get("model") or [] if isinstance(m, dict) and m.get("model_id")],
            "runs": [str(r) for r in done if isinstance(r, (str, int))],
            "queues": [str(k) for k in (load("experiment_queue.yaml") or {})],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Options of config keys for the panel's menus.")
    parser.add_argument("command", choices=["static", "refs"])
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    out = static() if args.command == "static" else refs(Path(args.root).resolve())
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
