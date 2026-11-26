# services/side_channel_core/utils/config_schema.py
"""
The keys TraceFlow reads from each configuration file, and the check that refuses any other.

A configuration key the code does not read changes nothing, silently: a misspelling
(`trainer.epoch`) or a key that was renamed or removed. check_keys() refuses it with a
ConfigError that names the file, the key path and the closest known key:

    ConfigError: params.yaml: trainer.epoch is not a key TraceFlow reads; did you mean
    trainer.epochs? Keys here: batch_size, early_stopping, epochs, ...

A schema is nested mappings of the keys a file may hold:
    {key: schema}   a mapping with exactly these keys (each optional)
    ANY             any value: a block whose keys are names or a function's arguments, or one
                    that its own reader checks (data entries: sca/dataset_types.py; model
                    architecture_args; optimizer and scheduler fields: utils/optimizer_util.py)
    Seq(schema)     a list of items (one item alone is accepted, as the readers accept it)
    Named(schema)   a mapping whose keys are names (optimizer types, ClearML profiles)

The key lists were read from the code that reads each file (2026-10-18), not from one project:
keys the templates show only as comments (sca.fixed_key_hex, trainer.patience, ...) are in,
and so are renamed or obsolete keys whose own reader refuses them with the specific reason
(perform_attack.attack_repetitions, trainer.lr, a facet's style), so that reason still shows.

Checked: the project's params.yaml, its master.yaml entry and the top of master.yaml
(utils/config_loader.get_project_config), graph_config.yaml (extra_configs.graph),
model_judge.yaml (services/model_judge), auto_search.yaml and import_config.yaml
(utils/model_import.py). test_config.yaml and
model_filter.yaml are checked by their readers (src/sca/test_request.py,
services/model_filter); the ENV files by config_loader.apply_env_overrides. Run snapshots
(params_out) are never checked here: they are read directly, and they hold resolved keys.

Author: Shraddha Mishra
"""

from __future__ import annotations

import difflib
from typing import Any, Mapping

from services.side_channel_core.utils.config_errors import ConfigError

ANY = object()


class Seq:
    """A list whose items follow `item`."""

    def __init__(self, item: Any):
        self.item = item


class Named:
    """A mapping whose keys are names (free) and whose values follow `item`."""

    def __init__(self, item: Any):
        self.item = item


def _keys(*names: str) -> dict:
    return {name: ANY for name in names}


# ── params.yaml ─────────────────────────────────────────────────────────────
_ATTACK_PROCESS = {
    **_keys("attack_times", "shuffle", "attack_distribution", "attack_trace_budget"),
    # renamed or removed; src/sca/perform_attack.py refuses them with the new name
    **_keys("attack_repetitions", "attack_shuffle", "sampling_mode", "budget_exceed_policy",
            "attack_budget_overflow_policy", "allow_budget_clipping", "require_independent_final_subsets",
            "require_distinct_final_subsets"),
}
_FRAMEWORK_BLOCK = {"weight_init": ANY, "weight_init_args": ANY, "bias_init": ANY, "bias_init_args": ANY,
                    "optimizer_args": Named(ANY),
                    "grad_clip_norm": ANY}            # the framework's gradient clipping; folded to trainer.grad_clip_norm by the loader
PARAMS: dict = {
    "deep_learning": {"framework": ANY, "pytorch": _FRAMEWORK_BLOCK, "keras": _FRAMEWORK_BLOCK},
    "seed": ANY,
    "device": ANY,
    "num_workers": _keys("train", "val"),
    "monitor_interval": ANY,
    "dataloader": _keys("pin_memory", "persistent_workers", "prefetch_factor", "drop_last_train"),
    "sca": {
        # rank_base: no longer a setting (TraceFlow ranks from 1); accepted so the snapshots of runs made
        # with it load and reproduce (rank_curve still refuses any value but 1).
        # variant, variable_key, variable_key_aggregation, fixed_key_hex, require_metadata (owner 2026-10-22,
        # RETIRED): the key mode is the data entry's type (sca/dataset_types.py: every registered type is
        # fixed_key with the key in its database), so none of them is a setting; accepted for the snapshots
        # of runs made before types, and a configured variant must agree with the type
        **_keys("attack_byte", "leakage_model", "rank_base", "fixed_key_hex", "require_metadata",
                "variable_key", "variable_key_aggregation", "variant"),
        "candidate_ranking": _keys("mode", "top_k"),
        # sca/key_enumeration.py (MODE_PROFILE_FIELDS): the chosen mode's block is read; comparison runs every mode
        "key_enumeration": {**_keys("enabled", "mode", "stop_on_success"),
                            "modes": {"local_hamming": _keys("top_k", "hamming_radius", "local_budget"),
                                      "neighbour_sweep": _keys("hamming_radius"),
                                      "support_weighted": _keys("top_k", "hamming_radius", "local_budget")}},
    },
    "evaluation": {"validation": _keys("checkpointing",
                                       "judge_log"),      # utils/validation_judge.py: a log line per epoch, not hashed
                   "test": _keys("load_model"),
                   "test_checkpoint_policy": ANY},       # older name of test.load_model, still read
    "cross_validation": {**_keys("enabled", "n_folds", "strategy", "shuffle", "report_aggregate", "partition_seed"),
                         "refit": _keys("enabled")},      # utils/cross_validation.refit_enabled
    "perform_attack": {**_ATTACK_PROCESS, "enabled": ANY, "attack_step": ANY, "ge_thresholds": ANY,
                       "validation": _ATTACK_PROCESS, "test": _ATTACK_PROCESS},
    "validation": _keys("rank_eval_interval", "rank_step"),
    "key_recovery_reporting": _keys("enabled", "log_validation", "log_test", "save_validation_reports",
                                    "save_test_report", "recovery_rank", "require_stable_recovery",
                                    "top_k_candidates", "include_score_margin", "key_display"),
    "data": Seq(ANY),                    # every entry: sca/dataset_types.validate_entry, by its type
    "metrics_cfg": _keys("ce_eps", "rank_log_eps", "preds_are_probs", "lm_enabled", "strict_prob_check",
                         "prob_atol", "prob_rtol"),
    "model": Seq({"model_id": ANY, "architecture": ANY, "architecture_args": ANY}),
    "trainer": {
        # early_stopping: the block {enabled, monitor_metric, patience, min_delta} params.yaml writes (2026-10-22), or
        # the flat true / false with monitor_metric, patience, min_delta beside it (a snapshot, an older file):
        # config_loader.canonical_trainer folds the block into the flat keys the code reads and checks its keys
        **_keys("early_stopping", "monitor_metric", "monitor_mode", "patience", "min_delta", "epochs", "batch_size",
                "precision", "grad_clip_norm",
                "checkpointing"),            # older name of evaluation.validation.checkpointing, still read
        # obsolete flat fields: src/core/model_contract_resolver.py refuses them with the nested names
        **_keys("lr", "learning_rate", "weight_decay", "optimizer_args", "loss_reduction", "scheduler"),
        "loss": {"type": ANY, "args": ANY},
        "optimizer": Seq(ANY),               # its fields: utils/optimizer_util.py
        "evaluation": {"output_transformation": {"type": ANY, "args": ANY}},   # args: the transformation's own (Softmax: dim)
    },
    "metrics_to_log": ANY,
    "metrics": ANY,                          # older name of metrics_to_log, still read
    "paths": _keys("artifacts_root", "data_out", "test_data", "model_out", "logs", "model_graphs", "checkpoints",
                   "params_out", "device", "data", "metrics", "cross_validation", "key_recovery", "test",
                   "generated_models", "tmux_logs", "params", "graphs", "log_dir", "validation_best", "model_preview"),
}

# ── master.yaml ─────────────────────────────────────────────────────────────
_CLEARML_PROFILE = _keys("web_url", "api_url", "files_url", "verify_certificate", "access_key_name",
                         "secret_key_name", "clearml_env_file", "healthcheck_enabled", "healthcheck_path",
                         "healthcheck_timeout_sec")
MASTER_PROJECT: dict = {
    **_keys("name", "description", "root", "params", "print_config", "use_core_src", "code_dir", "logs", "models",
            "metrics"),
    "auto_search": _keys("enabled", "config", "hard_replace_params"),
    "auto_mlops": _keys("enabled", "config", "hard_replace_params"),
    "exp_rerun": _keys("path", "repeat"),
    "scripts": Named(ANY),
    "env_configs": Named(ANY),
    "extra_configs": Named(ANY),
    "distributed_override": _keys("use", "num_gpus"),
    "visualization": {
        "clearml": {"enabled": ANY, "scope": ANY, "fail_fast_on_unreachable_api": ANY,
                    "profiles": Named(_CLEARML_PROFILE)},
        # graph_manager.ui and visualization.ui (dashboards) are no longer keys (2026-10-19): the
        # TraceFlow Monitor is TraceFlow's UI
        "graph_manager": {"enabled": ANY, "config": ANY},
    },
}
MASTER: dict = {
    "secrets": _keys("registry_env_file", "db_env_file"),
    "distributed": _keys("enabled", "backend", "ddp_script"),
    "project": ANY,                          # each entry: MASTER_PROJECT (the selected one is checked)
    "default_project": ANY,
    "timezone": _keys("current_location", "current_country"),
    "status_share": _keys("enabled", "host", "port"),   # services/visualization/server/status.py (optional)
    "data_fetch": _keys("on_mismatch"),                 # utils/dataset_fetch.py (optional)
}

# ── graph_config.yaml ───────────────────────────────────────────────────────
_FONT = _keys("family", "size")
_TEXT_RULE = {**_keys("target", "text", "bold", "italic", "color", "words", "size", "family"), "font": _FONT}
_STYLE = {
    "figure": {**_keys("orientation", "width", "height"), "margins": _keys("top", "bottom", "left", "right")},
    "font": _keys("family", "base_size", "title_size", "label_size", "tick_size", "legend_size"),
    "line": {**_keys("width", "marker", "marker_size", "linewidth"), "styles": Named(ANY)},   # styles: per entry (run) a line style
    "series": Named(_keys("color", "hatch", "width", "dashes")),  # graph_job_validator._normalize_series_style: per series {color, hatch}
    "colormap": ANY,                         # graph_job_validator._normalize_colormap: a heatmap's colour scale
    "bar_values": ANY,                       # graph_job_validator._normalize_bar_values: a bar's value at its end
    "heatmap": _keys("limits", "colorbar", "cell_values", "row_groups"),   # graph_job_validator._normalize_heatmap
    "grid": _keys("enabled", "alpha"),
    "std_band": _keys("enabled", "alpha"),
    "axes": _keys("x_scale", "y_scale", "x_limits", "y_limits", "x_step", "y_step", "x_grid", "x_grid_first",
                  "x_margin", "y_margin"),
    "text_rules": Seq(_TEXT_RULE),
}                                            # graph_manager/plot_style.py DEFAULT_STYLE
_EXPORT = {"formats": ANY, "margin": ANY, "save_raw_plot_data": ANY, "png": _keys("dpi"),
           "svg": _keys("sanitize", "validate_xml", "on_error"),
           "dpi": ANY, "pdf": ANY, "eps": ANY}  # moved or settings-free; the validators name the reason
_JOB = {
    **_keys("id", "name", "type", "mode", "enabled", "source", "metric", "plot_type", "models", "top_models",
            "processes", "process", "output_file_name", "output_name", "title", "metrics",
            "method", "aggregate", "traces", "layers",            # model_explanation jobs
            "combine", "combines", "active_context",
            "folder"),                       # job_paths.job_dir: run (a one-run job in its run's folder) | project
    "array_epoch": {**_keys("mode", "epoch", "epochs"), "summary": _keys("type", "threshold", "trace_index")},
    "relative": ANY,                         # graph_job_validator.py checks rule / where / label
    "metric_process": ANY,                   # graph_job_validator.normalize_metric_process: {metric: process}
    "groups": ANY,                           # graph_job_validator.py checks label / models / band
    "marks": ANY,                            # graph_job_validator.normalize_marks_config: points of the rank curves
    "trace_budget": ANY,                     # graph_manager/trace_budget.py: the attack curves at B traces per attack
    "test": ANY,                             # graph_manager/trace_budget.py: recorded, or an external test's index
    "overlays": {**_keys("ge_averages", "ntge", "trace_lines"), "colors": _keys("geometric", "median", "ntge")},   # graph_job_validator.normalize_overlays_config
    "export": _EXPORT,
    "legend": {**_keys("enabled", "position", "handle", "mode", "columns", "frame", "title", "anchor", "size",
                       "rule"), "labels": ANY,
               "entries": Seq(_keys("label", "color", "hatch", "linestyle", "width", "dashes"))},   # graph_job_validator._normalize_legend_entries
    "style": _STYLE,
}
_FACET = {
    **_keys("id", "name", "enabled", "output_file_name", "panel_height", "spacing"),
    "panels": Seq({**_keys("file_id", "address", "orientation", "label", "label_position")}),
    "arrangement": _keys("mode", "order", "rows", "columns", "fit"),
    "legend": {**_keys("enabled", "position", "columns", "handle", "mode", "title", "caption", "frame", "style"),
               "entries": Seq({**_keys("label", "color", "marker", "shape", "size", "style")}),
               "colormap": _keys("name", "limits", "label")},    # faceting._validate_legend: one spectrum for all
    "export": _EXPORT,
    "text_rules": Seq(_TEXT_RULE),
    "style": ANY,                            # moved to text_rules; faceting.py names the new key
}
GRAPH: dict = {"jobs": Seq(_JOB), "faceting": Seq(_FACET)}

# ── model_judge.yaml ────────────────────────────────────────────────────────
_PROCESS_METRIC = _keys("process", "metric")
JUDGE: dict = {
    "model_judge": {
        **_keys("output_dir", "decision"),
        "ranking": {"N_top_candidates": ANY, "primary": _PROCESS_METRIC, "tie_breakers": Seq(_PROCESS_METRIC)},
        "report_metrics": Seq(_PROCESS_METRIC),
        "filters": _keys("require_done", "require_processes"),
        "outputs": _keys("selected_model", "latest_selection", "leaderboard", "selected_candidates"),
    },
}

# ── auto_search.yaml ────────────────────────────────────────────────────────
# Its own search block, and params.yaml keys it overlays on the run (config_loader
# _apply_auto_search_overrides).
AUTO_SEARCH: dict = {
    **PARAMS,
    "search": {**_keys("engine", "method"), "budget": _keys("max_trials", "parallel_trials"),
               "objective": _keys("metric", "direction")},
    "tracking": {"clearml": _keys("enabled")},
}

# ── import_config.yaml (utils/model_import.py) ──────────────────────────────
# The settings of an import (defaults, and each model's own over them). perform_attack takes the keys an
# external test may change, checked by src/sca/test_request.py (_check_attack_block); trace_normalization is
# a data entry's, checked by its reader (sca/ascad_data_loader.py); model is a params.yaml model entry.
_IMPORT_SETTINGS = {
    **_keys("data", "trace_normalization", "fit_on", "seed", "perform_attack"),
    "sca": _keys("leakage_model", "attack_byte"),
}
IMPORT: dict = {
    "model_import": {
        "defaults": _IMPORT_SETTINGS,
        # one file: file (+ sha256); several: files, each file beside its own SHA-256 (one run each)
        "models": Seq({**_keys("name", "file", "sha256", "source", "format", "model"),
                       "files": Seq(_keys("file", "sha256")), **_IMPORT_SETTINGS}),
    },
}


# ── what a new project is offered ───────────────────────────────────────────
# Keys the readers still accept (an older file, a run's snapshot) but a new project is not offered: renamed
# or retired keys, and keys no code reads any more. The blueprint (config/blueprint/) holds every key of a
# file's schema but these, so a key added to a schema above reaches every project's panel form only once it
# is in the blueprint too: `python -m services.visualization.vscode_monitor.make_blueprint sync` adds it
# there, empty, and test_config_schema refuses a blueprint that lags behind (guide 10).
_RETIRED_ATTACK = {"attack_repetitions", "attack_shuffle", "sampling_mode", "budget_exceed_policy",
                   "attack_budget_overflow_policy", "allow_budget_clipping", "require_independent_final_subsets",
                   "require_distinct_final_subsets"}
RETIRED: dict[str, set[str]] = {
    "params.yaml": {
        "sca.rank_base", "evaluation.test_checkpoint_policy", "trainer.checkpointing", "trainer.lr",
        # the key mode and its overrides: the data entry's type gives it (owner 2026-10-22)
        "sca.variant", "sca.variable_key", "sca.variable_key_aggregation", "sca.fixed_key_hex", "sca.require_metadata",
        # the flat early-stopping keys: written as the trainer.early_stopping block since 2026-10-22 (folded back by the
        # loader); monitor_mode retired: the direction follows the metric
        "trainer.monitor_metric", "trainer.patience", "trainer.min_delta", "trainer.monitor_mode",
        "trainer.grad_clip_norm",    # written under deep_learning.<framework> since 2026-10-22 (folded back by the loader)
        "trainer.learning_rate", "trainer.weight_decay", "trainer.optimizer_args", "trainer.loss_reduction",
        "trainer.scheduler", "metrics", "paths.log_dir", "paths.validation_best",
        # the flat attack settings: read per process (validation, test) since 2026-09; the flat ones are
        # accepted for older snapshots only
        "perform_attack.attack_times", "perform_attack.shuffle", "perform_attack.attack_distribution",
        *{f"perform_attack{where}.{key}" for where in ("", ".validation", ".test") for key in _RETIRED_ATTACK},
        # the budget of one attack: the flat key (April 2026), then the per-process ones (2026-10-26) retired; it is
        # the data entry's since 2026-10-27 (data[].attack_trace_budget; MOVED refuses one with a value)
        "perform_attack.attack_trace_budget", "perform_attack.validation.attack_trace_budget",
        "perform_attack.test.attack_trace_budget",
    },
    "model_judge.yaml": set(),
    "auto_search.yaml": set(),
    "master_project.yaml": set(),
}


def offered_keys(schema: Any, path: str = "") -> set[str]:
    """Every key path a schema lists (dotted; a list's items as `*`), down to the blocks whose keys are
    names or arguments (ANY, Named): what a file following the schema may hold."""
    out: set[str] = set()
    if isinstance(schema, Seq):
        return offered_keys(schema.item, f"{path}.*" if path else "*")
    if schema is ANY or isinstance(schema, Named) or not isinstance(schema, Mapping):
        return out
    for key, sub in schema.items():
        where = f"{path}.{key}" if path else str(key)
        out.add(where)
        out |= offered_keys(sub, where)
    return out


def check_keys(data: Any, schema: Any, *, source: str, path: str = "") -> None:
    """Refuse every key of `data` that `schema` does not list (module docstring); `source` names the file."""
    if schema is ANY or data is None:
        return
    if isinstance(schema, Seq):
        items = data if isinstance(data, list) else [data]
        for index, item in enumerate(items):
            check_keys(item, schema.item, source=source, path=f"{path}[{index}]" if isinstance(data, list) else path)
        return
    if not isinstance(data, Mapping):
        raise ConfigError(f"{source}: {path or 'the file'} must be a mapping, got {type(data).__name__}")
    if isinstance(schema, Named):
        for name, value in data.items():
            check_keys(value, schema.item, source=source, path=f"{path}.{name}" if path else str(name))
        return
    for key, value in data.items():
        where = f"{path}.{key}" if path else str(key)
        if key not in schema:
            close = difflib.get_close_matches(str(key), [str(k) for k in schema], n=1)
            hint = f"; did you mean {path + '.' if path else ''}{close[0]}?" if close else "."
            raise ConfigError(
                f"{source}: {where} is not a key TraceFlow reads{hint} "
                f"Keys here: {', '.join(sorted(str(k) for k in schema))}"
            )
        check_keys(value, schema[key], source=source, path=where)


# Keys moved to another place (owner 2026-10-27): accepted empty (a file the panel completed), refused with a value.
MOVED = {
    "perform_attack.validation.attack_trace_budget": "data[].attack_trace_budget (the traces of one test attack, a "
                                                     "data entry's; a Budget test attacks with it: make test BUDGET=1)",
    "perform_attack.test.attack_trace_budget": "data[].attack_trace_budget (the traces of one test attack, a "
                                               "data entry's; a Budget test attacks with it: make test BUDGET=1)",
}


def refuse_moved(data: Any, *, source: str) -> None:
    """Refuse a MOVED key that has a value, naming where it went."""
    for dotted, where in MOVED.items():
        node = data
        for key in dotted.split("."):
            node = node.get(key) if isinstance(node, Mapping) else None
        if node is not None:
            raise ConfigError(f"{source}: {dotted} moved to {where}")


__all__ = ["ANY", "AUTO_SEARCH", "GRAPH", "IMPORT", "JUDGE", "MASTER", "MASTER_PROJECT", "PARAMS", "RETIRED", "Named", "Seq",
           "ConfigError", "MOVED", "check_keys", "offered_keys", "refuse_moved"]
