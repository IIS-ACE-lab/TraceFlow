# services/graph_manager/graph_job_validator.py
"""
Graph job validator for TraceFlow GraphManager.

Validates the job-driven graph_config.yaml contract used by GraphManager.

Current schema direction:
- output_file_name is the official output stem field.
- output_name is accepted only as backward-compatible fallback.
- plot_type is job-level.
- model_overlay uses models + processes directly.
- text_rules entries use `text`, not `words`.

Missing metric behavior is handled later by metric_loader / graph_manager:
- issue terminal warning
- continue safely

Author: Shraddha Mishra
"""

from __future__ import annotations

from services.side_channel_core.utils.config_errors import ConfigError
import difflib
from typing import Any, Mapping

from .plot_style import parse_legend_handle
from .metric_router import plot_types_for, route_metric


VALID_JOB_TYPES = {"solo", "process_overlay", "model_overlay", "model_explanation"}
VALID_JOB_MODES = {"auto"}

VALID_PLOT_TYPES = {
    "auto",
    "line",
    "scatter",
    "line_scatter",
    "step",
    "bar",
    "barh",
    "histogram",
    "distribution",
    "heatmap",
    "strip",
}

VALID_EXPORT_FORMATS = {"png", "svg", "pdf", "eps"}
FORBIDDEN_EXPORT_FORMATS = {"jpg", "jpeg"}

VALID_SVG_ON_ERROR = {"warn", "fail"}
VALID_LEGEND_MODES = {"auto", "custom"}
VALID_SCALES = {"linear", "log", "symlog", "logit"}
VALID_ORIENTATIONS = {"landscape", "portrait"}
VALID_COMBINE_MODES = {"sum", "mean"}
# Joins the processes of a combined series, e.g. "train+validation".
COMBINE_SEPARATOR = "+"

PROCESS_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "validate": "validation",
}

VALID_PROCESSES = {"train", "validation", "test"}

# Metrics directory -> processes it records.
VALID_SOURCES = {
    "attack_metrics": {"train", "validation", "test"},
    "cross_validation": {"train", "validation"},
    "key_recovery": {"validation", "test"},
}

# One figure per model: these cannot be overlaid across models/processes.
SOLO_ONLY_METRICS = {"fold_rank_curves", "key_candidate_scores"}

# Job 0 is the only automated job: the model judge rewrites its top_models.
DEFAULT_JOB_ID = 0

VALID_TEXT_TARGETS = {
    "title",
    "x_label",
    "y_label",
    "legend_text",
    "colorbar_label",   # a heat map's colour-bar label
}
# Removed targets: what each did, for the error that names the replacement.
REMOVED_TEXT_TARGETS = {
    "legend": "legend_text (both styled the legend's texts)",
    "plot_labels": "nothing (no text was ever styled by it)",
}

VALID_ARRAY_EPOCH_MODES = {"latest", "selected", "overlay", "summary"}
VALID_ARRAY_EPOCH_SUMMARIES = {
    "final_value",
    "value_at_trace",
    "traces_to_threshold",
    "min_value",
}


class GraphJobValidationError(ConfigError):     # a configuration error (utils/config_errors.py), a ValueError as before
    """Raised when a graph job is malformed."""


def normalize_process_name(process: str) -> str:
    p = str(process).strip().lower()
    return PROCESS_ALIASES.get(p, p)


def normalize_job_mode(job: Mapping[str, Any]) -> str:
    mode = str(job.get("mode", "auto")).strip().lower()

    if mode not in VALID_JOB_MODES:
        raise GraphJobValidationError(
            f"job id={job.get('id')} invalid mode={mode!r}. "
            f"Allowed: {sorted(VALID_JOB_MODES)}"
        )

    return mode


def normalize_plot_type(job: Mapping[str, Any]) -> str:
    plot_type = str(job.get("plot_type", "auto")).strip().lower()

    if plot_type not in VALID_PLOT_TYPES:
        raise GraphJobValidationError(
            f"job id={job.get('id')} invalid plot_type={plot_type!r}. "
            f"Allowed: {sorted(VALID_PLOT_TYPES)}"
        )

    return plot_type


def _require_keys(job: Mapping[str, Any], keys: list[str]) -> None:
    missing = [k for k in keys if k not in job]
    if missing:
        raise GraphJobValidationError(
            f"graph job id={job.get('id', '<unknown>')} missing required keys: {missing}"
        )


def _require_non_empty_list(job: Mapping[str, Any], key: str) -> list[Any]:
    value = job.get(key)

    if not isinstance(value, list) or not value:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} requires non-empty list field: {key}"
        )

    return value


def _validate_processes(job: Mapping[str, Any], processes: list[Any]) -> None:
    source = str(job.get("source", "attack_metrics"))
    allowed = VALID_SOURCES.get(source, VALID_PROCESSES)
    normalized = [normalize_process_name(str(p)) for p in processes]
    invalid = [p for p in normalized if p not in allowed]

    if any(p in VALID_SOURCES for p in invalid):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} lists {[p for p in invalid if p in VALID_SOURCES]} "
            "under processes, but that is a source. source says which records are read, "
            "processes which stage of them:\n"
            "    source: cross_validation + processes: [validation]  held-out fold traces, mean over folds\n"
            "    source: cross_validation + processes: [train]       fold training, mean over folds\n"
            "    source: attack_metrics   + processes: [train]       the final training\n"
            "    source: attack_metrics   + processes: [test]        the attack traces"
        )

    if invalid:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} has invalid processes={invalid} "
            f"for source={source!r}. Allowed: {sorted(allowed)}"
        )


def _normalize_one_metric(metric: Any) -> str:
    """The config spelling fit_time_{s} as the recorded column fit_time_s."""
    metric = str(metric if metric is not None else "").strip()
    plain = metric.lower().replace("{", "").replace("}", "")
    if plain.startswith(("fit_time", "mean_fit_time")):
        return plain + "_s" if plain in ("fit_time", "mean_fit_time") else plain
    return metric


def check_heatmap_job(job: Mapping[str, Any]) -> None:
    """
    plot_type heatmap (owner 2026-10-27, Fig. 20 c): a model per row and a cross-validation fold per column, each
    cell one scalar of that fold's key-recovery report (GraphManager.run_fold_heatmap). So: a model_overlay of one
    metric, source key_recovery, process validation (a fold's held-out traces).
    """
    where = f"graph job id={job.get('id')} plot_type heatmap"
    if str(job.get("type")) != "model_overlay":
        raise GraphJobValidationError(f"{where} is a model_overlay (a model per row)")
    if job.get("source") != "key_recovery":
        raise GraphJobValidationError(f"{where} reads the folds' key-recovery reports: source key_recovery")
    if len(job.get("metrics") or []) != 1:
        raise GraphJobValidationError(f"{where} draws one metric, got {job.get('metrics')}")
    if [normalize_process_name(p) for p in job.get("processes") or [] if isinstance(p, str)] != ["validation"]:
        raise GraphJobValidationError(f"{where}: processes [validation] (the folds' held-out traces)")


def check_strip_job(job: Mapping[str, Any]) -> None:
    """
    plot_type strip (TODO.md section 29, the Model Performance Report's key confidence): a dot per run at
    log10(1 + the score margin of its best key), teal when that key is the true key, red when not; a row per group
    (groups), else one row of all the job's runs (plotters.plot_strip). So: a model_overlay of
    top1_top2_score_margin, source key_recovery, process test (metrics/key_recovery/test.json).
    """
    where = f"graph job id={job.get('id')} plot_type strip"
    if str(job.get("type")) != "model_overlay":
        raise GraphJobValidationError(f"{where} is a model_overlay (a dot per run)")
    if job.get("source") != "key_recovery":
        raise GraphJobValidationError(f"{where} reads the runs' key-recovery reports: source key_recovery")
    if list(job.get("metrics") or []) != ["top1_top2_score_margin"]:
        raise GraphJobValidationError(f"{where} draws the best key's score margin: metric top1_top2_score_margin, "
                                      f"got {job.get('metrics')}")
    if [normalize_process_name(p) for p in job.get("processes") or [] if isinstance(p, str)] != ["test"]:
        raise GraphJobValidationError(f"{where}: processes [test] (the test's key-recovery report)")


def normalize_metric_process(job: Mapping[str, Any], combines: Mapping[str, Any]) -> dict[str, str] | None:
    """
    metric_process: {metric: process}, the process a metric of a model_overlay is read from instead of the job's
    (owner 2026-10-27: the validation accuracy beside the training fit time in one figure). Each key is a metric of
    the job, each value train | validation | test; the job then lists one process, the one of the metrics the map
    does not name. Empty or absent: every metric is read from the job's processes.
    """
    raw = job.get("metric_process")
    if raw in (None, "", {}):
        return None
    where = f"graph job id={job.get('id')} metric_process"
    if not isinstance(raw, Mapping):
        raise GraphJobValidationError(f"{where} is a mapping {{metric: process}}, got {raw!r}")
    if str(job.get("type", "")).strip() != "model_overlay":
        raise GraphJobValidationError(f"{where} is for a model_overlay")
    if len(job.get("processes") or []) != 1 or combines:
        raise GraphJobValidationError(
            f"{where}: processes lists one process (the one of the metrics the map does not name), "
            f"got {list(job.get('processes') or []) + list(combines)}")
    out: dict[str, str] = {}
    for key, value in raw.items():
        name = _normalize_one_metric(key)
        if name not in job["metrics"]:
            raise GraphJobValidationError(f"{where}: {key!r} is not a metric of the job ({job['metrics']})")
        process = normalize_process_name(value)
        if process not in VALID_PROCESSES:
            raise GraphJobValidationError(f"{where}[{key!r}]: {value!r} is not one of {sorted(VALID_PROCESSES)}")
        out[name] = process
    return out


LINE_STYLES = {"-", "--", ":", "-.", "solid", "dashed", "dotted", "dashdot"}
HATCH_CHARS = set("/\\|-+xoO.*")                     # Matplotlib's hatch patterns; "" is solid


def _colour(value: Any, where: str) -> str:
    from matplotlib.colors import is_color_like

    text = str(value).strip()
    if not is_color_like(text):
        raise GraphJobValidationError(f"{where}={value!r} is not a Matplotlib colour (e.g. \"C0\", \"#1f77b4\")")
    return text


def _hatch(value: Any, where: str) -> str:
    text = "" if value is None else str(value).strip()
    if set(text) - HATCH_CHARS:
        raise GraphJobValidationError(f"{where}={value!r}: a hatch is made of {''.join(sorted(HATCH_CHARS))}")
    return text


def _line_look(spec: Mapping[str, Any], where: str) -> dict[str, Any]:
    """width (pt, > 0) and dashes ([on, off, ...] pt, each > 0) of a line, as given."""
    out: dict[str, Any] = {}
    if spec.get("width") not in (None, ""):
        width = _normalize_float(spec["width"], default=-1.0)
        if width <= 0:
            raise GraphJobValidationError(f"{where}.width={spec['width']!r} is a line width in pt, > 0")
        out["width"] = width
    if spec.get("dashes") not in (None, "", []):
        dashes = spec["dashes"]
        if (not isinstance(dashes, list) or len(dashes) < 2 or len(dashes) % 2
                or not all(isinstance(d, (int, float)) and not isinstance(d, bool) and d > 0 for d in dashes)):
            raise GraphJobValidationError(f"{where}.dashes={dashes!r} is [on, off, ...]: pairs of lengths in pt, > 0")
        out["dashes"] = [float(d) for d in dashes]
    return out


def _normalize_series_style(raw: Any, job: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """
    style.series: a series ("<model_name>", or "<model_name>:<metric>" in a job of several metrics) ->
    {color, hatch, width, dashes}: its colour instead of its slot's, the hatch of its bars ("" solid), and its
    line's width (pt) and dash pattern ([on, off] pt). Owner 2026-10-27: Fig. 10's accuracy of both
    desynchronizations in one panel, a hatch per N_max; Fig. 26's HW curves in longer, thicker dashes.
    """
    if raw in (None, "", {}):
        return {}
    where = f"graph job id={job.get('id')} style.series"
    if not isinstance(raw, Mapping):
        raise GraphJobValidationError(f"{where} must be a mapping {{series: {{color, hatch}}}}")
    out: dict[str, dict[str, str]] = {}
    for key, spec in raw.items():
        if not isinstance(spec, Mapping) or set(spec) - {"color", "hatch", "width", "dashes"}:
            raise GraphJobValidationError(f"{where}[{key!r}] takes color, hatch, width and dashes, got {spec!r}")
        out[str(key).strip()] = {
            **({"color": _colour(spec["color"], f"{where}[{key!r}].color")} if spec.get("color") not in (None, "") else {}),
            **({"hatch": _hatch(spec.get("hatch"), f"{where}[{key!r}].hatch")} if "hatch" in spec else {}),
            **_line_look(spec, f"{where}[{key!r}]"),
        }
    return out


def _normalize_bar_values(raw: Any, job: Mapping[str, Any]) -> str | None:
    """style.bar_values: a Python format spec (".4f", ".1%", ",.0f") for the value printed at each bar's end;
    empty or false: no values."""
    if raw in (None, "", False):
        return None
    spec = str(raw).strip()
    try:
        format(0.00532, spec)
    except (ValueError, TypeError):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.bar_values={raw!r} is not a number format (e.g. \".4f\")") from None
    return spec


def _normalize_figure_margins(raw: Any, job: Mapping[str, Any]) -> dict[str, float] | None:
    """
    style.figure.margins: {top, bottom, left, right} in inches, the plot box's place in the figure (owner
    2026-10-27: panels of one row with the same height and the same top and bottom margins align in a facet).
    With it the figure keeps its exact width x height (no trimming to its drawing); a colour bar sits in the right
    margin. Empty: the figure is trimmed to its drawing plus export.margin, as before.
    """
    if raw in (None, "", {}):
        return None
    where = f"graph job id={job.get('id')} style.figure.margins"
    if not isinstance(raw, Mapping) or set(raw) - {"top", "bottom", "left", "right"}:
        raise GraphJobValidationError(f"{where} is {{top, bottom, left, right}} in inches, got {raw!r}")
    out = {}
    for side in ("top", "bottom", "left", "right"):
        value = _normalize_float(raw.get(side, 0.0), default=-1.0)
        if value < 0:
            raise GraphJobValidationError(f"{where}.{side}={raw.get(side)!r} is a length in inches, >= 0")
        out[side] = value
    return out


def _normalize_legend_anchor(raw: Any, job: Mapping[str, Any]) -> list[float] | None:
    """legend.anchor: [x, y], where the legend's position point (legend.position) sits, as fractions of the plot
    box (0 0 its lower left, 1 1 its upper right): a legend placed inside the box exactly (owner 2026-10-27)."""
    if raw in (None, "", []):
        return None
    if (not isinstance(raw, list) or len(raw) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw)):
        raise GraphJobValidationError(f"graph job id={job.get('id')} legend.anchor is [x, y] (plot-box fractions), got {raw!r}")
    return [float(raw[0]), float(raw[1])]


def _normalize_colormap(raw: Any, job: Mapping[str, Any]) -> str:
    """style.colormap: the colour scale of a heatmap, a Matplotlib colormap name; default viridis."""
    import matplotlib

    name = "viridis" if raw in (None, "") else str(raw).strip()
    if name not in matplotlib.colormaps:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.colormap={raw!r} is not a Matplotlib colormap (e.g. turbo, viridis)")
    return name


def _normalize_heatmap(raw: Any, job: Mapping[str, Any]) -> dict[str, Any]:
    """
    style.heatmap (owner 2026-10-28, Fig. 5.ii): limits [low, high], one colour scale for several heat maps (empty:
    0 to the largest value); colorbar (default true) false leaves the heat map's own out, a facet's legend.colormap
    drawing one for all; cell_values the cells' number format (default ",.0f"); row_groups (default false) true
    groups row names "group, item": the items as the row names, a bracket and the group's name beside them.
    """
    where = f"graph job id={job.get('id')} style.heatmap"
    raw = {} if raw in (None, "") else raw
    if not isinstance(raw, Mapping) or set(raw) - {"limits", "colorbar", "cell_values", "row_groups"}:
        raise GraphJobValidationError(f"{where} is {{limits, colorbar, cell_values, row_groups}}, got {raw!r}")
    limits = raw.get("limits")
    if limits not in (None, "", []):
        if (not isinstance(limits, list) or len(limits) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in limits)
                or not limits[0] < limits[1]):
            raise GraphJobValidationError(f"{where}.limits is [low, high] with low < high, got {limits!r}")
        limits = [float(limits[0]), float(limits[1])]
    else:
        limits = None
    cell = str(raw.get("cell_values") or ",.0f").strip()
    try:
        format(4999.0, cell)
    except (ValueError, TypeError):
        raise GraphJobValidationError(f"{where}.cell_values={cell!r} is not a number format (e.g. \",.0f\")") from None
    return {"limits": limits, "colorbar": _normalize_bool(raw.get("colorbar", True), default=True),
            "cell_values": cell, "row_groups": _normalize_bool(raw.get("row_groups", False), default=False)}


def _normalize_legend_entries(raw: Any, job: Mapping[str, Any]) -> list[dict[str, str]]:
    """
    legend.entries: the legend's own entries in place of one per series, each {label, color, hatch, linestyle}: a
    patch of that colour (default grey) and hatch, or with linestyle a line of that style. Owner 2026-10-27:
    "N_max = 0" solid, "N_max = 100" hatched; Fig. 26's epoch colours and "HW" dashed, "ID" solid.
    """
    if raw in (None, "", []):
        return []
    where = f"graph job id={job.get('id')} legend.entries"
    if not isinstance(raw, list):
        raise GraphJobValidationError(f"{where} must be a list of {{label, color, hatch}}")
    out = []
    for i, entry in enumerate(raw):
        if (not isinstance(entry, Mapping) or not str(entry.get("label") or "").strip()
                or set(entry) - {"label", "color", "hatch", "linestyle", "width", "dashes"}):
            raise GraphJobValidationError(
                f"{where}[{i}] is {{label, color, hatch, linestyle, width, dashes}} with a label, got {entry!r}")
        linestyle = entry.get("linestyle")
        if linestyle not in (None, "") and str(linestyle).strip() not in LINE_STYLES:
            raise GraphJobValidationError(f"{where}[{i}].linestyle={linestyle!r} invalid. Allowed: {sorted(LINE_STYLES)}")
        out.append({"label": str(entry["label"]).strip(),
                    "color": _colour(entry.get("color") or "0.6", f"{where}[{i}].color"),
                    "hatch": _hatch(entry.get("hatch"), f"{where}[{i}].hatch"),
                    **({"linestyle": str(linestyle).strip()} if linestyle not in (None, "") else {}),
                    **_line_look(entry, f"{where}[{i}]")})
    return out


def _normalize_line_styles(raw: Any, job: Mapping[str, Any]) -> dict[str, str]:
    """
    style.line.styles: series key -> Matplotlib line style, e.g.
        styles: {cross_entropy: "--"}            every curve of that metric
        styles: {"m1:cross_entropy": ":"}         one model's curve of it
        styles: {m1: "--"}                        every curve of that model
    A series without an entry is drawn solid.
    """
    if raw in (None, {}):
        return {}
    if not isinstance(raw, Mapping):
        raise GraphJobValidationError(f"graph job id={job.get('id')} style.line.styles must be a mapping")
    out = {}
    for key, value in raw.items():
        value = str(value).strip()
        if value not in LINE_STYLES:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} style.line.styles[{key!r}]={value!r} invalid. "
                f"Allowed: {sorted(LINE_STYLES)}"
            )
        out[str(key).strip()] = value
    return out


def normalize_metric_names(job: Mapping[str, Any]) -> list[str]:
    """
    The metrics of a job, in order. A job names one metric; a model_overlay of
    per-epoch scalars may name several, each model then draws one curve per
    metric (legend.labels key "<model_name>:<metric>").
    """
    metric = job.get("metric")
    items = list(metric) if isinstance(metric, list) else [metric]
    names = [_normalize_one_metric(m) for m in items]
    if len(names) > 1 and str(job.get("type", "")).strip() != "model_overlay":
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} metric must name one metric, got {metric!r}. "
            "Use one job per metric (several metrics only in a model_overlay)."
        )
    return names


def normalize_metric_name(job: Mapping[str, Any]) -> str:
    """The first (usually the only) metric of a job."""
    return normalize_metric_names(job)[0]


def split_processes(job: Mapping[str, Any]) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    """
    Split a job's processes into plain processes and combinations.

      processes:
        - "train"
        - "validation"
        - combine:
            processes: ["train", "validation"]
            mode: "sum"          # sum | mean

    Every item is one series: a name is that process, a combine item the
    metric of its processes added (or averaged) epoch by epoch, inside every
    fold first for source: cross_validation. Returns the plain names and the
    combinations keyed by their label, e.g. "train+validation".
    """
    raw = job.get("processes")
    if not isinstance(raw, list) or not raw:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} requires non-empty list field: processes"
        )

    plain: list[Any] = []
    combines: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            plain.append(item)
            continue
        if list(item) != ["combine"] or not isinstance(item["combine"], Mapping):
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} processes item {dict(item)!r} is not understood; "
                "write a process name or '- combine: {processes: [...], mode: sum}'"
            )
        spec = item["combine"]
        members = spec.get("processes")
        if not isinstance(members, list):
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} combine.processes must be a list"
            )
        _validate_processes(job, members)
        members = list(dict.fromkeys(normalize_process_name(str(p)) for p in members))
        if len(members) < 2:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} combine needs at least two processes, got {members}"
            )
        if str(job.get("source", "attack_metrics")) == "key_recovery":
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} combine is not available for source: key_recovery"
            )
        mode = str(spec.get("mode", "sum")).strip().lower()
        if mode not in VALID_COMBINE_MODES:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} combine.mode={mode!r} invalid. "
                f"Allowed: {sorted(VALID_COMBINE_MODES)}"
            )
        label = COMBINE_SEPARATOR.join(members)
        if label in combines:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} combines {members} twice"
            )
        combines[label] = {"processes": members, "mode": mode, "label": label}

    if job.get("combine") is not None:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')}: combine belongs under processes, as the list item "
            "'- combine: {processes: [...], mode: sum}'"
        )
    return plain, combines


def expand_models(job: Mapping[str, Any]) -> list[str]:
    """
    Resolve a job's models list: job 0 plots its top_models (written by the
    model judge), every other job the models it lists itself. An entry is a
    model_name or a model_hash; resolve_model_refs binds it to its run.
    """
    raw = job.get("top_models") if str(job.get("id")) == str(DEFAULT_JOB_ID) else job.get("models")
    if isinstance(raw, str):
        raw = [raw]
    for m in raw or []:
        # YAML reads an unquoted all-digit hash as a number (a leading 0 as
        # octal), which would name another model.
        if not isinstance(m, str):
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} model entry {m!r} is not text. List a "
                "model_name or a model_hash, and quote a hash made of digits: \"01234567\"."
            )
    return list(dict.fromkeys(m.strip() for m in raw or [] if m.strip()))


def resolve_model_refs(
    refs: list[str],
    *,
    registry: Mapping[str, Any],
    name_map: Mapping[str, Any],
    env: str,
    job_id: Any = None,
) -> list[dict[str, str]]:
    """
    Bind the models of a job to trained runs.

    An entry is a model_name or a model_hash:
      model_name  model_name_map.json and run_registry.json must give it the
                  same model_hash
      model_hash  must be a run in run_registry.json; its model_name there
                  must match model_name_map.json
    The run must be one of env, whose artifacts the job reads. Returns one
    {"ref", "model_name", "model_hash", "status"} per model in job order; an
    entry naming a model already listed adds nothing. Every entry that does
    not bind is reported at once, with the closest names and hashes of env.
    """
    runs = {str(h): e for h, e in registry.items() if isinstance(e, Mapping)}
    names = {str(h): str(n) for h, n in name_map.items()}

    def run_name(model_hash: str) -> str:
        return str(runs[model_hash].get("model_name") or "").strip()

    bound: dict[str, dict[str, str]] = {}
    problems: list[str] = []

    for ref in refs:
        map_hashes = sorted(h for h, n in names.items() if n == ref)
        run_hashes = sorted(h for h in runs if run_name(h) == ref)
        is_hash = ref in runs or ref in names

        if is_hash and (map_hashes or run_hashes) and {ref} != set(map_hashes) | set(run_hashes):
            problems.append(
                f"{ref}: is a model_hash and the model_name of {sorted(set(map_hashes) | set(run_hashes))}; "
                "list the model_hash you mean"
            )
            continue

        if is_hash:
            model_hash = ref
        elif map_hashes or run_hashes:
            if map_hashes != run_hashes:
                problems.append(
                    f"{ref}: model_name_map.json gives model_hash {map_hashes or 'none'}, "
                    f"run_registry.json {run_hashes or 'none'} (deleted, or the registries disagree)"
                )
                continue
            if len(map_hashes) > 1:
                problems.append(f"{ref}: names the runs {map_hashes}; list the model_hash you mean")
                continue
            model_hash = map_hashes[0]
        else:
            known = sorted({
                key
                for h, e in runs.items() if str(e.get("env") or "") == env
                for key in (h, run_name(h)) if key
            })
            close = difflib.get_close_matches(ref, known, n=5, cutoff=0.6)
            problems.append(
                f"{ref}: no model_name or model_hash of this project"
                + (f"; close matches: {close}" if close else "")
            )
            continue

        if model_hash not in runs:
            problems.append(
                f"{ref}: model_name_map.json names it {names[model_hash]!r}, but run_registry.json "
                "has no run of it (deleted, or never trained)"
            )
            continue
        model_name = run_name(model_hash)
        if not model_name or names.get(model_hash) != model_name:
            problems.append(
                f"{ref}: run_registry.json names model_hash {model_hash} {model_name or None!r}, "
                f"model_name_map.json {names.get(model_hash)!r}; the registries disagree"
            )
            continue
        run_env = str(runs[model_hash].get("env") or "")
        if run_env != env:
            problems.append(
                f"{ref}: {model_name} ({model_hash}) was trained with ENV={run_env or 'unknown'}; "
                f"run the graph job with ENV={run_env or '<its env>'}"
            )
            continue
        bound.setdefault(model_hash, {
            "ref": ref,
            "model_name": model_name,
            "model_hash": model_hash,
            "status": str(runs[model_hash].get("status") or "").upper() or "UNKNOWN",
        })

    if problems:
        raise GraphJobValidationError(
            f"graph job id={job_id}: cannot bind its models to trained runs of ENV={env}:\n  - "
            + "\n  - ".join(problems)
            + "\nList a model_name (artifacts/done_model_names.yaml) or a model_hash "
            "(artifacts/run_registry.json) of a run of this ENV."
        )
    return list(bound.values())


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "on"}:
        return True

    if text in {"0", "false", "no", "n", "off"}:
        return False

    return default


def _normalize_int(value: Any, *, default: int, min_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = default

    if min_value is not None:
        out = max(out, min_value)

    return out


def _normalize_float(
    value: Any,
    *,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        out = float(value)
    except Exception:
        out = default

    if min_value is not None:
        out = max(out, min_value)

    if max_value is not None:
        out = min(out, max_value)

    return out


def normalize_output_file_name(job: Mapping[str, Any]) -> str:
    """
    Official field:
        output_file_name

    Backward-compatible fallback:
        output_name
    """

    raw = job.get("output_file_name", job.get("output_name", ""))
    value = str(raw).strip()

    if not value:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} requires non-empty output_file_name"
        )

    return value


def normalize_export_config(job: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize export configuration.

    Supported (each format's own settings under its name; pdf and eps have none):
      export:
        formats: ["svg"]            # the default (owner 2026-10-27); png | svg | pdf | eps
        margin: 0.05
        save_raw_plot_data: false
        png:
          dpi: 300
        svg:
          sanitize: true
          validate_xml: true
          on_error: "warn"

    JPG/JPEG are rejected because they are inappropriate for scientific graphs.
    export.dpi moved to export.png.dpi (dpi is used by png only); export.pdf was
    removed (a PDF is vector, its embed_fonts / vector keys were never read).
    """

    raw = job.get("export", {}) or {}

    if not isinstance(raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} export must be mapping/dict-like"
        )

    formats_raw = raw.get("formats", ["svg"])     # owner 2026-10-27: SVG only unless a job asks for more

    if isinstance(formats_raw, str):
        formats_raw = [formats_raw]

    if not isinstance(formats_raw, list) or not formats_raw:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} export.formats must be a non-empty list"
        )

    formats: list[str] = []

    for fmt in formats_raw:
        f = str(fmt).strip().lower().lstrip(".")

        if not f:
            continue

        if f in FORBIDDEN_EXPORT_FORMATS:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} forbidden export format={f!r}. "
                "Do not use jpg/jpeg for scientific graphs. Use png for preview "
                "and pdf/svg/eps for vector outputs."
            )

        if f not in VALID_EXPORT_FORMATS:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} invalid export format={f!r}. "
                f"Allowed: {sorted(VALID_EXPORT_FORMATS)}"
            )

        if f not in formats:
            formats.append(f)

    if not formats:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} export.formats resolved to empty list"
        )

    svg_raw = raw.get("svg", {}) or {}
    if not isinstance(svg_raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} export.svg must be mapping/dict-like"
        )

    svg_on_error = str(svg_raw.get("on_error", "warn")).strip().lower()
    if svg_on_error not in VALID_SVG_ON_ERROR:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} export.svg.on_error={svg_on_error!r} invalid. "
            f"Allowed: {sorted(VALID_SVG_ON_ERROR)}"
        )

    if "dpi" in raw:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} export.dpi moved to export.png.dpi (only png uses it)"
        )
    if "pdf" in raw:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} export.pdf was removed: a PDF is vector and has no "
            "settings (embed_fonts / vector were never read); delete export.pdf"
        )
    png_raw = raw.get("png", {}) or {}
    if not isinstance(png_raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} export.png must be mapping/dict-like"
        )

    return {
        "formats": formats,
        "png": {"dpi": _normalize_int(png_raw.get("dpi", 300), default=300, min_value=72)},
        # White border around the drawn figure, in inches. The saved page is
        # the figure itself plus this margin, in every format.
        "margin": _normalize_float(
            raw.get("margin", 0.05), default=0.05, min_value=0.0, max_value=2.0
        ),
        "save_raw_plot_data": _normalize_bool(
            raw.get("save_raw_plot_data", False),
            default=False,
        ),
        "svg": {
            "sanitize": _normalize_bool(svg_raw.get("sanitize", True), default=True),
            "validate_xml": _normalize_bool(
                svg_raw.get("validate_xml", True),
                default=True,
            ),
            "on_error": svg_on_error,
        },
    }


def normalize_legend_config(job: Mapping[str, Any]) -> dict[str, Any]:
    raw = job.get("legend", {}) or {}

    if not isinstance(raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} legend must be mapping/dict-like"
        )

    labels = raw.get("labels", {}) or {}
    if not isinstance(labels, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} legend.labels must be mapping/dict-like"
        )

    mode = str(raw.get("mode", "auto")).strip().lower()
    if mode not in VALID_LEGEND_MODES:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} legend.mode={mode!r} invalid. "
            f"Allowed: {sorted(VALID_LEGEND_MODES)}"
        )

    try:
        handle = parse_legend_handle(raw.get("handle"))
    except ValueError as exc:
        raise GraphJobValidationError(f"graph job id={job.get('id')} {exc}") from exc

    # A position the plotter does not know used to become "best" without a word.
    from services.graph_manager.plotters import POSITION_MAP

    position = str(raw.get("position", "best")).strip().lower()
    if position not in POSITION_MAP:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} legend.position={position!r} invalid. "
            f"Allowed: {sorted(POSITION_MAP)}"
        )

    return {
        "enabled": _normalize_bool(raw.get("enabled", True), default=True),
        "position": position,
        "mode": mode,
        "labels": {str(k): str(v) for k, v in dict(labels).items()},
        # [shape, thickness pt] of the legend entries; None keeps Matplotlib's.
        "handle": handle,
        "entries": _normalize_legend_entries(raw.get("entries"), job),
        # entries per legend row (owner 2026-10-27: Fig. 24's eight activations, four by two, below the axes)
        "columns": _normalize_int(raw.get("columns", 1), default=1, min_value=1),
        "anchor": _normalize_legend_anchor(raw.get("anchor"), job),
    }


def _normalize_font_config(raw: Any, *, default_size: int = 10) -> dict[str, Any]:
    if raw is None:
        raw = {}

    if not isinstance(raw, Mapping):
        raise GraphJobValidationError("style.font must be mapping/dict-like")

    return {
        "family": str(raw.get("family", "serif")).strip() or "serif",
        "base_size": _normalize_int(
            raw.get("base_size", default_size),
            default=default_size,
            min_value=1,
        ),
        "title_size": _normalize_int(
            raw.get("title_size", raw.get("base_size", default_size)),
            default=default_size,
            min_value=1,
        ),
        "label_size": _normalize_int(
            raw.get("label_size", max(default_size - 1, 1)),
            default=max(default_size - 1, 1),
            min_value=1,
        ),
        "tick_size": _normalize_int(
            raw.get("tick_size", max(default_size - 2, 1)),
            default=max(default_size - 2, 1),
            min_value=1,
        ),
        "legend_size": _normalize_int(
            raw.get("legend_size", max(default_size - 2, 1)),
            default=max(default_size - 2, 1),
            min_value=1,
        ),
    }


def _normalize_text_rule(rule: Any, *, job_id: Any) -> dict[str, Any]:
    if not isinstance(rule, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job_id} style.text_rules entries must be mapping/dict-like"
        )

    target = str(rule.get("target", "")).strip()
    if not target:
        raise GraphJobValidationError(
            f"graph job id={job_id} style.text_rules entry missing target"
        )

    if target in REMOVED_TEXT_TARGETS:
        raise GraphJobValidationError(
            f"graph job id={job_id} style.text_rules target={target!r} was removed; use "
            f"{REMOVED_TEXT_TARGETS[target]}"
        )
    if target not in VALID_TEXT_TARGETS:
        raise GraphJobValidationError(
            f"graph job id={job_id} style.text_rules target={target!r} invalid. "
            f"Allowed: {sorted(VALID_TEXT_TARGETS)}"
        )
    if "underline" in rule:
        raise GraphJobValidationError(
            f"graph job id={job_id} style.text_rules target={target!r}: underline was removed, "
            "Matplotlib cannot underline a text (Text has no underline, mathtext no \\underline)"
        )

    # Current schema uses `text`.
    # Accept legacy `words` only as a compatibility bridge.
    text = rule.get("text", rule.get("words", ""))
    if isinstance(text, list):
        if len(text) > 1:
            raise GraphJobValidationError(
                f"graph job id={job_id} style.text_rules target={target!r} "
                "must use a single text string, not a multi-item list"
            )
        text = text[0] if text else ""

    text = str(text).strip()

    font_raw = rule.get("font", {}) or {}
    if not isinstance(font_raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job_id} style.text_rules target={target!r} font must be mapping"
        )

    font = {
        "family": str(font_raw.get("family", "serif")).strip() or "serif",
        "size": _normalize_int(font_raw.get("size", 10), default=10, min_value=1),
    }

    return {
        "target": target,
        "text": text,
        "bold": _normalize_bool(rule.get("bold", False), default=False),
        "italic": _normalize_bool(rule.get("italic", False), default=False),
        "color": str(rule.get("color", "black")).strip() or "black",
        "font": font,
    }


def normalize_style_config(job: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize style configuration.

    This validates the YAML contract. Actual rendering behavior is implemented
    in plot_style.py / plotters.py.
    """

    raw = job.get("style", {}) or {}

    if not isinstance(raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style must be mapping/dict-like"
        )

    font = _normalize_font_config(raw.get("font", {}), default_size=10)

    line_raw = raw.get("line", {}) or {}
    if not isinstance(line_raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.line must be mapping/dict-like"
        )

    grid_raw = raw.get("grid", {}) or {}
    if not isinstance(grid_raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.grid must be mapping/dict-like"
        )

    std_band_raw = raw.get("std_band", {}) or {}
    if not isinstance(std_band_raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.std_band must be mapping/dict-like"
        )

    axes_raw = raw.get("axes", {}) or {}
    if not isinstance(axes_raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.axes must be mapping/dict-like"
        )

    x_scale = str(axes_raw.get("x_scale", "linear")).strip().lower()
    y_scale = str(axes_raw.get("y_scale", "linear")).strip().lower()

    if x_scale not in VALID_SCALES:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.axes.x_scale={x_scale!r} invalid. "
            f"Allowed: {sorted(VALID_SCALES)}"
        )

    if y_scale not in VALID_SCALES:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.axes.y_scale={y_scale!r} invalid. "
            f"Allowed: {sorted(VALID_SCALES)}"
        )

    text_rules_raw = raw.get("text_rules", []) or []
    if not isinstance(text_rules_raw, list):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.text_rules must be a list"
        )

    def limits(name: str) -> list[float] | None:
        """[min, max] of an axis, or None to let the data decide."""
        value = axes_raw.get(name)
        if value is None:
            return None
        try:
            low, high = (float(v) for v in value)
        except Exception as exc:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} style.axes.{name} must be [min, max], got {value!r}"
            ) from exc
        if not low < high:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} style.axes.{name}: min must be below max, got {value!r}"
            )
        return [low, high]

    x_limits = limits("x_limits")
    y_limits = limits("y_limits")

    def step(name: str, scale: str) -> float | None:
        """Spacing of the ticks on a linear axis, or None for Matplotlib's."""
        value = axes_raw.get(name)
        if value is None:
            return None
        try:
            spacing = float(value)
        except Exception as exc:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} style.axes.{name} must be a number, got {value!r}"
            ) from exc
        if not spacing > 0:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} style.axes.{name} must be above 0, got {value!r}"
            )
        if scale != "linear":
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} style.axes.{name} needs a linear axis, "
                f"got scale {scale!r}"
            )
        return spacing

    x_step = step("x_step", x_scale)
    y_step = step("y_step", y_scale)
    # Draw only the points whose x is a multiple of x_grid (the paper's trace
    # grid, e.g. 50); the recorded curve is unchanged.
    x_grid = step("x_grid", x_scale)
    # Also draw the first recorded point (n = 1 of a run recorded at every trace),
    # which x_grid would leave out; it stays at its recorded x = 1.
    x_grid_first_raw = axes_raw.get("x_grid_first", False)
    if not isinstance(x_grid_first_raw, bool):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.axes.x_grid_first must be true or false, got {x_grid_first_raw!r}"
        )
    if x_grid_first_raw and x_grid is None:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.axes.x_grid_first keeps a point x_grid would drop; set style.axes.x_grid too"
        )

    def margin(name: str, limits_name: str, limits_value) -> float:
        """Padding as a fraction of the limits' span (Matplotlib's default is 0.05); 0 for none."""
        value = axes_raw.get(name, 0.0)
        try:
            fraction = float(0.0 if value is None else value)
        except Exception as exc:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} style.axes.{name} must be a number, got {value!r}"
            ) from exc
        if fraction < 0:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} style.axes.{name} must be 0 or above, got {value!r}"
            )
        if fraction > 0 and limits_value is None:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} style.axes.{name} pads style.axes.{limits_name}; set it too"
            )
        return fraction

    x_margin = margin("x_margin", "x_limits", x_limits)
    y_margin = margin("y_margin", "y_limits", y_limits)

    figure_raw = raw.get("figure", {}) or {}
    if not isinstance(figure_raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.figure must be mapping/dict-like"
        )
    orientation = str(figure_raw.get("orientation", "landscape")).strip().lower()
    if orientation not in VALID_ORIENTATIONS:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} style.figure.orientation={orientation!r} "
            f"invalid. Allowed: {sorted(VALID_ORIENTATIONS)}"
        )

    return {
        "figure": {
            "orientation": orientation,
            # Inches; 0 keeps the default of the orientation.
            "width": _normalize_float(figure_raw.get("width", 0.0), default=0.0, min_value=0.0),
            "height": _normalize_float(figure_raw.get("height", 0.0), default=0.0, min_value=0.0),
            "margins": _normalize_figure_margins(figure_raw.get("margins"), job),
        },
        "font": font,
        "line": {
            "width": _normalize_float(
                line_raw.get("width", 1.0),
                default=1.0,
                min_value=0.1,
            ),
            "marker": str(line_raw.get("marker", "o")).strip(),
            "marker_size": _normalize_float(
                line_raw.get("marker_size", 3.5),
                default=3.5,
                min_value=0.1,
            ),
            "styles": _normalize_line_styles(line_raw.get("styles"), job),
        },
        "series": _normalize_series_style(raw.get("series"), job),
        # a heatmap's colour scale: any Matplotlib colormap (owner 2026-10-27: turbo, the RGB spectrum)
        "colormap": _normalize_colormap(raw.get("colormap"), job),
        "heatmap": _normalize_heatmap(raw.get("heatmap"), job),
        # each bar's value printed at its end, in this format (owner 2026-10-27: Fig. 20 b's accuracies)
        "bar_values": _normalize_bar_values(raw.get("bar_values"), job),
        "grid": {
            "enabled": _normalize_bool(grid_raw.get("enabled", True), default=True),
            "alpha": _normalize_float(
                grid_raw.get("alpha", 0.35),
                default=0.35,
                min_value=0.0,
                max_value=1.0,
            ),
        },
        # Mean +/- std over the cross-validation folds. false: the std is
        # not loaded, so no band and no error bars (the paper's figures).
        "std_band": {
            "enabled": _normalize_bool(std_band_raw.get("enabled", True), default=True),
            "alpha": _normalize_float(
                std_band_raw.get("alpha", 0.15),
                default=0.15,
                min_value=0.0,
                max_value=1.0,
            ),
        },
        "axes": {
            "x_scale": x_scale,
            "y_scale": y_scale,
            "x_limits": x_limits,
            "y_limits": y_limits,
            "x_step": x_step,
            "y_step": y_step,
            "x_grid": x_grid,
            "x_grid_first": x_grid_first_raw,
            "x_margin": x_margin,
            "y_margin": y_margin,
        },
        "text_rules": [
            _normalize_text_rule(rule, job_id=job.get("id"))
            for rule in text_rules_raw
        ],
    }

def normalize_array_epoch_config(job: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize epoch-sliced array artifact policy.

    Supported YAML contract:

      array_epoch:
        mode: "latest"          # latest | selected | overlay | summary

        # for selected mode
        epoch: "ep0010"

        # for overlay mode
        epochs:
          - "ep0001"
          - "ep0005"
          - "ep0010"

        # for summary mode
        summary:
          type: "final_value"   # final_value | value_at_trace | traces_to_threshold | min_value
          trace_index: 2000
          threshold: 1.0
    """

    raw = job.get("array_epoch", {}) or {}

    if not isinstance(raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} array_epoch must be mapping/dict-like"
        )

    mode = str(raw.get("mode", "latest")).strip().lower()

    if mode not in VALID_ARRAY_EPOCH_MODES:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} array_epoch.mode={mode!r} invalid. "
            f"Allowed: {sorted(VALID_ARRAY_EPOCH_MODES)}"
        )

    epoch_raw = raw.get("epoch", None)
    epoch = str(epoch_raw).strip() if epoch_raw is not None else ""

    epochs_raw = raw.get("epochs", []) or []
    if isinstance(epochs_raw, str):
        epochs_raw = [epochs_raw]

    if not isinstance(epochs_raw, list):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} array_epoch.epochs must be a list"
        )

    epochs = [str(e).strip() for e in epochs_raw if str(e).strip()]

    summary_raw = raw.get("summary", {}) or {}
    if not isinstance(summary_raw, Mapping):
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} array_epoch.summary must be mapping/dict-like"
        )

    summary_type = str(summary_raw.get("type", "final_value")).strip().lower()

    if summary_type not in VALID_ARRAY_EPOCH_SUMMARIES:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} array_epoch.summary.type={summary_type!r} invalid. "
            f"Allowed: {sorted(VALID_ARRAY_EPOCH_SUMMARIES)}"
        )

    trace_index = _normalize_int(
        summary_raw.get("trace_index", -1),
        default=-1,
        min_value=-1,
    )

    threshold = _normalize_float(
        summary_raw.get("threshold", 1.0),
        default=1.0,
    )

    if mode == "selected" and not epoch:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} array_epoch.mode='selected' requires array_epoch.epoch"
        )

    if mode == "overlay" and not epochs:
        raise GraphJobValidationError(
            f"graph job id={job.get('id')} array_epoch.mode='overlay' requires non-empty array_epoch.epochs"
        )

    if mode == "summary":
        if summary_type == "value_at_trace" and trace_index < 1:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} summary.type='value_at_trace' "
                "requires array_epoch.summary.trace_index >= 1 (a trace count)"
            )

        if summary_type == "traces_to_threshold" and threshold <= 0:
            raise GraphJobValidationError(
                f"graph job id={job.get('id')} summary.type='traces_to_threshold' "
                "requires array_epoch.summary.threshold > 0"
            )

    return {
        "mode": mode,
        "epoch": epoch,
        "epochs": epochs,
        "summary": {
            "type": summary_type,
            "trace_index": trace_index,
            "threshold": threshold,
        },
    }

RELATIVE_RULES = ("run", "anchor", "probe")


def _relative_label(conditions: list[Mapping[str, Any]]) -> str:
    """'batch_size = 200' from [{path: trainer.batch_size, equals: 200}]."""
    parts = []
    for cond in conditions:
        name = str(cond["path"]).split(".")[-1]
        op, value = cond["op"], cond["value"]
        if op == "equals":
            parts.append(f"{name} = {value}")
        elif op == "in":
            parts.append(f"{name} in {', '.join(map(str, value))}")
        else:
            parts.append(f"{name} {op} {value}")
    return ", ".join(parts)


def normalize_relative_config(job: Mapping[str, Any]) -> dict[str, Any] | None:
    """
    relative: every series divided by the same metric of one reference run,
    read exactly as the job reads its models, so the reference is 1.0 and any
    scalar (fit time, accuracy, loss, ...) becomes a ratio. Empty or absent:
    the values as recorded.

        relative:
          rule: run      # run: the reference is one of the job's models
                         # anchor: any finished run of this ENV (listed or not)
          where:         # model_filter conditions; exactly one run must match
            - {path: trainer.batch_size, equals: 200}
          label: "batch size 200"   # optional; default from where

        relative:
          rule: probe    # each run's time divided by its own device probe
                         # (<run>/device.yaml (paths.device)), in probe steps: comparable
                         # across devices; time metrics only, no where

    With rule run or anchor, a time metric is divided only between runs whose
    device records agree (GPU, torch, CUDA, cuDNN, TF32 switches).
    """
    raw = job.get("relative")
    if raw in (None, "", {}, []):
        return None
    job_id = job.get("id")
    if not isinstance(raw, Mapping):
        raise GraphJobValidationError(f"graph job id={job_id} relative must be a mapping with rule and where")
    unknown = sorted(set(raw) - {"rule", "where", "label"})
    if unknown:
        raise GraphJobValidationError(f"graph job id={job_id} relative: unknown key(s) {unknown}; keys: rule, where, label")
    rule = str(raw.get("rule") or "").strip().lower()
    if rule not in RELATIVE_RULES:
        raise GraphJobValidationError(f"graph job id={job_id} relative.rule must be one of {RELATIVE_RULES}")
    if str(job.get("type")) != "model_overlay":
        raise GraphJobValidationError(f"graph job id={job_id} relative compares runs: use type: model_overlay")
    if rule == "probe":
        if raw.get("where") not in (None, [], ""):
            raise GraphJobValidationError(
                f"graph job id={job_id} relative.rule: probe divides each run by its own device probe; "
                "it takes no where"
            )
        label = str(raw.get("label") or "").strip() or "the device probe (probe steps)"
        return {"rule": rule, "where": [], "label": label}
    where = raw.get("where")
    if not isinstance(where, list) or not where:
        raise GraphJobValidationError(
            f"graph job id={job_id} relative.where needs at least one condition, "
            "e.g. [{path: trainer.batch_size, equals: 200}]"
        )
    from services.model_filter.model_filter import ModelFilterError, _validate_condition

    try:
        conditions = [_validate_condition(cond, index=i) for i, cond in enumerate(where)]
    except ModelFilterError as exc:
        raise GraphJobValidationError(f"graph job id={job_id} relative.{exc}") from exc
    label = str(raw.get("label") or "").strip() or _relative_label(conditions)
    return {
        "rule": rule,
        "where": [
            {"path": c["path"], c["op"]: c["value"], **({"quantifier": "all"} if c["quantifier"] == "all" else {})}
            for c in conditions
        ],
        "label": label,
    }


# groups.band: the spread a group of runs is shaded with (owner 2026-10-24; Wouters et al. TCHES 2020(3) Figs. 1-2
# draw "error bands" over ten trainings without saying which statistic, so the job must name one)
GROUP_BANDS = ("min_max", "std", "none")


def normalize_groups_config(job: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """
    groups: runs pooled into one series each, in a model_overlay of a curve (e.g. the mean rank over the
    attacks) or of per-epoch scalars (e.g. the loss: one series per group and metric): every member's curve
    drawn thin, their mean bold, and the band the group names shaded.

        groups:
          - label: "Zaid N0, fs01"
            models: [run_a, run_b, ...]   # runs of the job's models; one group each
            band: min_max                 # min_max | std (mean +- sample std) | none; no default
            members: false                # optional: the mean and band only, no thin curve per run (default true)
            color: "#4A148C"              # optional: the group's colour (default: its slot of the colour cycle)
            band_color: "#7E57C2"         # optional: its band's (default: its colour)

    Empty or absent: one curve per model, as before. A model of the job in no group is drawn as usual.
    """
    raw = job.get("groups")
    if raw in (None, "", [], {}):
        return None
    job_id = job.get("id")
    if str(job.get("type")) != "model_overlay":
        raise GraphJobValidationError(f"graph job id={job_id} groups pool runs: use type: model_overlay")
    if not isinstance(raw, list):
        raise GraphJobValidationError(f"graph job id={job_id} groups must be a list of {{label, models, band}}")
    listed = [str(m).strip() for m in (job.get("models") or []) if str(m).strip()]
    seen: dict[str, str] = {}
    out = []
    for index, group in enumerate(raw):
        where = f"graph job id={job_id} groups[{index}]"
        if not isinstance(group, Mapping):
            raise GraphJobValidationError(f"{where} must be a mapping of label, models, band")
        unknown = sorted(set(group) - set(GROUP_KEYS))
        if unknown:
            raise GraphJobValidationError(f"{where}: unknown key(s) {unknown}; keys: {', '.join(GROUP_KEYS)}")
        if "members" in group and not isinstance(group["members"], bool):
            raise GraphJobValidationError(f"{where}.members must be true or false, got {group['members']!r}")
        for key in ("color", "band_color"):
            if key in group and not _is_colour(group[key]):
                raise GraphJobValidationError(f"{where}.{key} must be a Matplotlib colour, got {group[key]!r}")
        label = str(group.get("label") or "").strip()
        if not label:
            raise GraphJobValidationError(f"{where}.label is required (the group's legend entry)")
        band = str(group.get("band") or "").strip().lower()
        if band not in GROUP_BANDS:
            raise GraphJobValidationError(
                f"{where}.band must be one of {list(GROUP_BANDS)} (no default: the statistic is stated, "
                f"never assumed), got {group.get('band')!r}")
        members = [str(m).strip() for m in (group.get("models") or []) if str(m).strip()]
        if not members:
            raise GraphJobValidationError(f"{where}.models lists no run")
        if band != "none" and len(members) < 2:
            raise GraphJobValidationError(f"{where}: a {band} band needs two runs at least, got {members}")
        for model in members:
            if model not in listed:
                raise GraphJobValidationError(f"{where}: {model!r} is not one of the job's models")
            if model in seen:
                raise GraphJobValidationError(f"{where}: {model!r} is already in group {seen[model]!r}")
            seen[model] = label
        out.append({"label": label, "models": members, "band": band,
                    **{key: group[key] for key in ("members", "color", "band_color") if key in group}})
    return out


# marks: points of a run's rank curves (the TraceFlow Monitor's three Done ranks), drawn on its figure
RANK_MARKS = ("validation_final", "test_at_validation", "test_final")
_MARK_PROCESSES = {"validation_final": ("validation",), "test_at_validation": ("validation", "test"),
                   "test_final": ("test",)}


def normalize_marks_config(job: Mapping[str, Any]) -> list[str] | None:
    """
    marks: points marked on a run's mean rank curves, as the TraceFlow Monitor's Done list reads them:

        type: process_overlay
        metric: mean_attack_rank_curve
        processes: [validation, test]
        marks: [validation_final, test_at_validation, test_final]   # any of them, in any order

    validation_final the validation curve's last point; test_at_validation the test curve at the validation's
    trace count (the like-for-like value: a rank falls as traces are added, so only these two compare);
    test_final the test curve's last point. Empty or absent: no marks.
    """
    raw = job.get("marks")
    if raw in (None, "", [], {}):
        return None
    job_id = job.get("id")
    marks = [raw] if isinstance(raw, str) else raw
    if not isinstance(marks, list) or not all(isinstance(m, str) for m in marks):
        raise GraphJobValidationError(f"graph job id={job_id} marks must be a list of {list(RANK_MARKS)}")
    marks = list(dict.fromkeys(m.strip() for m in marks))
    unknown = [m for m in marks if m not in RANK_MARKS]
    if unknown:
        raise GraphJobValidationError(f"graph job id={job_id} marks {unknown}: use any of {list(RANK_MARKS)}")
    if str(job.get("type")) != "process_overlay":
        raise GraphJobValidationError(
            f"graph job id={job_id} marks mark one run's validation and test curves: use type: process_overlay")
    if list(job.get("metrics") or [job.get("metric")]) != ["mean_attack_rank_curve"]:
        raise GraphJobValidationError(f"graph job id={job_id} marks are points of the mean rank curve: "
                                      "use metric: mean_attack_rank_curve")
    processes = set(job.get("processes") or [])
    for mark in marks:
        needed = [p for p in _MARK_PROCESSES[mark] if p not in processes]
        if needed:
            raise GraphJobValidationError(
                f"graph job id={job_id} mark {mark} reads the {' and '.join(_MARK_PROCESSES[mark])} curve: "
                f"add {needed} to processes (source attack_metrics holds both)")
    return marks


# overlays: what is drawn over a model_overlay of the GE curve (TODO.md section 29, the Model Performance Report)
OVERLAY_KEYS = ("ge_averages", "ntge", "trace_lines", "colors")
JOB_FOLDERS = ("run", "project")
OVERLAY_COLOURS = ("geometric", "median", "ntge")
GROUP_KEYS = ("label", "models", "band", "members", "color", "band_color")


def _is_colour(value: Any) -> bool:
    from matplotlib.colors import is_color_like

    return isinstance(value, str) and is_color_like(value)
GE_AVERAGES = ("geometric", "median")
TRACE_LINES = ("n_val", "budget")


def _overlay_list(raw: Any, allowed: tuple[str, ...], where: str) -> list[str]:
    """A list of some of `allowed` (one name alone too), duplicates dropped, in the order given."""
    items = [raw] if isinstance(raw, str) else raw
    if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
        raise GraphJobValidationError(f"{where} must be a list of {list(allowed)}, got {raw!r}")
    items = list(dict.fromkeys(i.strip().lower() for i in items))
    unknown = [i for i in items if i not in allowed]
    if unknown:
        raise GraphJobValidationError(f"{where} {unknown}: use any of {list(allowed)}")
    return items


def normalize_overlays_config(job: Mapping[str, Any]) -> dict[str, Any] | None:
    """
    overlays: drawn over each run's (or group's) GE curve, from the same record as the curve (the job's process
    and, for the test, its `test`):

        type: model_overlay
        metric: mean_attack_rank_curve
        source: attack_metrics
        processes: [test]                       # validation or test
        overlays:
          ge_averages: [geometric, median]      # over the attacks (per_attack_curves), beside the arithmetic GE
          ntge: true                            # each run's N̄tGE (per_attack_traces_to_ge_1) at GE 1, their median
          trace_lines: [n_val, budget]          # the validation traces, the data entry's attack_trace_budget
          colors: {geometric: "#00C853", median: "#FF00FF", ntge: "#222222"}   # optional: a mark's own colour
                                                # (default: its run's or group's; the N̄tGE median stays red)

    Empty or absent: none. A run without the record an overlay reads is left out of it with a warning.
    """
    raw = job.get("overlays")
    if raw in (None, "", {}):
        return None
    job_id = job.get("id")
    where = f"graph job id={job_id} overlays"
    if not isinstance(raw, Mapping):
        raise GraphJobValidationError(f"{where} must be a mapping of {list(OVERLAY_KEYS)}")
    unknown = sorted(set(raw) - set(OVERLAY_KEYS))
    if unknown:
        raise GraphJobValidationError(f"{where}: unknown key(s) {unknown}; keys: {list(OVERLAY_KEYS)}")
    if str(job.get("type")) != "model_overlay":
        raise GraphJobValidationError(f"{where} are drawn over the runs' GE curves: use type: model_overlay")
    if list(job.get("metrics") or [job.get("metric")]) != ["mean_attack_rank_curve"]:
        raise GraphJobValidationError(f"{where} are drawn over the GE curve: use metric: mean_attack_rank_curve")
    if str(job.get("source", "attack_metrics")) != "attack_metrics":
        raise GraphJobValidationError(f"{where} read the runs' attack records: use source: attack_metrics")
    if job.get("combines") or not set(job.get("processes") or []) <= {"validation", "test"}:
        raise GraphJobValidationError(
            f"{where} read a validation or test attack: processes validation and/or test, got {job.get('processes')}")
    if str((job.get("array_epoch") or {}).get("mode", "latest")) not in ("latest", "selected"):
        raise GraphJobValidationError(f"{where} read one record per run: array_epoch.mode latest or selected")
    out: dict[str, Any] = {}
    if raw.get("ge_averages") not in (None, "", []):
        out["ge_averages"] = _overlay_list(raw["ge_averages"], GE_AVERAGES, f"{where}.ge_averages")
    ntge = raw.get("ntge")
    if ntge not in (None, "", False):
        if ntge is not True:
            raise GraphJobValidationError(f"{where}.ntge is true or false, got {ntge!r}")
        if job.get("trace_budget") is not None:
            raise GraphJobValidationError(
                f"{where}.ntge is the N̄tGE recorded over the whole attack, not at trace_budget "
                f"{job['trace_budget']}: drop trace_budget, or draw a Budget test (test: budget)")
        out["ntge"] = True
    if raw.get("trace_lines") not in (None, "", []):
        out["trace_lines"] = _overlay_list(raw["trace_lines"], TRACE_LINES, f"{where}.trace_lines")
    colours = raw.get("colors")
    if colours not in (None, "", {}):
        if not isinstance(colours, Mapping) or set(colours) - set(OVERLAY_COLOURS):
            raise GraphJobValidationError(f"{where}.colors maps any of {list(OVERLAY_COLOURS)} to a colour, got {colours!r}")
        bad = {k: v for k, v in colours.items() if not _is_colour(v)}
        if bad:
            raise GraphJobValidationError(f"{where}.colors: not a Matplotlib colour: {bad}")
        out["colors"] = dict(colours)
    return out or None


def validate_graph_job(job: Mapping[str, Any], *, require_models: bool = True) -> dict[str, Any]:
    """
    Validate and return a normalized copy of one graph job.

    This does not check whether metric files exist.
    Metric existence belongs to metric_loader.py.
    """

    if not isinstance(job, Mapping):
        raise GraphJobValidationError(f"graph job must be dict-like, got {type(job)}")
    if str(job.get("type", "")).strip() == "model_explanation":
        return _validate_explanation_job(job, require_models=require_models)

    _require_keys(
        job,
        ["id", "name", "type", "enabled", "metric"],
    )

    out = dict(job)
    plain_processes, combines = split_processes(out)
    # The checks below see the plain processes; the combinations are added after them.
    out["processes"] = plain_processes or list(next(iter(combines.values()))["processes"])

    try:
        out["id"] = int(out["id"])
    except Exception as exc:
        raise GraphJobValidationError(
            f"graph job id must be int-like, got {job.get('id')!r}"
        ) from exc

    out["name"] = str(out["name"]).strip()
    out["type"] = str(out["type"]).strip()
    out["metrics"] = normalize_metric_names(out)
    out["metric"] = out["metrics"][0]
    out["mode"] = normalize_job_mode(out)
    out["plot_type"] = normalize_plot_type(out)
    out["output_file_name"] = normalize_output_file_name(out)
    folder = str(out.get("folder") or "run").strip().lower()
    if folder not in JOB_FOLDERS:
        raise GraphJobValidationError(
            f"graph job id={out['id']} folder={out.get('folder')!r}: one of {list(JOB_FOLDERS)} (run: a job of one run "
            "draws into that run's graphs folder; project: into the project's graphs/jobs/ like any other)")
    if folder == "project":
        out["folder"] = folder
    else:
        out.pop("folder", None)

    # Backward-compatible normalized alias for older downstream code.
    out["output_name"] = out["output_file_name"]

    out["source"] = str(out.get("source", "attack_metrics")).strip().lower()
    if out["source"] not in VALID_SOURCES:
        raise GraphJobValidationError(
            f"graph job id={out['id']} invalid source={out['source']!r}. "
            f"Allowed: {sorted(VALID_SOURCES)}"
        )
    if out["metric"].lower() in SOLO_ONLY_METRICS and str(out["type"]) != "solo":
        raise GraphJobValidationError(
            f"graph job id={out['id']} metric={out['metric']!r} draws one figure per "
            "model; use type: solo"
        )
    if out["metric"].lower() == "fold_rank_curves" and out["source"] != "cross_validation":
        raise GraphJobValidationError(
            f"graph job id={out['id']} metric='fold_rank_curves' requires source: cross_validation"
        )
    if out["metric"].lower() == "key_candidate_scores" and out["source"] != "key_recovery":
        raise GraphJobValidationError(
            f"graph job id={out['id']} metric='key_candidate_scores' requires source: key_recovery"
        )

    for removed in ("normalize_to", "paper_values"):
        if out.get(removed) not in (None, ""):
            raise GraphJobValidationError(
                f"graph job id={out['id']} {removed} was removed (2026-10-14): use relative: "
                "{rule: run | anchor, where: [model_filter conditions]}"
            )
        out.pop(removed, None)
    relative = normalize_relative_config(out)
    if relative is None:
        out.pop("relative", None)
    else:
        out["relative"] = relative
    metric_process = normalize_metric_process(out, combines)
    if metric_process is None:
        out.pop("metric_process", None)
    else:
        out["metric_process"] = metric_process

    groups = normalize_groups_config(out)
    if groups is None:
        out.pop("groups", None)
    else:
        out["groups"] = groups

    out["export"] = normalize_export_config(out)
    out["legend"] = normalize_legend_config(out)
    out["style"] = normalize_style_config(out)
    out["array_epoch"] = normalize_array_epoch_config(out)
    
    if "x_axis" in out:
        out["x_axis"] = str(out["x_axis"]).strip()

    if "y_axis" in out:
        out["y_axis"] = str(out["y_axis"]).strip()

    if not out["name"]:
        raise GraphJobValidationError(f"graph job id={out['id']} has empty name")

    if not out["metric"]:
        raise GraphJobValidationError(f"graph job id={out['id']} has empty metric")

    if not out["output_file_name"]:
        raise GraphJobValidationError(
            f"graph job id={out['id']} has empty output_file_name"
        )

    if out["type"] not in VALID_JOB_TYPES:
        raise GraphJobValidationError(
            f"graph job id={out['id']} invalid type={out['type']!r}. "
            f"Allowed: {sorted(VALID_JOB_TYPES)}"
        )

    fits = plot_types_for(out["type"], out["metrics"], out["array_epoch"].get("mode", "latest"))
    if out["plot_type"] not in fits:
        raise GraphJobValidationError(
            f"graph job id={out['id']} plot_type={out['plot_type']!r} does not change how "
            f"{', '.join(out['metrics'])} is drawn in a {out['type']} job. Allowed: {fits}"
        )
    if out["plot_type"] == "heatmap":
        check_heatmap_job(out)
    if out["plot_type"] == "strip":
        check_strip_job(out)

    # Remove deprecated plot block if user accidentally leaves it in YAML.
    out.pop("plot", None)

    if out["type"] == "solo":
        models = _require_non_empty_list(out, "models") if require_models else list(out.get("models") or [])
        processes = _require_non_empty_list(out, "processes")
        _validate_processes(out, processes)

        out["models"] = [str(m).strip() for m in models if str(m).strip()]
        out["processes"] = [normalize_process_name(str(p)) for p in processes]

        if not out["models"] and require_models:
            raise GraphJobValidationError(f"solo job id={out['id']} has no valid models")

    elif out["type"] == "process_overlay":
        models = _require_non_empty_list(out, "models") if require_models else list(out.get("models") or [])
        processes = _require_non_empty_list(out, "processes")
        _validate_processes(out, processes)

        out["models"] = [str(m).strip() for m in models if str(m).strip()]
        out["processes"] = [normalize_process_name(str(p)) for p in processes]

        if not out["models"] and require_models:
            raise GraphJobValidationError(
                f"process_overlay job id={out['id']} has no valid models"
            )

    elif out["type"] == "model_overlay":
        models = _require_non_empty_list(out, "models") if require_models else list(out.get("models") or [])
        processes = _require_non_empty_list(out, "processes")
        _validate_processes(out, processes)

        out["models"] = [str(m).strip() for m in models if str(m).strip()]
        out["processes"] = [normalize_process_name(str(p)) for p in processes]

        if not out["models"] and require_models:
            raise GraphJobValidationError(
                f"model_overlay job id={out['id']} has no valid models"
            )

    out["combines"] = combines
    out["processes"] = (out["processes"] if plain_processes else []) + list(combines)

    marks = normalize_marks_config(out)
    if marks is None:
        out.pop("marks", None)
    else:
        out["marks"] = marks

    for key, normalize in (("trace_budget", normalize_trace_budget), ("test", normalize_test_choice),
                           ("overlays", normalize_overlays_config)):
        value = normalize(out)
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value

    return out


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip()) or None
    return None


def normalize_trace_budget(job: Mapping[str, Any]) -> int | None:
    """trace_budget (trace_budget.py): the attack curves read at B traces per attack; None when unset."""
    from services.graph_manager.trace_budget import is_budget_metric

    raw = job.get("trace_budget")
    if raw in (None, ""):
        return None
    budget = _positive_int(raw)
    where = f"graph job id={job.get('id')} trace_budget"
    if budget is None:
        raise GraphJobValidationError(f"{where} must be a trace count >= 1, got {raw!r}")
    if str(job.get("source", "attack_metrics")) != "attack_metrics":
        raise GraphJobValidationError(f"{where} reads a run's attack curves: use source: attack_metrics")
    others = [m for m in job.get("metrics") or [job.get("metric")] if not is_budget_metric(route_metric(m).canonical_metric)]
    if others:
        raise GraphJobValidationError(
            f"{where} applies to the attack curves (mean_attack_rank_curve, per_attack_curves, full_ranks, "
            f"sr_curve) and final_ranks, not {others}"
        )
    if job.get("combines"):
        raise GraphJobValidationError(f"{where} does not apply to combined processes")
    if str((job.get("array_epoch") or {}).get("mode", "latest")) == "summary":
        raise GraphJobValidationError(f"{where} does not apply to array_epoch.mode: summary")
    return budget


def normalize_test_choice(job: Mapping[str, Any]) -> int | str | None:
    """test: which test the test process is drawn from (trace_budget.py): latest (empty too: None), budget,
    recorded, or a test's index."""
    from services.graph_manager.trace_budget import LATEST_TEST, TEST_CHOICES

    raw = job.get("test")
    if raw in (None, "") or str(raw).strip().lower() == LATEST_TEST:
        return None
    where = f"graph job id={job.get('id')} test"
    word = str(raw).strip().lower()
    index = None if word in TEST_CHOICES else _positive_int(raw)
    if word not in TEST_CHOICES and index is None:
        raise GraphJobValidationError(
            f"{where} is one of {list(TEST_CHOICES)} (latest: the latest All test) or a test's index >= 1, got {raw!r}"
        )
    if "test" not in (job.get("processes") or []):
        raise GraphJobValidationError(f"{where}: add test to processes (it chooses the test the test process reads)")
    if job.get("combines"):
        raise GraphJobValidationError(f"{where} does not apply to combined processes")
    return index if index is not None else word


def _validate_explanation_job(job: Mapping[str, Any], *, require_models: bool) -> dict[str, Any]:
    """A model_explanation job (services/graph_manager/explain.py): no metric, a method instead."""
    from services.graph_manager.explain import AGGREGATES, METHODS

    _require_keys(job, ["id", "name", "type", "enabled", "method"])
    out = dict(job)
    try:
        out["id"] = int(out["id"])
    except Exception as exc:
        raise GraphJobValidationError(f"graph job id must be int-like, got {job.get('id')!r}") from exc
    out["name"] = str(out["name"]).strip()
    out["method"] = str(out["method"]).strip()
    if out["method"] not in METHODS:
        raise GraphJobValidationError(f"graph job id={out['id']} method={out['method']!r}: use one of {list(METHODS)}")
    out["aggregate"] = str(out.get("aggregate", "mean_abs")).strip()
    if out["aggregate"] not in AGGREGATES:
        raise GraphJobValidationError(f"graph job id={out['id']} aggregate={out['aggregate']!r}: use one of {list(AGGREGATES)}")
    traces = out.get("traces")
    if traces is not None and (not isinstance(traces, int) or isinstance(traces, bool) or traces < 1):
        raise GraphJobValidationError(f"graph job id={out['id']} traces must be a positive count of test traces (or absent: all)")
    layers = out.get("layers")
    if layers is not None and (out["method"] != "heatmap" or not isinstance(layers, list)
                               or not all(isinstance(h, int) and h >= 1 for h in layers)):
        raise GraphJobValidationError(f"graph job id={out['id']} layers: a list of conv layer numbers (from 1), heatmap only")
    out["models"] = [str(m).strip() for m in (out.get("models") or []) if str(m).strip()]
    if require_models and not out["models"]:
        raise GraphJobValidationError(f"model_explanation job id={out['id']} has no valid models")
    out["output_file_name"] = normalize_output_file_name({**out, "metric": out["method"]})
    out["output_name"] = out["output_file_name"]
    return out


def validate_graph_jobs(jobs: Any, *, target_id: int | None = None) -> list[dict[str, Any]]:
    """
    Validate the full cfg['graph']['jobs'] list.

    Every job is validated; only the target job (all jobs when None) must
    resolve to at least one model, since top_models stays empty until the
    model judge has run.
    """

    if not isinstance(jobs, list) or not jobs:
        raise GraphJobValidationError("cfg['graph']['jobs'] must be a non-empty list")

    seen_ids: set[int] = set()
    normalized: list[dict[str, Any]] = []

    for job in jobs:
        require = True
        if isinstance(job, Mapping):
            require = target_id is None or str(job.get("id")) == str(target_id)
            models = expand_models(job)
            if not models and require:
                source = "top_models (written by the model judge: make judge)" if (
                    str(job.get("id")) == str(DEFAULT_JOB_ID)
                ) else (
                    "models (a model_name from artifacts/done_model_names.yaml or a "
                    "model_hash from artifacts/run_registry.json)"
                )
                raise GraphJobValidationError(
                    f"graph job id={job.get('id')} lists no models: fill {source}."
                )
            job = {**job, "models": models}
        checked = validate_graph_job(job, require_models=require)

        if checked["id"] in seen_ids:
            raise GraphJobValidationError(f"duplicate graph job id={checked['id']}")

        seen_ids.add(checked["id"])
        normalized.append(checked)

    return normalized


def get_graph_job_by_id(jobs: Any, graph_id: int) -> dict[str, Any]:
    """
    Validate jobs and return one normalized job by id.
    """

    graph_id = int(graph_id)
    normalized = validate_graph_jobs(jobs, target_id=graph_id)

    for job in normalized:
        if int(job["id"]) == graph_id:
            return job

    available = [job["id"] for job in normalized]
    raise GraphJobValidationError(
        f"No graph job found with id={graph_id}. Available IDs: {available}"
    )
