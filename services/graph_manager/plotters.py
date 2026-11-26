# services/graph_manager/plotters.py
"""
Generic plotters for TraceFlow GraphManager.

Handles:
- scalar line plots
- overlay scalar plots
- array curve plots
- distribution / bar plots
- plot_type dispatch
- paper-safe PNG/SVG/PDF/EPS export

SCA-specific attack curve helpers live in attack_plotters.py.

Author: Shraddha Mishra
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .plot_style import (
    apply_all_text_styles,
    apply_axes_style,
    apply_configured_text,
    apply_matplotlib_style,
    apply_text_object_style,
    get_bar_kwargs,
    get_line_kwargs,
    get_line_style,
    get_scatter_kwargs,
    get_std_band,
    get_style,
    get_text_for_target,
    get_text_rule,
    on_x_grid,
    parse_legend_handle,
    render_template,
)
from .svg_sanitizer import sanitize_and_validate_svg


POSITION_MAP = {
    "top-right": "upper right",
    "upper-right": "upper right",
    "top-left": "upper left",
    "upper-left": "upper left",
    "bottom-right": "lower right",
    "lower-right": "lower right",
    "bottom-left": "lower left",
    "lower-left": "lower left",
    "center": "center",
    "top": "upper center",       # inside the box at its top, centred (owner 2026-10-27: Fig. 24 a's long legend)
    "bottom": "lower center",    # inside the box at its bottom, centred (owner 2026-10-28: Fig. 4 II)
    "best": "best",
    "below": "upper center",     # outside the axes, under the x label (owner 2026-10-27: it covers no bar)
}


ALLOWED_EXPORT_FORMATS = {"png", "svg", "pdf", "eps"}
FORBIDDEN_EXPORT_FORMATS = {"jpg", "jpeg"}


def safe_stem(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "unnamed"

    clean = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            clean.append(ch)
        else:
            clean.append("_")

    return "".join(clean).strip("_") or "unnamed"


def normalize_formats(job: Mapping[str, Any]) -> list[str]:
    export_cfg = job.get("export", {}) or {}
    formats = export_cfg.get("formats", ["svg"])

    if isinstance(formats, str):
        formats = [formats]

    out: list[str] = []

    for fmt in formats:
        f = str(fmt).lower().strip().lstrip(".")

        if not f:
            continue

        if f in FORBIDDEN_EXPORT_FORMATS:
            raise ValueError(
                "JPG/JPEG export is forbidden for scientific graphs. "
                "Use PNG for preview and PDF/SVG/EPS for vector outputs."
            )

        if f not in ALLOWED_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported graph export format={f!r}. "
                f"Allowed: {sorted(ALLOWED_EXPORT_FORMATS)}"
            )

        if f not in out:
            out.append(f)

    return out or ["svg"]


def get_dpi(job: Mapping[str, Any]) -> int:
    """export.png.dpi: the resolution of the png (no other format uses one)."""
    png_cfg = (job.get("export", {}) or {}).get("png") or {}
    try:
        dpi = int(png_cfg.get("dpi", 300))
    except Exception:
        dpi = 300
    return max(dpi, 72)


def get_margin(job: Mapping[str, Any]) -> float:
    export_cfg = job.get("export", {}) or {}
    try:
        return max(float(export_cfg.get("margin", 0.05)), 0.0)
    except Exception:
        return 0.05


def get_plot_type(job: Mapping[str, Any]) -> str:
    plot_type = str(job.get("plot_type", "auto")).strip().lower()
    if not plot_type:
        return "auto"
    return plot_type


def legend_location(job: Mapping[str, Any]) -> str | None:
    legend_cfg = job.get("legend", {}) or {}

    if not bool(legend_cfg.get("enabled", True)):
        return None

    pos = str(legend_cfg.get("position", "best")).strip().lower()
    return POSITION_MAP.get(pos, "best")


def _handle_colour(handle: Any) -> Any:
    """Colour of a legend handle: a line, a bar group or a patch."""
    if hasattr(handle, "get_color"):
        return handle.get_color()
    patches = getattr(handle, "patches", None)
    if patches:
        return patches[0].get_facecolor()
    if hasattr(handle, "get_facecolor"):
        return handle.get_facecolor()
    return None


def legend_handles(job: Mapping[str, Any], handles: list[Any]) -> list[Any]:
    """
    legend.handle: [shape, thickness] redraws every legend entry in the colour
    of its series: "line" as a line thickness pt wide, a marker or character
    as that marker, thickness pt in size. Without it the handles stay as drawn.
    """
    from matplotlib.lines import Line2D

    spec = parse_legend_handle((job.get("legend") or {}).get("handle"))
    if spec is None:
        return handles
    out = []
    for handle in handles:
        colour = _handle_colour(handle)
        if colour is None:
            out.append(handle)
        elif spec["shape"] == "line":
            # keep the curve's line style (style.line.styles), so a dashed curve has a dashed entry
            style = handle.get_linestyle() if hasattr(handle, "get_linestyle") else "-"
            out.append(Line2D([], [], color=colour, linewidth=spec["size"], linestyle=style,
                              solid_capstyle="butt"))
        else:
            out.append(Line2D(
                [], [], color=colour, linestyle="none", marker=spec["marker"],
                markersize=spec["size"], markerfacecolor=colour, markeredgecolor=colour,
            ))
    return out


def legend_label(job: Mapping[str, Any], key: str, default: str) -> str:
    legend_cfg = job.get("legend", {}) or {}

    if not bool(legend_cfg.get("enabled", True)):
        return default

    mode = str(legend_cfg.get("mode", "auto")).strip().lower()
    labels = legend_cfg.get("labels", {}) or {}

    if mode == "custom":
        return str(labels.get(key, default))

    return default


def graph_title(job: Mapping[str, Any], context: Mapping[str, Any], fallback: str) -> str:
    """
    Backward-compatible title resolver.

    New schema should prefer style.text_rules target='title'.
    Old schema may still have:
      title:
        name: "..."
    """

    title_cfg = job.get("title", {}) or {}
    raw = title_cfg.get("name")

    if raw:
        return render_template(str(raw), context)

    return fallback


def below_offset(ax: Any, gap_in: float = 0.75) -> float:
    """The y (plot-box fraction) of a legend under the x label: gap_in inches below the box, whatever its height."""
    return -gap_in / max(ax.get_position().height * ax.figure.get_size_inches()[1], 0.1)


def tuck_below(ax: Any, legend: Any, pad_in: float = 0.1) -> None:
    """A legend below the axes moved up to pad_in inches under the x axis's tick labels and label, as drawn (owner
    2026-10-27: no space wasted between the stacked rows)."""
    renderer = ax.figure.canvas.get_renderer()
    bottom = ax.xaxis.get_tightbbox(renderer).y0 - pad_in * ax.figure.dpi
    legend.set_bbox_to_anchor((0.5, ax.transAxes.inverted().transform((0.0, bottom))[1]), transform=ax.transAxes)


def place_plot_box(fig: Any, margins: Mapping[str, float]) -> None:
    """style.figure.margins (inches): the plot box at those margins of the figure; a colour bar beside it, in the
    right margin (0.12 in away, 0.15 in wide, the box's height)."""
    width, height = fig.get_size_inches()
    boxes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    bars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    left, bottom = margins["left"] / width, margins["bottom"] / height
    w = max(width - margins["left"] - margins["right"], 0.1) / width
    h = max(height - margins["top"] - margins["bottom"], 0.1) / height
    for ax in boxes:
        ax.set_position([left, bottom, w, h])
    for bar in bars:
        bar.set_position([left + w + 0.12 / width, bottom, 0.15 / width, h])


def save_figure(fig: Any, out_base: Path, job: Mapping[str, Any]) -> list[Path]:
    """
    Save a Matplotlib figure using format-specific export policy.

    PNG:
      raster preview, uses dpi

    PDF:
      vector paper figure, no dpi needed for vector primitives

    SVG:
      vector/editable artifact, sanitized/validated after save

    EPS:
      optional vector artifact
    """

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # The page of every format is the drawn figure plus export.margin: no
    # paper size, no rotation, no other white space. style.figure.margins instead keeps the figure's exact size,
    # its plot box at those margins (place_plot_box), so panels of one row align in a facet.
    margins = ((job.get("style") or {}).get("figure") or {}).get("margins")
    if margins:                                               # the box was placed by _finalize_axes
        # the figure's whole page: bbox_inches None would mean rcParams savefig.bbox, which the style sets to tight
        from matplotlib.transforms import Bbox

        trim = {"bbox_inches": Bbox.from_bounds(0, 0, *fig.get_size_inches())}
    else:
        trim = {"bbox_inches": "tight", "pad_inches": get_margin(job)}

    try:
        for fmt in normalize_formats(job):
            path = out_base.with_suffix(f".{fmt}")

            if fmt == "png":
                fig.savefig(path, format="png", dpi=get_dpi(job), **trim)

            elif fmt == "pdf":
                fig.savefig(path, format="pdf", **trim)

            elif fmt == "svg":
                fig.savefig(path, format="svg", **trim)
                ok = sanitize_and_validate_svg(path, job)
                if not ok:
                    continue

            elif fmt == "eps":
                fig.savefig(path, format="eps", orientation="portrait", **trim)

            else:
                raise ValueError(f"Unsupported export format after normalization: {fmt}")

            written.append(path)

        # export.save_raw_plot_data: what this figure draws, beside it (raw_plot_data/<figure>.json)
        from services.graph_manager.raw_exporter import export_figure_data
        export_figure_data(fig, out_base, job)

    finally:
        plt.close(fig)

    return written


def _coerce_curve_array(data: Any) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        arr = data.to_numpy()
    else:
        arr = np.asarray(data)

    arr = np.squeeze(arr)

    if arr.ndim == 1:
        y = arr.astype(float)
        x = np.arange(1, len(y) + 1, dtype=float)
        return np.stack([x, y], axis=1)

    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            return arr[:, :2].astype(float)

        if arr.shape[0] >= 2:
            return arr[:2, :].T.astype(float)

    raise ValueError(f"Cannot coerce data into curve array, shape={arr.shape}")


def _draw_band(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    std: np.ndarray | None,
    *,
    job: Mapping[str, Any],
    color: str | None = None,
) -> None:
    """
    Shade the area between mean - std and mean + std (std over the
    cross-validation folds) in the colour of the series (color, else the last
    drawn line's), at style.std_band.alpha. Nothing is drawn when
    style.std_band.enabled is false or the data carry no std (then the loader
    did not compute it).
    """
    band = get_std_band(job)
    if std is None or not band["enabled"]:
        return
    x, y, std = on_x_grid(job, x, y, std)
    std = np.asarray(std, dtype=float)
    ok = np.isfinite(std)
    if not np.any(ok):
        return
    color = color or (ax.get_lines()[-1].get_color() if ax.get_lines() else None)
    ax.fill_between(x[ok], (y - std)[ok], (y + std)[ok], color=color, alpha=band["alpha"], linewidth=0)


def _std_column(df: pd.DataFrame, y_col: str) -> np.ndarray | None:
    col = f"{y_col}_std"
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _scalar_xy_std(df: pd.DataFrame, *, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = (
        pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        if x_col in df.columns else np.arange(1, len(df) + 1, dtype=float)
    )
    if y_col not in df.columns:
        raise KeyError(f"DataFrame missing y column: {y_col}")
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    std = _std_column(df, y_col)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], (std[mask] if std is not None else None)


def _curve_std(data: Any) -> np.ndarray | None:
    """Third column of a fold-mean curve [N, 3] = x, mean, std."""
    arr = np.squeeze(np.asarray(data)) if not isinstance(data, pd.DataFrame) else None
    if arr is not None and arr.ndim == 2 and arr.shape[1] == 3:
        return arr[:, 2].astype(float)
    return None


def slot_colour(index: int) -> str:
    """
    Colour of the series in slot ``index`` of an overlay: the index-th colour
    of the cycle, counted over the job's model order. A model that draws
    nothing keeps its slot, so the models after it keep their colours and a
    facet legend built from the job's order (faceting._job_entries) names the
    right run.
    """
    return f"C{index % 10}"


METRIC_HATCHES = ("", "////", "....", "xxxx")       # the bars of a job's 1st, 2nd, ... metric


def _legend_only(ax: Any, label: str, reason: str, colour: str) -> None:
    """A legend entry without a line for a series that drew nothing: "<label> (no value: <reason>)"."""
    ax.plot([], [], color=colour, linestyle="none", marker=None, label=f"{label} (no value: {reason})")


def _overlay_order(series: Mapping[str, Any], order: Any, dropped: Mapping[str, str] | None) -> list[str]:
    """The slots of an overlay: the job's order when given, then any series key not in it."""
    keys = [str(k) for k in (order or [])] or list(series)
    for key in list(series) + list(dropped or {}):
        if key not in keys:
            keys.append(key)
    return keys


def _draw_categorical_bars(
    ax: Any,
    labels: list[str],
    values: list[float],
    std: list[float | None],
    *,
    horizontal: bool = False,
    colours: list[str] | None = None,
    hatches: list[str] | None = None,
    value_format: str | None = None,
    value_size: float = 8,
) -> None:
    """
    One bar per series when every series holds a single value (e.g. test).

    Bar i takes colours[i] (its slot's colour of the colour cycle, the colour
    the same series has as a curve in an overlay of the same models), hatches[i]
    (a job of several metrics: one hatch per metric) and carries labels[i] as its
    legend entry (shown when legend.enabled is true). value_format (style.bar_values) prints each bar's value
    just past its end.
    """
    errors = [0.0 if e is None or not np.isfinite(e) else e for e in std]
    has_errors = any(errors)
    for i, (label, value, error) in enumerate(zip(labels, values, errors)):
        err = error if has_errors else None
        colour = colours[i] if colours else slot_colour(i)
        hatch = {"hatch": hatches[i], "edgecolor": "white", "linewidth": 0} if hatches and hatches[i] else {}
        if horizontal:
            ax.barh(i, value, xerr=err, capsize=3, height=0.6, color=colour, label=label, **hatch)
        else:
            ax.bar(i, value, yerr=err, capsize=3, width=0.6, color=colour, label=label, **hatch)
        if value_format and np.isfinite(value):
            text = " " + format(value, value_format)
            if horizontal:
                ax.text(value, i, text, ha="left", va="center", fontsize=value_size)
            else:
                ax.text(i, value, text.strip(), ha="center", va="bottom", fontsize=value_size)


def _set_category_ticks(ax: Any, n_bars: int, *, horizontal: bool = False) -> None:
    """
    One unlabelled tick slot per bar. The bars are named by the legend
    (legend.enabled) or by colour alone; the category axis carries no text
    besides its own label (style.text_rules target x_label / y_label).
    """
    # After apply_axes_style: set_xscale() resets tick locators and labels.
    positions = np.arange(n_bars)
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()  # first series on top, like the legend
        return
    ax.set_xticks(positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)


def _integer_epoch_ticks(ax: Any, x_col: str, job: Mapping[str, Any]) -> None:
    # style.axes.x_step, when set, keeps its ticks (owner 2026-10-27: Fig. 26 b, a tick every 200 epochs)
    if x_col == "epoch" and not ((job.get("style") or {}).get("axes") or {}).get("x_step"):
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _draw_xy(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    job: Mapping[str, Any],
    label: str | None = None,
    allow_bar: bool = True,
    series: str | None = None,
    color: str | None = None,
) -> None:
    """
    Draw one x/y series using job['plot_type'].

    For unsupported plot_type in a curve context, falls back to line.
    series names the curve for style.line.styles; color, when given, is the
    series' slot colour (else the colour cycle's next).
    """

    plot_type = get_plot_type(job)
    x, y = on_x_grid(job, x, y)

    if plot_type == "auto":
        plot_type = "line"

    colour = {"color": color} if color else {}
    line_kwargs = {**get_line_kwargs(job), **colour}
    linestyle = get_line_style(job, series)
    if linestyle:
        line_kwargs["linestyle"] = linestyle
    look = (((job.get("style") or {}).get("series") or {}).get(series) or {}) if series else {}
    if look.get("width"):
        line_kwargs["linewidth"] = look["width"]           # style.series: this line's width
    if look.get("dashes"):
        line_kwargs["linestyle"] = (0, tuple(look["dashes"]))  # style.series: its [on, off] dash pattern

    if plot_type == "line":
        ax.plot(x, y, label=label, **line_kwargs)

    elif plot_type == "line_scatter":
        ax.plot(x, y, label=label, **line_kwargs)

    elif plot_type == "scatter":
        ax.scatter(x, y, label=label, **get_scatter_kwargs(job), **colour)

    elif plot_type == "step":
        kwargs = {**get_line_kwargs(job), **colour}
        ax.step(x, y, where="post", label=label, **kwargs)

    elif plot_type == "bar" and allow_bar:
        ax.bar(x, y, label=label, **get_bar_kwargs(job), **colour)

    else:
        print(
            f"[GraphManager][WARN] plot_type={plot_type!r} is not supported "
            "for this plotter; falling back to line."
        )
        ax.plot(x, y, label=label, **get_line_kwargs(job), **colour)


def _finalize_axes(
    ax: Any,
    *,
    job: Mapping[str, Any],
    context: Mapping[str, Any],
    default_title: str,
    default_x_label: str,
    default_y_label: str,
) -> None:
    apply_axes_style(ax, job)
    margins = ((job.get("style") or {}).get("figure") or {}).get("margins")
    if margins:                                               # style.figure.margins: the box placed first
        place_plot_box(ax.figure, margins)

    loc = legend_location(job)
    handles, labels = ax.get_legend_handles_labels()
    entries = (job.get("legend") or {}).get("entries") or []
    below = str((job.get("legend") or {}).get("position", "")).strip().lower() == "below"
    anchor = (job.get("legend") or {}).get("anchor")          # legend.anchor: [x, y] in the plot box
    place = {"loc": loc, "ncol": int((job.get("legend") or {}).get("columns") or 1),
             **({"bbox_to_anchor": (0.5, below_offset(ax)), "frameon": False} if below else {}),
             **({"bbox_to_anchor": tuple(anchor)} if anchor else {})}
    if loc and entries:
        # legend.entries: the legend's own patches (or lines, with a linestyle) in place of one per series
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        def line(e):
            style = (0, tuple(e["dashes"])) if e.get("dashes") else e.get("linestyle") or "-"
            return Line2D([], [], color=e["color"], linestyle=style, linewidth=e.get("width") or 1.2)

        legend_args = ([line(e) if e.get("linestyle") or e.get("dashes")
                        else Patch(facecolor=e["color"], hatch=e["hatch"], edgecolor="white", linewidth=0)
                        for e in entries], [e["label"] for e in entries])
    elif loc and handles:
        legend_args = (legend_handles(job, handles), labels)
    else:
        legend_args = None
    if legend_args:
        ax.legend(*legend_args, **place)

    # Job context (model, process, source, ...) fills gaps in the plotter's.
    context = {**(job.get("active_context") or {}), **{k: v for k, v in context.items() if v not in (None, "")}}

    apply_configured_text(
        ax,
        job,
        context=context,
        default_title=default_title,
        default_x_label=default_x_label,
        default_y_label=default_y_label,
    )

    # apply_configured_text already calls apply_all_text_styles(), but this is
    # harmless and ensures legend text is styled after legend creation.
    apply_all_text_styles(ax, job)
    if legend_args and not below and not anchor:
        fit_legend_in_box(ax, lambda **extra: ax.legend(*legend_args, **place, **extra))
    if legend_args and below:
        tuck_below(ax, ax.get_legend())


def fit_legend_in_box(ax: Any, make_legend: Any, minimum: float = 6.0) -> None:
    """
    A legend inside the plot box no wider than the box (owner 2026-10-27: legends inside the figure, at its top or
    top-right corner): one wider is made again (`make_legend`) at the size that fits, its marks and spacing (font
    units) shrinking with its text, never below `minimum` points.
    """
    legend = ax.get_legend()
    if legend is None or not legend.get_texts():
        return
    renderer = ax.figure.canvas.get_renderer()
    box = ax.get_window_extent(renderer).width - 4.0      # pixels: the frame clear of the box's edges
    text = legend.get_texts()[0]
    prop, colour = text.get_fontproperties().copy(), text.get_color()
    size = prop.get_size_in_points()
    # first shorter marks and narrower gaps at the full text size, then smaller text (owner 2026-10-28: larger
    # legend text in Fig. 5.i V a and VI a)
    tight = {"handlelength": 1.2, "columnspacing": 0.8, "handletextpad": 0.4}
    if legend.get_window_extent(renderer).width > box:
        legend = make_legend(prop=prop, **tight)
        for t in legend.get_texts():
            t.set_color(colour)
    for _ in range(8):
        wide = legend.get_window_extent(renderer).width
        if wide <= box or size <= minimum:
            return
        size = max(minimum, math.floor(2.0 * size * box / wide) / 2.0)
        prop.set_size(size)
        legend = make_legend(prop=prop, **tight)
        for t in legend.get_texts():
            t.set_color(colour)


def plot_scalar_line(
    *,
    df: pd.DataFrame,
    job: Mapping[str, Any],
    out_base: Path,
    metric: str,
    model: str,
    process: str,
    x_col: str = "epoch",
    y_col: str | None = None,
    title: str | None = None,
) -> list[Path]:
    apply_matplotlib_style(job)

    y_col = y_col or metric
    x, y, std = _scalar_xy_std(df, x_col=x_col, y_col=y_col)

    if len(x) == 0:
        print(
            f"[GraphManager][WARN] No finite scalar data to plot: "
            f"metric={metric} model={model} process={process}"
        )
        return []

    fig, ax = plt.subplots()

    label_key = f"{model}:{process}:{metric}"
    # Also allow custom labels by raw model name for ablation-style configs.
    label = legend_label(job, model, legend_label(job, label_key, f"{model} / {process}"))

    _draw_xy(ax, x, y, job=job, label=label)
    _draw_band(ax, x, y, std, job=job)

    context = {
        "metric": metric,
        "model": model,
        "model_name": model,
        "process": process,
        "x_axis": x_col,
        "y_axis": y_col,
    }

    _finalize_axes(
        ax,
        job=job,
        context=context,
        default_title=title
        or graph_title(job, context, fallback=f"{metric} over {x_col}"),
        default_x_label=x_col,
        default_y_label=y_col,
    )
    _integer_epoch_ticks(ax, x_col, job)

    return save_figure(fig, out_base, job)


def plot_scalar_overlay(
    *,
    series: Mapping[str, pd.DataFrame],
    job: Mapping[str, Any],
    out_base: Path,
    metric: str,
    x_col: str = "epoch",
    y_col: str | None = None,
    title: str | None = None,
    order: Any = None,
    dropped: Mapping[str, str] | None = None,
    groups: list[Mapping[str, Any]] | None = None,
) -> list[Path]:
    """
    One curve per series, or one bar per series when each holds one value.
    With the job's relative: block the values are ratios to its reference run,
    and a dashed line marks 1.0. With ``groups`` (the job's), each group is one
    series per metric (_draw_groups: the mean over its runs at each epoch and
    its band), coloured by its first run's style.series colour of that metric;
    the runs in no group are drawn as usual.

    Colours follow the slots of ``order`` (the job's model order; default: the
    series' order). A series that draws nothing, one of ``dropped`` (key ->
    reason, e.g. a metric the run never recorded) or one with no finite value,
    keeps its slot: the others keep their colours and it stays in the legend
    as "<label> (no value: <reason>)".
    """
    apply_matplotlib_style(job)

    y_col = y_col or metric

    fig, ax = plt.subplots()

    slots = _overlay_order(series, order, dropped)
    points: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    reasons: dict[str, str] = dict(dropped or {})
    for label_key in slots:
        if label_key in reasons or label_key not in series:
            reasons.setdefault(label_key, "not recorded")
            continue
        try:
            x, y, std = _scalar_xy_std(series[label_key], x_col=x_col, y_col=y_col)
        except Exception as exc:
            print(f"[GraphManager][WARN] Skipping overlay series={label_key}: {exc}")
            reasons[label_key] = "unreadable"
            continue
        if len(x) == 0:
            print(f"[GraphManager][WARN] No finite {y_col} values for series={label_key}")
            reasons[label_key] = f"NaN every {x_col}"
            continue
        points[label_key] = (x, y, std)
    colour_of = {key: slot_colour(i) for i, key in enumerate(slots)}
    series_style = (job.get("style") or {}).get("series") or {}     # style.series: a series' own colour and hatch
    # several metrics (keys "<model>:<metric>", owner 2026-10-27): a model keeps its colour, the one it has in the
    # job's other figures of the same models, and a bar its metric's hatch, solid for the job's first metric
    metrics = [str(m) for m in job.get("metrics") or []]
    hatch_of: dict[str, str] = {}
    if len(metrics) > 1:
        models = list(dict.fromkeys(key.partition(":")[0] for key in slots))
        colour_of = {key: slot_colour(models.index(key.partition(":")[0])) for key in slots}
        hatch_of = {key: METRIC_HATCHES[metrics.index(key.partition(":")[2]) % len(METRIC_HATCHES)]
                    for key in slots if key.partition(":")[2] in metrics}
    for key in slots:
        spec = series_style.get(key) or {}
        if spec.get("color"):
            colour_of[key] = spec["color"]
        if "hatch" in spec:
            hatch_of[key] = spec["hatch"]

    plotted = len(points)
    single_values = plotted > 0 and all(len(x) == 1 for x, _, _ in points.values())
    horizontal = single_values and get_plot_type(job) == "barh"

    if groups and single_values:
        raise ValueError(f"groups pool curves; {metric} holds one value per run: draw it with plot_type: strip")
    if single_values:
        # One value per model (test, key recovery): a bar per model, not dots.
        labels = [legend_label(job, key, key) for key in points]
        _draw_categorical_bars(
            ax,
            labels,
            [float(y[0]) for _, y, _ in points.values()],
            [float(std[0]) if std is not None else None for _, _, std in points.values()],
            horizontal=horizontal,
            colours=[colour_of[key] for key in points],
            hatches=[hatch_of.get(key, "") for key in points] if hatch_of else None,
            value_format=(job.get("style") or {}).get("bar_values"),
            value_size=(get_style(job).get("font") or {}).get("tick_size", 8),
        )
        for label_key, reason in reasons.items():
            _legend_only(ax, legend_label(job, label_key, label_key), reason, colour_of[label_key])
        x_col = "model"
    else:
        labels = []
        grouped: set[str] = set()
        for index, group in enumerate(groups or []):
            for m, one in enumerate(metrics if len(metrics) > 1 else [""]):
                keys = [f"{run}:{one}" if one else run for run in group["models"]]
                grouped.update(keys)
                colour = (series_style.get(keys[0]) or {}).get("color") or slot_colour(index * max(len(metrics), 1) + m)
                curves = {key: np.column_stack([points[key][0], points[key][1]]) for key in keys if key in points}
                _draw_groups(ax, curves, [{**group, "label": f"{group['label']}: {one}" if one else group["label"],
                                           "models": keys}], job, reasons, colour=colour)
        for label_key in slots:
            if label_key in grouped:
                continue
            # model_overlay uses model names as keys; custom labels map
            # model name -> paper label.
            label = legend_label(job, label_key, label_key)
            if label_key in reasons:
                _legend_only(ax, label, reasons[label_key], colour_of[label_key])
                continue
            x, y, std = points[label_key]
            _draw_xy(ax, x, y, job=job, label=label, series=label_key, color=colour_of[label_key])
            _draw_band(ax, x, y, std, job=job, color=colour_of[label_key])

    if plotted == 0:
        plt.close(fig)
        print(f"[GraphManager][WARN] No scalar overlay data plotted for metric={metric}")
        return []

    if job.get("relative"):
        line = ax.axvline if horizontal else ax.axhline
        line(1.0, color="0.35", linestyle="--", linewidth=0.8, zorder=0)

    context = {
        "metric": metric,
        "process": job.get("process", ""),
        "x_axis": x_col,
        "y_axis": y_col,
    }

    _finalize_axes(
        ax,
        job=job,
        context=context,
        default_title=title
        or graph_title(job, context, fallback=f"{metric} comparison"),
        default_x_label=y_col if horizontal else x_col,
        default_y_label=x_col if horizontal else y_col,
    )
    if labels:
        _set_category_ticks(ax, len(labels), horizontal=horizontal)
    _integer_epoch_ticks(ax, x_col, job)

    return save_figure(fig, out_base, job)


# plot_type strip (graph_job_validator.check_strip_job): a run's dot by whether its best key hypothesis is the true
# key (the Model Performance Report's key confidence: teal recovered, red not), grey when its record names neither
STRIP_DOTS = {True: ("#00897B", "the best key is the true key"), False: ("#E53935", "it is not"),
              None: ("0.6", "the record names no best key")}
STRIP_OTHER_ROW = "other runs"       # with groups: the row of the job's runs in no group


def plot_strip(
    *,
    series: Mapping[str, pd.DataFrame],
    job: Mapping[str, Any],
    out_base: Path,
    metric: str,
    groups: list[Mapping[str, Any]] | None = None,
    order: Any = None,
    dropped: Mapping[str, str] | None = None,
) -> list[Path]:
    """
    plot_type strip: a dot per run at x = log10(1 + its score margin) (the last record of each series: the test's
    key-recovery report, top1_top2_score_margin with best_is_key, metric_loader._load_key_recovery_dir), coloured by
    STRIP_DOTS. With ``groups`` one row per group, named by its label (then a row of the runs in no group); else one
    row of all the job's runs, in ``order``. A run without a margin is left out with a warning.
    """
    apply_matplotlib_style(job)

    values: dict[str, tuple[float, bool | None]] = {}
    for key in _overlay_order(series, order, dropped):
        if key in (dropped or {}) or key not in series:
            continue
        frame = series[key]
        frame = frame[pd.to_numeric(frame[metric], errors="coerce").notna()] if metric in frame else frame.iloc[0:0]
        if frame.empty:
            print(f"[GraphManager][WARN] No finite {metric} for series={key}: left out of the strip")
            continue
        last = frame.iloc[-1]
        best = last.get("best_is_key")
        values[key] = (float(last[metric]), None if best is None or pd.isna(best) else bool(best))
    if not values:
        print(f"[GraphManager][WARN] No strip data plotted for metric={metric}")
        return []

    if groups:
        grouped = {key for group in groups for key in group["models"]}
        rows = [(group["label"], [key for key in group["models"] if key in values]) for group in groups]
        rest = [key for key in values if key not in grouped]
        rows += [(STRIP_OTHER_ROW, rest)] if rest else []
    else:
        rows = [("", list(values))]

    fig, ax = plt.subplots()
    used = set()
    for row, (_, members) in enumerate(rows):
        for key in members:
            margin, best = values[key]
            ax.plot(np.log10(1.0 + max(margin, 0.0)), row, linestyle="none", marker="o", markersize=6,
                    color=STRIP_DOTS[best][0], markeredgecolor="none", alpha=0.75, zorder=3)
            used.add(best)
    for best in (True, False, None):
        if best in used:
            ax.plot([], [], linestyle="none", marker="o", markersize=6, color=STRIP_DOTS[best][0],
                    label=STRIP_DOTS[best][1])

    context = {"metric": metric, "process": job.get("process", ""), "x_axis": "log(1 + score margin)", "y_axis": ""}
    _finalize_axes(
        ax,
        job=job,
        context=context,
        default_title=graph_title(job, context, fallback="Key confidence"),
        default_x_label="log(1 + score margin)",
        default_y_label="",
    )
    # after apply_axes_style (it resets the tick locators): a row per group, the first on top
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.5)

    return save_figure(fig, out_base, job)


NEVER_COLOUR = "0.85"      # a heatmap cell without a value: light grey, off the colour scale
# the colour bar's label of a heatmap metric (GraphManager.run_fold_heatmap)
HEATMAP_LABELS = {
    "stable_rank1_trace": "traces until the key stays at rank 1",
    "first_rank1_trace": "traces until the key first reaches rank 1",
    "final_true_key_rank": "final key rank",
}


def _draw_row_groups(ax: Any, groups: Sequence[Sequence[str]], rule: Mapping[str, Any], size: float,
                     colours: Sequence[Any] = ()) -> None:
    """
    style.heatmap.row_groups: each run of rows of one group ("HW, 20" ... "HW, 800") gets a bracket left of its
    row names and the group's name beside it, turned upright (owner 2026-10-28: Fig. 26 c, HW and ID over the
    epochs); the name in its rows' colour when they share one (Fig. 5.ii I to III: the model's), else black.
    """
    runs: list[list[Any]] = []
    for i, (group, _) in enumerate(groups):
        if group and runs and runs[-1][0] == group and runs[-1][2] == i - 1:
            runs[-1][2] = i
        elif group:
            runs.append([group, i, i])
    if not runs:
        return
    renderer = ax.figure.canvas.get_renderer()
    left = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels() if t.get_text())
    x = ax.transAxes.inverted().transform((left - 0.06 * ax.figure.dpi, 0.0))[0]       # 0.06 in left of the names
    blend = ax.get_yaxis_transform()
    for group, first, last in runs:
        shared = {str(c) for c in colours[first:last + 1]}
        colour = colours[first] if len(shared) == 1 and len(colours) > last else "black"
        ax.plot([x, x], [first - 0.4, last + 0.4], transform=blend, color="black", linewidth=0.8, clip_on=False)
        ax.text(x - 0.012, (first + last) / 2.0, group, transform=blend, rotation=90, ha="right", va="center",
                fontsize=size, fontweight="bold" if rule.get("bold") else "normal", color=colour, clip_on=False)


def plot_fold_heatmap(
    *,
    values: Sequence[Sequence[float]],
    notes: Sequence[Sequence[str]],
    rows: Sequence[str],
    colours: Sequence[str],
    folds: Sequence[int],
    job: Mapping[str, Any],
    out_base: Path,
    metric: str,
) -> list[Path]:
    """
    A model per row, a cross-validation fold per column (GraphManager.run_fold_heatmap): a cell is coloured by its
    value on one scale from 0 (style.colormap, default viridis) and printed with it; a cell without a value (the
    key never stays at rank 1) is light grey, off the scale, printed with the last line of its note (the fold's
    final key rank), and the legend under the axes says so (owner 2026-10-27). A row's label takes the model's
    colour in the job's other figures; a heat map without a coloured cell has no colour bar.
    """
    apply_matplotlib_style(job)
    fig, ax = plt.subplots()
    grid = np.asarray(values, dtype=float)
    finite = grid[np.isfinite(grid)]
    # style.heatmap (owner 2026-10-28, Fig. 5.ii): limits one scale for several heat maps (else 0 to the largest
    # value), colorbar, cell_values the cells' number format, row_groups "group, item" row names grouped
    heat = (job.get("style") or {}).get("heatmap") or {}
    low, top = heat.get("limits") or (0.0, float(finite.max()) if finite.size else 1.0)
    cell_format = heat.get("cell_values") or ",.0f"
    # style.colormap as the validator wrote it (plot_style.get_style rebuilds the style without it)
    cmap = plt.get_cmap((job.get("style") or {}).get("colormap") or "viridis").copy()
    cmap.set_bad(NEVER_COLOUR)
    never = ~np.isfinite(grid)
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, aspect="auto", vmin=low, vmax=top)
    size = (get_style(job).get("font") or {}).get("tick_size", 8)
    for (i, j), value in np.ndenumerate(grid):
        shade = (0.85, 0.85, 0.85) if never[i, j] else image.cmap(image.norm(value))
        light = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2] > 0.5
        # a grey cell: the rank alone (owner 2026-10-27: Fig. 5.ii, one look in all six heat maps)
        ax.text(j, i, notes[i][j].split("\n")[-1] if never[i, j] else format(value, cell_format), ha="center",
                va="center", fontsize=size, linespacing=1.0, color="black" if light else "white")
    # no colour bar without a coloured cell: its 0 to 1 scale would name nothing (Fig. 10: no fold recovers);
    # style.heatmap.colorbar false: none (a facet's legend draws one spectrum for all)
    bar = fig.colorbar(image, ax=ax) if finite.size and heat.get("colorbar", True) else None
    if bar is not None:
        # text_rules target colorbar_label: its text ("" none: the title names the quantity) and look
        bar.set_label(get_text_for_target(job, "colorbar_label", default=HEATMAP_LABELS.get(metric, metric)))
        apply_text_object_style(bar.ax.yaxis.label, job, "colorbar_label")
    _finalize_axes(ax, job=job, context={"metric": metric, "process": "validation"},
                   default_title=graph_title(job, {"metric": metric}, fallback=f"{metric} per fold"),
                   default_x_label="fold", default_y_label="")
    ax.set_xticks(range(len(folds)), [str(f) for f in folds])
    groups = [r.split(", ", 1) if heat.get("row_groups") and ", " in r else ["", r] for r in rows]
    ax.set_yticks(range(len(rows)), [item for _, item in groups])
    # the row names are the heat map's y labels: the y_label rule's weight (owner 2026-10-28: bold)
    y_rule = get_text_rule(job, "y_label") or {}
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold" if y_rule.get("bold") else "normal")
    # every cell's text at one size that fits its cell, at most the tick size (owner 2026-10-27: larger figure text)
    bbox = ax.get_position()
    cell_w = bbox.width * fig.get_size_inches()[0] * 72.0 / max(len(folds), 1)
    cell_h = bbox.height * fig.get_size_inches()[1] * 72.0 / max(len(rows), 1)
    cell_size = min([size] + [min(cell_h / 1.25, cell_w / (0.58 * len(t.get_text()) + 1e-9)) for t in ax.texts])
    for t in ax.texts:
        t.set_fontsize(cell_size)
    for tick, colour in zip(ax.get_yticklabels(), colours):
        tick.set_color(colour)
    # a colour-bar label longer than its bar on as few lines as fit it, at most four, its words spread evenly
    # (owner 2026-10-27: Fig. 26 c's label ran past the title; Fig. 5.ii's short heat maps)
    renderer = fig.canvas.get_renderer()
    label = bar.ax.yaxis.label if bar is not None else None
    words = label.get_text().split(" ") if label is not None else []
    for lines in range(2, min(4, len(words)) + 1):
        if label.get_window_extent(renderer).height <= bar.ax.get_window_extent(renderer).height:
            break
        target = len(" ".join(words)) / lines
        parts, part = [], []
        for word in words:
            longer, now = len(" ".join(part + [word])), len(" ".join(part))
            if part and abs(longer - target) > abs(now - target) and len(parts) < lines - 1:
                parts.append(part)
                part = []
            part.append(word)
        label.set_text("\n".join(" ".join(r) for r in parts + [part]))
    ax.grid(False)
    _draw_row_groups(ax, groups, y_rule, size, colours)
    if never.any() and (job.get("legend") or {}).get("enabled", True):
        # what a grey cell means, under the axes (owner 2026-10-27): it covers no cell; legend.enabled false leaves
        # it out (a facet row whose heat maps share one panel's legend)
        from matplotlib.patches import Patch

        ax.legend([Patch(facecolor=NEVER_COLOUR, edgecolor="none")],
                  ["key never stays at rank 1: the fold's final key rank"],
                  loc="upper center", bbox_to_anchor=(0.5, below_offset(ax)), frameon=False, fontsize=size, handlelength=1.4)
        tuck_below(ax, ax.get_legend())
    return save_figure(fig, out_base, job)


def plot_curve(
    *,
    data: Any,
    job: Mapping[str, Any],
    out_base: Path,
    metric: str,
    model: str,
    process: str,
    x_label: str,
    y_label: str,
    title: str | None = None,
) -> list[Path]:
    apply_matplotlib_style(job)

    try:
        curve = _coerce_curve_array(data)
    except Exception as exc:
        print(
            f"[GraphManager][WARN] Cannot plot curve: metric={metric} "
            f"model={model} process={process}: {exc}"
        )
        return []

    x = curve[:, 0]
    y = curve[:, 1]
    mask = np.isfinite(x) & np.isfinite(y)

    if not np.any(mask):
        print(
            f"[GraphManager][WARN] No finite curve data: "
            f"metric={metric} model={model} process={process}"
        )
        return []

    fig, ax = plt.subplots()

    label_key = f"{model}:{process}:{metric}"
    label = legend_label(job, model, legend_label(job, label_key, f"{model} / {process}"))

    _draw_xy(ax, x[mask], y[mask], job=job, label=label)
    std = _curve_std(data)
    _draw_band(ax, x[mask], y[mask], std[mask] if std is not None else None, job=job)

    context = {
        "metric": metric,
        "model": model,
        "model_name": model,
        "process": process,
        "x_axis": x_label,
        "y_axis": y_label,
    }

    _finalize_axes(
        ax,
        job=job,
        context=context,
        default_title=title
        or graph_title(job, context, fallback=f"{metric} curve"),
        default_x_label=x_label,
        default_y_label=y_label,
    )

    return save_figure(fig, out_base, job)


_BAND_TEXT = {"min_max": "band: min to max", "std": "band: mean \u00b1 1 std", "none": ""}


def _draw_groups(ax: Any, series: Mapping[str, Any], groups: list[Mapping[str, Any]], job: Mapping[str, Any],
                 reasons: Mapping[str, str], colour: str | None = None) -> int:
    """
    The job's groups (graph_job_validator.normalize_groups_config), one colour each (``colour``, else the
    group's slot): every member run's curve thin (unless the group's members: false), their mean bold (the
    legend entry: "<label> (mean of n runs; band: ...)"), and the band the group names (min to max, or mean +-
    the sample std within the runs' min to max) shaded at style.std_band.alpha. The members must share their x values (runs attacked on
    the same test traces, or trained for the same epochs). Returns the number of groups drawn.
    """
    drawn = 0
    alpha = get_std_band(job)["alpha"]
    given = colour
    for index, group in enumerate(groups):
        colour = given or group.get("color") or slot_colour(index)
        band_colour = group.get("band_color") or colour
        label, band = group["label"], group["band"]
        curves = []
        for key in group["models"]:
            if key in reasons or key not in series:
                continue
            curve = _coerce_curve_array(series[key])
            curves.append((key, curve[:, 0], curve[:, 1]))
        if not curves:
            _legend_only(ax, label, "no run recorded", colour)
            continue
        x = curves[0][1]
        for key, other_x, _ in curves[1:]:
            if other_x.shape != x.shape or not np.array_equal(other_x, x):
                raise ValueError(
                    f"group {label!r}: run {key} has other x values than {curves[0][0]} (the number of traces); "
                    "a group pools runs attacked on the same test traces")
        ys = np.vstack([y for _, _, y in curves]).astype(float)
        for y in ys if group.get("members", True) else ():
            ax.plot(x, y, color=colour, linewidth=0.8, alpha=0.35)
        mean = np.nanmean(ys, axis=0)
        missing = len(group["models"]) - len(curves)
        parts = [f"mean of {len(curves)} run{'s' if len(curves) != 1 else ''}"
                 + (f", {missing} not recorded" if missing else "")]
        if _BAND_TEXT[band]:
            parts.append(_BAND_TEXT[band])
        ok = np.isfinite(x) & np.isfinite(mean)
        _draw_xy(ax, x[ok], mean[ok], job=job, label=f"{label} ({'; '.join(parts)})", allow_bar=False, color=colour)
        if band == "min_max":
            low, high = np.nanmin(ys, axis=0), np.nanmax(ys, axis=0)
        elif band == "std" and len(curves) > 1:
            # within the runs' own range: a GE band below rank 1 claims a value no run had
            spread = np.nanstd(ys, axis=0, ddof=1)
            low = np.maximum(mean - spread, np.nanmin(ys, axis=0))
            high = np.minimum(mean + spread, np.nanmax(ys, axis=0))
        else:
            low = high = None
        if low is not None:
            keep = ok & np.isfinite(low) & np.isfinite(high)
            ax.fill_between(x[keep], low[keep], high[keep], color=band_colour, alpha=alpha, linewidth=0)
        drawn += 1
    return drawn


# marks (graph job, graph_job_validator.normalize_marks_config): marker, colour (the TraceFlow Monitor's Done
# ranks), legend text
MARK_STYLES = {
    "validation_final": ("o", "#f5b301", "validation final"),
    "test_at_validation": ("D", "#ff9a3d", "test at the same traces"),
    "test_final": ("s", "#ff5a1f", "test final"),
}


def rank_marks(validation: Any, test: Any, marks: Sequence[str]) -> list[dict[str, float]]:
    """The marked points of a run's validation and test rank curves [traces, rank]: validation_final its last
    point, test_at_validation the test curve's last point at or before the validation's trace count (none when
    the test is shorter), test_final the test's last point. A mark whose curve is missing is left out."""
    def finite(data):
        if data is None:
            return None
        curve = _coerce_curve_array(data)
        curve = curve[np.isfinite(curve[:, 0]) & np.isfinite(curve[:, 1])]
        return curve if len(curve) else None

    val, tst = finite(validation), finite(test)
    out = []
    for mark in marks:
        point = None
        if mark == "validation_final" and val is not None:
            point = val[-1]
        elif mark == "test_final" and tst is not None:
            point = tst[-1]
        elif mark == "test_at_validation" and val is not None and tst is not None and tst[-1, 0] >= val[-1, 0]:
            point = tst[np.searchsorted(tst[:, 0], val[-1, 0], side="right") - 1]
        if point is not None:
            out.append({"mark": mark, "traces": float(point[0]), "rank": float(point[1])})
    return out


def _draw_rank_marks(ax: Any, series: Mapping[str, Any], marks: Sequence[str]) -> None:
    """The job's marks on its validation and test curves, each in the legend with its rank and trace count;
    a dotted line at the validation's trace count, where the first two meet."""
    points = rank_marks(series.get("validation"), series.get("test"), marks)
    at = [p["traces"] for p in points if p["mark"] in ("validation_final", "test_at_validation")]
    if at:
        ax.axvline(at[0], color="grey", linestyle=":", linewidth=0.8, zorder=1)
    for p in points:
        marker, colour, text = MARK_STYLES[p["mark"]]
        rank = f"{p['rank']:g}" if float(p["rank"]).is_integer() else f"{p['rank']:.4g}"
        ax.plot(p["traces"], p["rank"], linestyle="none", marker=marker, markersize=8, color=colour,
                markeredgecolor="black", markeredgewidth=0.6, zorder=5,
                label=f"{text}: {rank} @ {int(round(p['traces'])):,}")


# overlays (graph job, graph_job_validator.normalize_overlays_config; TODO.md section 29, the Model Performance
# Report's GE graph): the averages' dashes, the N̄tGE marks' median colour, and the trace lines in the TraceFlow
# Monitor's rank key (webview/panel.css .rk: "test, first N_val traces" and "test, budget N_test traces")
GE_AVERAGE_STYLES = {
    "geometric": ((0, (3, 1.5)), "geometric mean over the attacks"),
    "median": ((0, (4, 1.2, 1, 1.2)), "median over the attacks"),     # Wu, Perin, Picek, COSADE 2022 recommend it
}
NTGE_MEDIAN_COLOUR = "#E53935"
TRACE_LINE_STYLES = {
    "n_val": ("#9a4a00", (0, (1.5, 1.5)), "$N_\\mathrm{val}$: validation traces"),
    "budget": ("#c00000", (0, (3, 1.5)), "$N_\\mathrm{budget}$: traces per Budget-test attack"),
}
OVERLAY_KEY_COLOUR = "0.3"         # a legend entry of an overlay: its mark or dash, not a run's colour


def ge_average_curves(per_attack: Any) -> dict[str, np.ndarray]:
    """A run's per-attack rank curves [attacks, traces, 2] (trace count, rank) averaged over the attacks at each trace
    count, as [traces, 2] curves: geometric exp(mean(log(max(rank, 1)))) and median."""
    curves = np.asarray(per_attack, dtype=float)
    if curves.ndim != 3 or curves.shape[2] != 2 or curves.shape[0] == 0:
        raise ValueError(f"per_attack_curves are [attacks, traces, 2], got shape {curves.shape}")
    x, ranks = curves[0, :, 0], curves[:, :, 1]
    return {"geometric": np.column_stack([x, np.exp(np.mean(np.log(np.maximum(ranks, 1.0)), axis=0))]),
            "median": np.column_stack([x, np.median(ranks, axis=0)])}


def _overlay_sets(drawn: Sequence[str], groups: list[Mapping[str, Any]] | None, colour_of: Mapping[str, str],
                  *, pool_rest: bool) -> list[tuple[list[str], str | None]]:
    """The runs an overlay pools, each set with its colour: every group's runs that drew a curve (in the group's
    colour), then the runs of no group, each alone in its own colour or, with `pool_rest`, all together (colour
    None: each its own)."""
    out, grouped = [], set()
    for index, group in enumerate(groups or []):
        grouped |= set(group["models"])
        members = [key for key in group["models"] if key in drawn]
        if members:
            out.append((members, group.get("color") or slot_colour(index)))
    rest = [key for key in drawn if key not in grouped]
    if pool_rest:
        return out + ([(rest, None)] if rest else [])
    return out + [([key], colour_of[key]) for key in rest]


def _draw_overlays(ax: Any, overlays: Mapping[str, Any], groups: list[Mapping[str, Any]] | None,
                   colour_of: Mapping[str, str], drawn: Sequence[str], job: Mapping[str, Any]) -> None:
    """
    The job's overlays over its GE curves, from the values GraphManager._overlay_data read for the runs that drew a
    curve (`drawn`; `colour_of` the colours of the runs in no group):

        ge_averages  {geometric | median: {run: [traces, 2] curve}}: each run's (a group's: the mean over its
                     runs) geometric mean and median over the attacks, dashed and dash-dotted in its colour
        ntge         {run: N̄tGE, NaN when never stably at rank 1}: a dot at GE 1 in the run's (group's) colour,
                     '>' at the x axis' end for NaN, and the median of each group's (else all the job's) runs
                     as a red diamond
        trace_lines  {n_val | budget: [trace counts]}: a vertical line at each distinct count

    One legend entry per kind of mark, after the runs'.
    """
    from matplotlib.transforms import blended_transform_factory

    width = get_line_kwargs(job)["linewidth"]
    own = (job.get("overlays") or {}).get("colors") or {}          # overlays.colors: a mark's own colour
    for name, curves in (overlays.get("ge_averages") or {}).items():
        dashes, text = GE_AVERAGE_STYLES[name]
        for members, colour in _overlay_sets(drawn, groups, colour_of, pool_rest=False):
            have = [_coerce_curve_array(curves[key]) for key in members if key in curves]
            if not have:
                continue
            x = have[0][:, 0]
            if any(c.shape != have[0].shape or not np.array_equal(c[:, 0], x) for c in have[1:]):
                raise ValueError(f"overlays.ge_averages: runs {members} have other x values (the number of traces); "
                                 "a group pools runs attacked on the same test traces")
            gx, gy = on_x_grid(job, x, np.mean([c[:, 1] for c in have], axis=0))
            ax.plot(gx, gy, color=own.get(name) or colour, linewidth=width, linestyle=dashes, zorder=3)
        ax.plot([], [], color=own.get(name) or OVERLAY_KEY_COLOUR, linewidth=width, linestyle=dashes, label=text)

    if "ntge" in overlays:
        values = overlays["ntge"]
        at_end = blended_transform_factory(ax.transAxes, ax.transData)     # x: the axis' end, y: GE 1
        never = False
        for members, colour in _overlay_sets(drawn, groups, colour_of, pool_rest=True):
            reached = []
            for key in (k for k in members if k in values):
                dot = own.get("ntge") or colour or colour_of[key]
                if not np.isfinite(values[key]):
                    ax.plot(1.0, 1.0, linestyle="none", marker=">", markersize=6, color=dot, markeredgecolor="none",
                            transform=at_end, clip_on=False, zorder=6)
                    never = True
                    continue
                reached.append(float(values[key]))
                ax.plot(values[key], 1.0, linestyle="none", marker="o", markersize=4, color=dot,
                        markeredgecolor="none", zorder=6)
            if reached:
                ax.plot(float(np.median(reached)), 1.0, linestyle="none", marker="D", markersize=6,
                        color=NTGE_MEDIAN_COLOUR, markeredgecolor="white", markeredgewidth=0.5, zorder=7)
        ax.plot([], [], linestyle="none", marker="o", markersize=4, color=own.get("ntge") or OVERLAY_KEY_COLOUR,
                label="each training's $\\bar{N}_{tGE}$")
        ax.plot([], [], linestyle="none", marker="D", markersize=6, color=NTGE_MEDIAN_COLOUR, label="their median")
        if never:
            ax.plot([], [], linestyle="none", marker=">", markersize=6, color=own.get("ntge") or OVERLAY_KEY_COLOUR,
                    label="never stably at rank 1")

    for name, at in (overlays.get("trace_lines") or {}).items():
        colour, dashes, text = TRACE_LINE_STYLES[name]
        for value in sorted(set(at)):
            ax.axvline(value, color=colour, linestyle=dashes, linewidth=0.9, zorder=2)
        if at:
            ax.plot([], [], color=colour, linestyle=dashes, linewidth=0.9, label=text)


def plot_curve_overlay(
    *,
    series: Mapping[str, Any],
    job: Mapping[str, Any],
    out_base: Path,
    metric: str,
    x_label: str,
    y_label: str,
    title: str | None = None,
    order: Any = None,
    dropped: Mapping[str, str] | None = None,
    groups: list[Mapping[str, Any]] | None = None,
    marks: Sequence[str] | None = None,
    overlays: Mapping[str, Any] | None = None,
) -> list[Path]:
    """
    One curve per series. Colours follow the slots of ``order`` (the job's
    model order; default: the series' order): a series that draws nothing
    (``dropped`` key -> reason, or no finite point) keeps its slot and stays in
    the legend as "<label> (no value: <reason>)". With ``groups`` (the job's), each group is one series
    (its runs thin, their mean bold, its band) in the first colours, and the models in no group follow.
    With ``marks`` (the job's, a process_overlay of validation and test): the marked points, _draw_rank_marks.
    With ``overlays`` (the job's overlays with the values GraphManager._overlay_data read): _draw_overlays.
    """
    apply_matplotlib_style(job)

    fig, ax = plt.subplots()
    plotted = 0

    slots = _overlay_order(series, order, dropped)
    reasons: dict[str, str] = dict(dropped or {})
    first = 0
    if groups:
        plotted += _draw_groups(ax, series, groups, job, reasons)
        grouped = {key for group in groups for key in group["models"]}
        slots = [key for key in slots if key not in grouped]
        first = len(groups)
    series_style = (job.get("style") or {}).get("series") or {}     # style.series: a curve's own colour
    colour_of = {key: (series_style.get(key) or {}).get("color") or slot_colour(index)
                 for index, key in enumerate(slots, start=first)}
    for label_key in slots:
        label = legend_label(job, label_key, label_key)
        colour = colour_of[label_key]
        if label_key in reasons or label_key not in series:
            _legend_only(ax, label, reasons.get(label_key, "not recorded"), colour)
            continue
        data = series[label_key]
        try:
            curve = _coerce_curve_array(data)
        except Exception as exc:
            print(f"[GraphManager][WARN] Skipping curve series={label_key}: {exc}")
            _legend_only(ax, label, "unreadable", colour)
            continue

        x = curve[:, 0]
        y = curve[:, 1]
        mask = np.isfinite(x) & np.isfinite(y)

        if not np.any(mask):
            _legend_only(ax, label, "NaN at every point", colour)
            continue

        _draw_xy(ax, x[mask], y[mask], job=job, label=label, allow_bar=False, color=colour, series=label_key)
        std = _curve_std(data)
        _draw_band(ax, x[mask], y[mask], std[mask] if std is not None else None, job=job, color=colour)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"[GraphManager][WARN] No curve overlay data plotted for metric={metric}")
        return []
    if marks:
        _draw_rank_marks(ax, series, marks)
    if overlays:
        _draw_overlays(ax, overlays, groups, colour_of, [key for key in series if key not in reasons], job)

    context = {
        "metric": metric,
        "process": job.get("process", ""),
        "x_axis": x_label,
        "y_axis": y_label,
    }

    _finalize_axes(
        ax,
        job=job,
        context=context,
        default_title=title
        or graph_title(job, context, fallback=f"{metric} comparison"),
        default_x_label=x_label,
        default_y_label=y_label,
    )

    return save_figure(fig, out_base, job)


def plot_distribution(
    *,
    data: Any,
    job: Mapping[str, Any],
    out_base: Path,
    metric: str,
    model: str,
    process: str,
    x_label: str,
    y_label: str,
    title: str | None = None,
) -> list[Path]:
    apply_matplotlib_style(job)

    if isinstance(data, pd.DataFrame):
        if metric in data.columns:
            values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)
        else:
            values = data.select_dtypes(include=[np.number]).to_numpy().reshape(-1)
    else:
        values = np.asarray(data).reshape(-1).astype(float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        print(
            f"[GraphManager][WARN] No finite distribution data: "
            f"metric={metric} model={model} process={process}"
        )
        return []

    fig, ax = plt.subplots()

    x = np.arange(1, len(values) + 1)

    plot_type = get_plot_type(job)
    if plot_type in {"histogram", "distribution"}:
        ax.hist(values, **get_bar_kwargs(job))
    else:
        ax.bar(x, values, **get_bar_kwargs(job))

    context = {
        "metric": metric,
        "model": model,
        "model_name": model,
        "process": process,
        "x_axis": x_label,
        "y_axis": y_label,
    }

    _finalize_axes(
        ax,
        job=job,
        context=context,
        default_title=title
        or graph_title(job, context, fallback=f"{metric} distribution"),
        default_x_label=x_label,
        default_y_label=y_label,
    )

    return save_figure(fig, out_base, job)
