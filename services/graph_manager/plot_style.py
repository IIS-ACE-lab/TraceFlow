# services/graph_manager/plot_style.py
"""
Plot styling helpers for TraceFlow GraphManager.

Purpose
-------
Centralize graph_config.yaml-driven visual styling:

- Matplotlib fallback paper settings
- YAML-overridden font sizes/families
- Line/marker style
- Grid style
- Axis scaling
- Text content + conservative text styling

Current schema
--------------
style:
  text_rules:
    - target: "title"
      text: "Model Key Rank Comparison on {{process}} Set"
      bold: true
      italic: false
      color: "black"
      font:
        family: "serif"
        size: 10

Notes
-----
Matplotlib does not reliably support mixed bold/italic/color inside
one normal title/axis-label string across PNG/PDF/SVG.

Therefore, TraceFlow v1 applies whole-object styling:
- whole title
- whole x label
- whole y label
- whole legend text

No underline: Matplotlib (3.10) has none, neither Text.set_underline nor a
mathtext \\underline; a rule with underline is refused.

Author: Shraddha Mishra
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

from services.graph_manager.job_paths import render_template  # noqa: F401  (re-exported: plotters import it from here)


# Figure size in inches (width, height) of each orientation.
FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "landscape": (6.4, 4.8),
    "portrait": (4.8, 6.4),
}

DEFAULT_STYLE: dict[str, Any] = {
    "figure": {
        "orientation": "landscape",
        "width": 0.0,
        "height": 0.0,
    },
    "font": {
        "family": "serif",
        "base_size": 10,
        "title_size": 10,
        "label_size": 9,
        "tick_size": 8,
        "legend_size": 8,
    },
    "line": {
        "width": 1.0,
        "marker": "o",
        "marker_size": 3.5,
    },
    "grid": {
        "enabled": True,
        "alpha": 0.35,
    },
    # Shaded mean +/- std over the cross-validation folds (error bars on bars).
    "std_band": {
        "enabled": True,
        "alpha": 0.15,
    },
    "axes": {
        "x_scale": "linear",
        "y_scale": "linear",
        "x_limits": None,
        "y_limits": None,
        "x_step": None,
        "y_step": None,
        "x_grid": None,
        # true: also draw the first recorded point when it is off x_grid. A run
        # recorded at every trace then starts at its recorded x, n = 1 (the
        # paper's MLP figures draw the rank after one trace next to x = 0).
        "x_grid_first": False,
        # Padding on each side, as a fraction of the x_limits / y_limits span
        # (Matplotlib's Axes.margins; its own default is 0.05). With a margin the
        # points outside the limits are not drawn: x_limits [0, 950], x_margin
        # 0.05 shows the axis from -47.5 to 997.5 and the curve from 0 to 950.
        "x_margin": 0.0,
        "y_margin": 0.0,
    },
    "text_rules": [],
}


# Latin Modern is the OpenType form of Computer Modern, the text font of a
# LaTeX document that loads no font package. A TeX installation ships it, but
# outside the font directories Matplotlib scans, so style.font.family:
# "Latin Modern Roman" resolves only after these files are registered. Only
# the 10 pt design is registered: every optical size carries the same family
# name, and Matplotlib could not tell them apart.
TEX_FONT_FAMILIES = ("Latin Modern Roman", "Latin Modern Sans", "Latin Modern Mono")
_TEX_FONT_FILES = (
    "lmroman10-regular.otf", "lmroman10-bold.otf", "lmroman10-italic.otf", "lmroman10-bolditalic.otf",
    "lmsans10-regular.otf", "lmsans10-bold.otf", "lmsans10-oblique.otf", "lmsans10-boldoblique.otf",
    "lmmono10-regular.otf", "lmmono10-italic.otf",
)
_TEX_FONT_DIRS = (
    "/usr/share/texmf/fonts/opentype/public/lm",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm",
    "/usr/share/fonts/opentype/lmodern",
    "/usr/share/fonts/OTF",
)
_tex_fonts_registered: bool | None = None


def register_tex_fonts() -> bool:
    """
    Register the Latin Modern fonts of the TeX installation with Matplotlib.

    Returns True when "Latin Modern Roman" is available afterwards. Runs the
    search once per process.
    """
    global _tex_fonts_registered
    if _tex_fonts_registered is not None:
        return _tex_fonts_registered

    dirs: list[str] = []
    try:
        hit = subprocess.run(
            ["kpsewhich", _TEX_FONT_FILES[0]], capture_output=True, text=True, timeout=20
        ).stdout.strip()
        if hit:
            dirs.append(str(Path(hit).parent))
    except (OSError, subprocess.SubprocessError):
        pass

    known = {entry.fname for entry in font_manager.fontManager.ttflist}
    for directory in (*dirs, *_TEX_FONT_DIRS):
        if not (Path(directory) / _TEX_FONT_FILES[0]).is_file():
            continue
        for name in _TEX_FONT_FILES:
            path = str(Path(directory) / name)
            if path in known or not Path(path).is_file():
                continue
            try:
                font_manager.fontManager.addfont(path)
            except Exception:  # an unreadable font file is skipped
                continue
        break

    _tex_fonts_registered = any(
        entry.name == TEX_FONT_FAMILIES[0] for entry in font_manager.fontManager.ttflist
    )
    return _tex_fonts_registered


def _is_tex_font(family: str) -> bool:
    return str(family).strip() in TEX_FONT_FAMILIES


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _safe_str(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _safe_int(value: Any, default: int, *, min_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = default

    if min_value is not None:
        out = max(out, min_value)

    return out


def _safe_float(
    value: Any,
    default: float,
    *,
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


def _safe_bool(value: Any, default: bool) -> bool:
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


def get_style(job: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a normalized style dictionary.

    graph_job_validator.py already validates and normalizes style when the
    normal GraphManager path is used. This function remains defensive so
    plotters do not crash if a legacy/non-normalized job reaches them.
    """

    raw = _as_mapping(job.get("style", {}))

    font_raw = _as_mapping(raw.get("font", {}))
    line_raw = _as_mapping(raw.get("line", {}))
    grid_raw = _as_mapping(raw.get("grid", {}))
    std_band_raw = _as_mapping(raw.get("std_band", {}))
    axes_raw = _as_mapping(raw.get("axes", {}))

    default_font = DEFAULT_STYLE["font"]
    default_line = DEFAULT_STYLE["line"]
    default_grid = DEFAULT_STYLE["grid"]
    default_std_band = DEFAULT_STYLE["std_band"]
    default_axes = DEFAULT_STYLE["axes"]

    base_size = _safe_int(
        font_raw.get("base_size", default_font["base_size"]),
        default_font["base_size"],
        min_value=1,
    )

    text_rules = raw.get("text_rules", [])
    if not isinstance(text_rules, list):
        text_rules = []

    figure_raw = _as_mapping(raw.get("figure", {}))
    orientation = _safe_str(figure_raw.get("orientation"), "landscape").lower()
    if orientation not in FIGURE_SIZES:
        orientation = "landscape"

    return {
        "figure": {
            "orientation": orientation,
            "width": _safe_float(figure_raw.get("width", 0.0), 0.0, min_value=0.0),
            "height": _safe_float(figure_raw.get("height", 0.0), 0.0, min_value=0.0),
        },
        "font": {
            "family": _safe_str(font_raw.get("family"), default_font["family"]),
            "base_size": base_size,
            "title_size": _safe_int(
                font_raw.get("title_size", default_font["title_size"]),
                default_font["title_size"],
                min_value=1,
            ),
            "label_size": _safe_int(
                font_raw.get("label_size", default_font["label_size"]),
                default_font["label_size"],
                min_value=1,
            ),
            "tick_size": _safe_int(
                font_raw.get("tick_size", default_font["tick_size"]),
                default_font["tick_size"],
                min_value=1,
            ),
            "legend_size": _safe_int(
                font_raw.get("legend_size", default_font["legend_size"]),
                default_font["legend_size"],
                min_value=1,
            ),
        },
        "line": {
            "width": _safe_float(
                line_raw.get("width", default_line["width"]),
                default_line["width"],
                min_value=0.1,
            ),
            "marker": _safe_str(line_raw.get("marker"), default_line["marker"]),
            "marker_size": _safe_float(
                line_raw.get("marker_size", default_line["marker_size"]),
                default_line["marker_size"],
                min_value=0.1,
            ),
            # series -> line style (validated by graph_job_validator)
            "styles": {str(k): str(v) for k, v in _as_mapping(line_raw.get("styles", {})).items()},
        },
        "grid": {
            "enabled": _safe_bool(
                grid_raw.get("enabled", default_grid["enabled"]),
                default_grid["enabled"],
            ),
            "alpha": _safe_float(
                grid_raw.get("alpha", default_grid["alpha"]),
                default_grid["alpha"],
                min_value=0.0,
                max_value=1.0,
            ),
        },
        "std_band": {
            "enabled": _safe_bool(
                std_band_raw.get("enabled", default_std_band["enabled"]),
                default_std_band["enabled"],
            ),
            "alpha": _safe_float(
                std_band_raw.get("alpha", default_std_band["alpha"]),
                default_std_band["alpha"],
                min_value=0.0,
                max_value=1.0,
            ),
        },
        "axes": {
            "x_scale": _safe_str(axes_raw.get("x_scale"), default_axes["x_scale"]),
            "y_scale": _safe_str(axes_raw.get("y_scale"), default_axes["y_scale"]),
            "x_limits": axes_raw.get("x_limits"),
            "y_limits": axes_raw.get("y_limits"),
            "x_step": axes_raw.get("x_step"),
            "y_step": axes_raw.get("y_step"),
            "x_grid": axes_raw.get("x_grid"),
            "x_grid_first": bool(axes_raw.get("x_grid_first", default_axes["x_grid_first"])),
            "x_margin": _safe_float(axes_raw.get("x_margin", default_axes["x_margin"]), default_axes["x_margin"], min_value=0.0),
            "y_margin": _safe_float(axes_raw.get("y_margin", default_axes["y_margin"]), default_axes["y_margin"], min_value=0.0),
        },
        "text_rules": text_rules,
    }


def apply_matplotlib_style(job: Mapping[str, Any]) -> None:
    """
    Apply global Matplotlib settings for paper-safe output.

    Important:
    - pdf.fonttype=42 helps avoid Type 3 fonts in PDFs.
    - ps.fonttype=42 helps EPS/PS font handling.
    - svg.fonttype='path' writes SVG text as its glyphs' outlines (owner 2026-10-27): the figure shows Latin
      Modern in every viewer and in Inkscape (svg.sty), which find no font named "Latin Modern Roman".
    """

    style = get_style(job)
    font = style["font"]
    figure = style["figure"]
    width, height = FIGURE_SIZES[figure["orientation"]]

    # The paper's font: registered from the TeX installation, with the
    # Computer Modern math font for any math in a label.
    tex_font = _is_tex_font(font["family"])
    if tex_font and not register_tex_fonts():
        warnings.warn(
            f"style.font.family {font['family']!r}: the Latin Modern fonts were not found "
            "(no TeX installation?); Matplotlib falls back to its default font.",
            stacklevel=2,
        )

    mpl.rcParams.update(
        {
            "figure.figsize": (figure["width"] or width, figure["height"] or height),
            "font.family": font["family"],
            "mathtext.fontset": "cm" if tex_font else mpl.rcParamsDefault["mathtext.fontset"],
            "font.size": font["base_size"],
            "axes.titlesize": font["title_size"],
            "axes.labelsize": font["label_size"],
            "xtick.labelsize": font["tick_size"],
            "ytick.labelsize": font["tick_size"],
            "legend.fontsize": font["legend_size"],
            "lines.linewidth": style["line"]["width"],
            "lines.markersize": style["line"]["marker_size"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


def get_text_rule(job: Mapping[str, Any], target: str) -> dict[str, Any] | None:
    """
    Return the first text rule matching a target.
    """

    style = get_style(job)
    target = str(target).strip()

    for rule in style.get("text_rules", []):
        if not isinstance(rule, Mapping):
            continue

        if str(rule.get("target", "")).strip() == target:
            return dict(rule)

    return None


def get_text_for_target(
    job: Mapping[str, Any],
    target: str,
    *,
    default: str = "",
    context: Mapping[str, Any] | None = None,
) -> str:
    """
    Return rendered text content for a target.

    New schema:
      text_rules[].text

    Compatibility:
      text_rules[].words is accepted only as a fallback.
    """

    rule = get_text_rule(job, target)
    if not rule:
        return render_template(default, context)

    text = rule.get("text", rule.get("words", default))

    if isinstance(text, list):
        text = text[0] if text else default

    return render_template(text, context)


def parse_legend_handle(value: Any) -> dict[str, Any] | None:
    """
    legend.handle: [shape, thickness] -> {"shape", "marker", "size"}, or None.

    shape  "line"     a line of width thickness (pt)
           a Matplotlib marker ("o", "s", "^", "D", "*", "x", "+", ...) or
           any other single character, drawn as text ("■", "●", "a")
           -> a marker of size thickness (pt)
    Raises ValueError on anything else.
    """
    if value is None:
        return None
    if isinstance(value, Mapping) and {"shape", "marker", "size"} <= set(value):
        return dict(value)  # already parsed
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"legend.handle must be [shape, thickness], got {value!r}")
    shape, thickness = value
    shape = str(shape).strip() if shape is not None else ""
    if isinstance(thickness, bool):
        raise ValueError(f"legend.handle thickness must be a number of pt, got {thickness!r}")
    try:
        size = float(thickness)
    except (TypeError, ValueError):
        raise ValueError(f"legend.handle thickness must be a number of pt, got {thickness!r}") from None
    if not size > 0:
        raise ValueError(f"legend.handle thickness must be > 0 pt, got {thickness!r}")
    if shape.lower() == "line":
        return {"shape": "line", "marker": None, "size": size}

    from matplotlib.lines import Line2D

    named = {m for m in Line2D.markers if isinstance(m, str) and m.strip() and m not in {"None", "none"}}
    if shape in named or (len(shape) > 2 and shape.startswith("$") and shape.endswith("$")):
        return {"shape": "marker", "marker": shape, "size": size}
    if len(shape) == 1 and shape not in {"$", "\\"}:
        return {"shape": "marker", "marker": f"${shape}$", "size": size}
    raise ValueError(
        f"legend.handle shape {shape!r} invalid: use \"line\", a Matplotlib marker "
        f"({', '.join(sorted(named))}) or one character"
    )


def get_x_grid(job: Mapping[str, Any]) -> float | None:
    """style.axes.x_grid: draw only the points whose x is a multiple of it, or None."""
    value = (get_style(job).get("axes") or {}).get("x_grid")
    return None if value in (None, 0) else float(value)


def on_x_grid(job: Mapping[str, Any], x: Any, *columns: Any) -> tuple:
    """(x, *columns) restricted to the points of style.axes.x_grid.

    style.axes.x_grid_first true keeps the first (smallest x) recorded point too.

    A rank curve recorded at every trace then draws as one recorded on the
    paper's grid (e.g. every 50 traces): the rank at n = 50 is the same number
    in both, only the points in between are left out.
    """
    grid = get_x_grid(job)
    axes = get_style(job).get("axes") or {}
    cut = axes.get("x_limits") if float(axes.get("x_margin") or 0.0) > 0 else None
    x = np.asarray(x)
    if (grid is None and cut is None) or x.size == 0 or not np.issubdtype(x.dtype, np.number):
        return (x, *columns)
    xf = x.astype(float)
    keep = np.ones(x.shape, dtype=bool)
    if grid is not None:
        on_grid = np.isclose(np.mod(xf, grid), 0.0) | np.isclose(np.mod(xf, grid), grid)
        if axes.get("x_grid_first"):            # the first recorded point, e.g. n = 1
            on_grid[int(np.argmin(xf))] = True
        keep &= on_grid
    if cut is not None:            # a padded axis: the points outside x_limits would show in the margin
        lo, hi = (float(v) for v in cut)
        keep &= (xf >= lo - 1e-9) & (xf <= hi + 1e-9)
    return (x[keep], *[None if c is None else np.asarray(c)[keep] for c in columns])


def get_std_band(job: Mapping[str, Any]) -> dict[str, Any]:
    """style.std_band: {"enabled": bool, "alpha": float in [0, 1]}."""
    return get_style(job)["std_band"]


def get_line_style(job: Mapping[str, Any], series: str | None) -> str | None:
    """
    style.line.styles for one series ("<model>" or "<model>:<metric>"): the
    exact key, else its metric part, else its model part; None = solid.
    """
    styles = (get_style(job).get("line") or {}).get("styles") or {}
    if not series or not styles:
        return None
    model, _, metric = str(series).partition(":")
    for key in (series, metric, model):
        if key and key in styles:
            return styles[key]
    return None


def get_line_kwargs(job: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return Matplotlib kwargs for line-like plots.
    """

    style = get_style(job)
    line = style["line"]

    marker = line["marker"]
    if str(marker).lower() in {"none", "null", "false", ""}:
        marker = None

    return {
        "linewidth": line["width"],
        "marker": marker,
        "markersize": line["marker_size"],
    }


def get_scatter_kwargs(job: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return Matplotlib kwargs for scatter plots.
    """

    style = get_style(job)
    marker_size = float(style["line"]["marker_size"])

    return {
        "s": max(marker_size * marker_size, 1.0),
    }


def get_bar_kwargs(job: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return Matplotlib kwargs for bar/distribution plots.

    Kept minimal because default Matplotlib colors are acceptable and
    paper-safe. Color customization can be added later if needed.
    """

    return {}


def apply_axes_style(ax: Any, job: Mapping[str, Any]) -> None:
    """
    Apply axis scaling, tick font sizes, and grid settings.
    """

    style = get_style(job)
    axes = style["axes"]
    font = style["font"]
    grid = style["grid"]

    try:
        ax.set_xscale(axes["x_scale"])
    except Exception:
        ax.set_xscale("linear")

    try:
        ax.set_yscale(axes["y_scale"])
    except Exception:
        ax.set_yscale("linear")

    # A fixed range, e.g. the first 600 traces of a 5,000-trace curve, padded by
    # x_margin / y_margin times its span on each side (0: the range exactly).
    for limits, margin, setter in (("x_limits", "x_margin", ax.set_xlim), ("y_limits", "y_margin", ax.set_ylim)):
        if axes.get(limits):
            lo, hi = (float(v) for v in axes[limits])
            pad = float(axes.get(margin) or 0.0) * (hi - lo)
            setter(lo - pad, hi + pad)
    # A tick every step units, e.g. every 20 traces.
    if axes.get("x_step"):
        ax.xaxis.set_major_locator(MultipleLocator(axes["x_step"]))
    if axes.get("y_step"):
        ax.yaxis.set_major_locator(MultipleLocator(axes["y_step"]))

    ax.tick_params(axis="both", labelsize=font["tick_size"])

    if grid["enabled"]:
        ax.grid(True, alpha=grid["alpha"])
    else:
        ax.grid(False)


def _rule_matches_text(rule: Mapping[str, Any], current_text: str) -> bool:
    """
    Decide whether a style rule applies to an existing Matplotlib text object.

    In the new schema, rule['text'] is the target text/content.
    Template placeholders such as {model_name} and {value} are treated as
    wildcard-style rules.
    """

    raw_text = rule.get("text", rule.get("words", ""))

    if isinstance(raw_text, list):
        raw_text = raw_text[0] if raw_text else ""

    rule_text = str(raw_text).strip()

    if not rule_text:
        return True

    if rule_text in {"{model_name}", "{{model_name}}", "{value}", "{{value}}"}:
        return True

    # Templated text ("... {{model_name}}") matches on its literal prefix,
    # since the rendered text carries the substituted value.
    if "{" in rule_text:
        prefix = rule_text.split("{", 1)[0].strip()
        return not prefix or prefix in current_text

    return rule_text in current_text


def apply_text_object_style(text_obj: Any, job: Mapping[str, Any], target: str) -> None:
    """
    Apply conservative whole-object text styling based on matching text_rules.

    target examples:
      - title
      - x_label
      - y_label
      - legend_text
    """

    if text_obj is None:
        return

    style = get_style(job)
    rules = style.get("text_rules", [])

    try:
        current_text = text_obj.get_text()
    except Exception:
        current_text = ""

    if current_text is None:
        current_text = ""

    target = str(target).strip()

    for rule in rules:
        if not isinstance(rule, Mapping):
            continue

        rule_target = str(rule.get("target", "")).strip()
        if rule_target != target:
            continue

        if not _rule_matches_text(rule, str(current_text)):
            continue

        font = _as_mapping(rule.get("font", {}))

        try:
            text_obj.set_fontfamily(_safe_str(font.get("family"), style["font"]["family"]))
        except Exception:
            pass

        try:
            text_obj.set_fontsize(
                _safe_int(
                    font.get("size", style["font"]["base_size"]),
                    style["font"]["base_size"],
                    min_value=1,
                )
            )
        except Exception:
            pass

        try:
            text_obj.set_fontweight(
                "bold" if _safe_bool(rule.get("bold"), False) else "normal"
            )
        except Exception:
            pass

        try:
            text_obj.set_fontstyle(
                "italic" if _safe_bool(rule.get("italic"), False) else "normal"
            )
        except Exception:
            pass

        try:
            text_obj.set_color(_safe_str(rule.get("color"), "black"))
        except Exception:
            pass



def apply_title_style(ax: Any, job: Mapping[str, Any]) -> None:
    apply_text_object_style(ax.title, job, "title")


def apply_axis_label_style(ax: Any, job: Mapping[str, Any]) -> None:
    apply_text_object_style(ax.xaxis.label, job, "x_label")
    apply_text_object_style(ax.yaxis.label, job, "y_label")


def apply_legend_text_style(ax: Any, job: Mapping[str, Any]) -> None:
    legend = ax.get_legend()
    if legend is None:
        return

    for text_obj in legend.get_texts():
        apply_text_object_style(text_obj, job, "legend_text")


def apply_all_text_styles(ax: Any, job: Mapping[str, Any]) -> None:
    """
    Apply all currently supported text styling hooks.
    """

    apply_title_style(ax, job)
    apply_axis_label_style(ax, job)
    apply_legend_text_style(ax, job)


def apply_configured_text(
    ax: Any,
    job: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    default_title: str = "",
    default_x_label: str = "",
    default_y_label: str = "",
) -> None:
    """
    Set title/xlabel/ylabel from text_rules, then apply text styles.

    Plotters should call this after all data series are drawn and before
    saving the figure.
    """

    title = get_text_for_target(
        job,
        "title",
        default=default_title,
        context=context,
    )
    x_label = get_text_for_target(
        job,
        "x_label",
        default=default_x_label,
        context=context,
    )
    y_label = get_text_for_target(
        job,
        "y_label",
        default=default_y_label,
        context=context,
    )

    if title:
        ax.set_title(title)

    if x_label:
        ax.set_xlabel(x_label)

    y_label = with_rank_base(y_label, (context or {}).get("metric"))
    if y_label:
        ax.set_ylabel(y_label)

    apply_all_text_styles(ax, job)


# Metrics whose values are key ranks: TraceFlow ranks from 1 (rank 1 = the key recovered), the
# paper's rank function (Eq. 6) from 0. A rank figure says so on its axis, whatever label the job
# writes, so no reader has to know the convention.
RANK_METRICS = frozenset({
    "mean_attack_rank_curve", "per_attack_curves", "fold_rank_curves", "full_ranks", "final_ranks",
    "guessing_entropy", "key_rank", "final_rank", "mean_rank", "median_rank", "rank_percentile",
    "ge_at_traces", "rank_at_traces",
})
RANK_BASE_NOTE = "(1 = key recovered)"


def with_rank_base(label: str, metric: Any) -> str:
    """`label` with the rank base noted when `metric` is a rank (once; never on another metric)."""
    if not label or str(metric or "") not in RANK_METRICS or RANK_BASE_NOTE in label:
        return label
    return f"{label} {RANK_BASE_NOTE}"
