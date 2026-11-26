# services/graph_manager/faceting.py
"""
Faceting: place finished graph files side by side in one figure.

A facet lists graph files that graph jobs already produced, gives each one a
``file_id``, and arranges them by those ids.

YAML contract (graph_config.yaml, top level, next to ``jobs``):

  faceting:
    - id: 1
      name: "rank_and_fit_time"
      enabled: true
      output_file_name: "rank_and_fit_time"
      panel_height: 3.2          # inches, height of every row
      spacing: 0.15              # inches between panels
      panels:
        - file_id: 1
          address: "jobs/job_001_.../curve.pdf"   # relative to <project>/graphs/
          orientation: "landscape"                # landscape | portrait
          label: "(a) Adam"                       # optional; {letter} {index} {roman} {file_id} {file}
          label_position: "bottom"                # bottom | bottom-left | bottom-right | top | top-left | top-right | left
        - file_id: 2
          address: "jobs/job_012_.../bars.pdf"
          orientation: "portrait"
      arrangement:
        mode: "grid"             # horizontal | vertical | grid
        fit: "height"            # height (default): every row panel_height tall | width: every row the widest's width
        order: [[1, 2],          # file_ids; in a grid every inner list is one
                [3, 4, 5]]       # row, and rows may hold different numbers
      legend:
        enabled: true
        position: "bottom"       # top | bottom | right
        caption: "Fig. a"        # optional: a line under the legend, in its font
        columns: 4               # entries per legend row
        handle: ["line", 1.5]    # [shape, thickness pt]: line | patch | a Matplotlib marker | one character
        mode: "custom"           # auto (default): the entries of the job that drew the first panel
        entries:                 # mode custom
          - label: "batch size=50"
            color: "C0"          # optional; default: colour cycle in order
      export:
        formats: ["svg"]         # svg (the default: the panels' SVGs nested, all vector) | png | pdf
        margin: 0.05             # inches of white border
        png:
          dpi: 300               # the png, and a pdf / svg that embeds the png panels
        svg:                     # as a graph job's: sanitize, validate_xml, on_error
          sanitize: true
      text_rules:                # targets: title | x_label | y_label | panel_label | legend_text
        - target: "title"
          text: "..."

arrangement.mode
    horizontal  one row, ``order`` from left to right
    vertical    one column, ``order`` from top to bottom
    grid        rows from top to bottom, each row from left to right.
                ``order: [[1, 2], [3, 4, 5]]`` names the rows; a flat
                ``order`` is cut into rows of ``columns`` panels.
``order`` defaults to the panels as listed; a panel left out of it is not drawn.
Rows of equal length share their column widths; rows of different lengths
are centred.

legend
    Panels are finished files, so a facet cannot read their curves. mode
    auto takes the entries of the graph job that drew the first panel (its
    folder jobs/job_<id>_...): one per model of its models (job 0:
    top_models), labelled as that job's legend labels them, in colour i of
    the colour cycle (C0, C1, ...), the colour graph jobs give to their i-th
    model. mode custom takes ``entries`` as written: entry i in colour i
    unless ``color`` is set; ``labels: [...]`` is their short form.
    ``handle`` is the job legend's [shape, thickness] plus the shape patch.

panels[].label
    The panel's own label, below or above it (label_position); its font is
    the panel_label text rule's. Without one, the panel_label rule's text
    (e.g. "({letter})") labels every panel, below it.

The extension of ``address`` is free: PNG output reads the .png next to it and
PDF output the .pdf. PDF panels are placed as vector graphics when the pypdf
package is installed; otherwise the PNG panels are embedded in the PDF. SVG
output (the default, owner 2026-10-27) nests each panel's own SVG, every id
prefixed so no two panels' definitions clash, all vector; a panel without an
SVG makes it embed the PNG panels instead. Text is written as outlined glyphs
(svg.fonttype path), so Latin Modern shows in every viewer. It is sanitized and checked as a graph
job's SVG (export.svg: sanitize, validate_xml, on_error).

Author: Shraddha Mishra
"""

from __future__ import annotations

from services.side_channel_core.utils.config_errors import ConfigError
from services.graph_manager.job_paths import facet_safe as _safe  # the panel names a facet's folder the same way
import io
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from .graph_job_validator import VALID_SVG_ON_ERROR
from .plot_style import parse_legend_handle, register_tex_fonts
from .svg_sanitizer import sanitize_and_validate_svg

VALID_ORIENTATIONS = {"landscape", "portrait"}
VALID_ARRANGEMENTS = {"horizontal", "vertical", "grid"}
VALID_FACET_FORMATS = {"png", "pdf", "svg"}
VALID_FACET_TEXT_TARGETS = {"title", "x_label", "y_label", "panel_label", "legend_text"}
VALID_LEGEND_POSITIONS = {"top", "bottom", "right"}
VALID_LEGEND_MODES = {"auto", "custom"}
VALID_LABEL_POSITIONS = ("bottom", "bottom-left", "bottom-right", "top", "top-left", "top-right", "left")


def _roman(n: int) -> str:
    """n as a Roman numeral (a panel label's {roman}: I, II, III, ...)."""
    out = ""
    for value, digits in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                          (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + digits, n - value
    return out
# Keys renamed or removed (2026-10-17), for the error that names the new one.
MOVED_KEYS = {
    "style": "the facet's text_rules (style.text_rules is now text_rules)",
    "export.dpi": "export.png.dpi (only png, and a pdf of png panels, use it)",
    "legend.style": "legend.handle: [line | patch, thickness]",
}
PANEL_SUFFIXES = {".png", ".pdf", ".svg", ".eps"}

# Width / height of a cell whose file does not have the declared orientation.
FALLBACK_ASPECT = {"landscape": 4.0 / 3.0, "portrait": 3.0 / 4.0}


class FacetValidationError(ConfigError):     # a configuration error (utils/config_errors.py), a ValueError as before
    """Raised when a faceting entry is malformed."""


@dataclass
class Panel:
    file_id: int
    file: str  # the panel's address
    orientation: str
    row: int
    column: int
    label: str | None = None
    label_position: str = "bottom"
    stem: Path | None = None
    aspect: float = 1.0       # width / height of the file
    cell_width: float = 0.0   # inches
    size: tuple[float, float] | None = None   # the file's own width, height in inches (an SVG or PDF)


def _as_int(value: Any, *, name: str, facet_id: Any, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise FacetValidationError(f"facet id={facet_id} {name} must be an integer")
    try:
        out = int(value)
    except Exception as exc:
        raise FacetValidationError(
            f"facet id={facet_id} {name} must be an integer, got {value!r}"
        ) from exc
    if out < minimum:
        raise FacetValidationError(f"facet id={facet_id} {name} must be >= {minimum}")
    return out


def _as_float(value: Any, *, name: str, facet_id: Any, minimum: float, default: float) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except Exception as exc:
        raise FacetValidationError(
            f"facet id={facet_id} {name} must be a number, got {value!r}"
        ) from exc
    if out < minimum:
        raise FacetValidationError(f"facet id={facet_id} {name} must be >= {minimum}")
    return out


def _text_rule(rule: Any, *, facet_id: Any) -> dict[str, Any]:
    if not isinstance(rule, Mapping):
        raise FacetValidationError(f"facet id={facet_id} text_rules entries must be mappings")
    target = str(rule.get("target", "")).strip()
    if target not in VALID_FACET_TEXT_TARGETS:
        raise FacetValidationError(
            f"facet id={facet_id} text_rules target={target!r} invalid. "
            f"Allowed: {sorted(VALID_FACET_TEXT_TARGETS)}"
        )
    font = rule.get("font", {}) or {}
    if not isinstance(font, Mapping):
        raise FacetValidationError(f"facet id={facet_id} text rule {target!r} font must be a mapping")
    return {
        "target": target,
        "text": str(rule.get("text", "")).strip(),
        "bold": bool(rule.get("bold", False)),
        "italic": bool(rule.get("italic", False)),
        "color": str(rule.get("color", "black")).strip() or "black",
        "family": str(font.get("family", "serif")).strip() or "serif",
        "size": _as_int(font.get("size", 10), name=f"text rule {target!r} font.size", facet_id=facet_id),
    }


def _handle(value: Any, *, facet_id: Any) -> dict[str, Any]:
    """legend.handle [shape, thickness]: a job legend's handle (line, marker, character) or patch."""
    if value is None:
        return {"shape": "line", "marker": None, "size": 1.5}
    if isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]).strip().lower() == "patch":
        size = parse_legend_handle(["line", value[1]])["size"] if value[1] is not None else 1.5
        return {"shape": "patch", "marker": None, "size": size}
    try:
        return parse_legend_handle(value)
    except ValueError as exc:
        raise FacetValidationError(f"facet id={facet_id} {exc} (or patch)") from None


def _job_entries(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A graph job's legend entries: its models (job 0: top_models), as its legend labels them."""
    models = job.get("top_models") if int(job.get("id", -1)) == 0 else job.get("models")
    legend = job.get("legend") or {}
    labels = legend.get("labels") or {} if str(legend.get("mode", "auto")).lower() == "custom" else {}
    return [{"label": str(labels.get(m) or m), "color": f"C{i % 10}"} for i, m in enumerate(models or [])]


def _panel_job(address: str) -> int | None:
    """The graph job id of a panel drawn under jobs/job_<id>_<name>/ (a job of several runs) or under a
    run's graphs/job_<id>_<name>/ (artifacts/..., a job of one run: job_paths.py), else None."""
    from services.graph_manager import job_paths

    found = job_paths.address_job(address)
    return found[1] if found else None


def _legend(raw: Any, *, facet_id: Any, first_address: str = "", jobs: Any = None) -> dict[str, Any]:
    """Normalize the optional legend of a facet; mode auto reads the jobs when given."""
    if raw is None:
        return {"enabled": False, "entries": []}
    if not isinstance(raw, Mapping):
        raise FacetValidationError(f"facet id={facet_id} legend must be a mapping")
    if "style" in raw:
        raise FacetValidationError(f"facet id={facet_id} legend.style moved to {MOVED_KEYS['legend.style']}")
    mode = str(raw.get("mode", "auto")).strip().lower()
    if mode not in VALID_LEGEND_MODES:
        raise FacetValidationError(
            f"facet id={facet_id} legend.mode={mode!r} invalid. Allowed: {sorted(VALID_LEGEND_MODES)}"
        )
    handle = _handle(raw.get("handle"), facet_id=facet_id)

    position = str(raw.get("position", "bottom")).strip().lower()
    if position not in VALID_LEGEND_POSITIONS:
        raise FacetValidationError(
            f"facet id={facet_id} legend.position={position!r} invalid. "
            f"Allowed: {sorted(VALID_LEGEND_POSITIONS)}"
        )
    enabled = bool(raw.get("enabled", True))
    entries: list[dict[str, Any]] = []
    if mode == "auto":
        if raw.get("entries") or raw.get("labels"):
            raise FacetValidationError(
                f"facet id={facet_id} legend.entries are written for mode: custom; mode auto takes "
                "the entries of the first panel's graph job"
            )
        job_id = _panel_job(first_address)
        if job_id is None and enabled:
            raise FacetValidationError(
                f"facet id={facet_id} legend.mode auto reads the graph job of the first panel, but "
                f"{first_address!r} is not under jobs/job_<id>_<name>/; use mode: custom with entries"
            )
        if jobs is not None and job_id is not None:
            job = next((j for j in jobs if isinstance(j, Mapping) and str(j.get("id")) == str(job_id)), None)
            if job is None:
                raise FacetValidationError(
                    f"facet id={facet_id} legend.mode auto: the first panel is job {job_id}'s, and "
                    "graph_config.yaml has no such job"
                )
            entries = _job_entries(job)
    else:
        raw_entries = raw.get("entries")
        if raw_entries is None:
            raw_entries = raw.get("labels") or []
        if not isinstance(raw_entries, list):
            raise FacetValidationError(f"facet id={facet_id} legend.entries must be a list")
        for index, entry in enumerate(raw_entries):
            if not isinstance(entry, Mapping):
                entry = {"label": entry}
            if "style" in entry:
                raise FacetValidationError(
                    f"facet id={facet_id} legend entry {index + 1}: style moved to {MOVED_KEYS['legend.style']}"
                )
            label = str(entry.get("label", "")).strip()
            if not label:
                raise FacetValidationError(f"facet id={facet_id} legend entry {index + 1} has no label")
            entries.append({"label": label, "color": str(entry.get("color") or f"C{index % 10}").strip()})
        if enabled and not entries:
            raise FacetValidationError(
                f"facet id={facet_id} legend is enabled but has no entries; list them under "
                "legend.entries, or take a job's with mode: auto"
            )
    default_columns = 1 if position == "right" else max(len(entries), 1)
    colormap = _legend_colormap(raw.get("colormap"), facet_id=facet_id, position=position)
    return {
        "enabled": enabled,
        "position": position,
        "columns": _as_int(raw.get("columns", default_columns), name="legend.columns", facet_id=facet_id),
        "title": str(raw.get("title") or "").strip(),
        "caption": str(raw.get("caption") or "").strip(),     # a line under the legend (owner 2026-10-27: "Fig. a")
        "frame": bool(raw.get("frame", False)),
        "mode": mode,
        "handle": handle,
        "entries": entries,
        "colormap": colormap,
    }


def _legend_colormap(raw: Any, *, facet_id: Any, position: str) -> dict[str, Any] | None:
    """legend.colormap {name, limits: [low, high], label}: a horizontal spectrum above the legend's entries, the scale
    the facet's heat maps share (style.heatmap.limits, colorbar false) (owner 2026-10-28: Fig. 5.ii). Top or bottom."""
    if raw in (None, "", {}):
        return None
    where = f"facet id={facet_id} legend.colormap"
    if not isinstance(raw, Mapping) or set(raw) - {"name", "limits", "label"}:
        raise FacetValidationError(f"{where} is {{name, limits, label}}, got {raw!r}")
    if position == "right":
        raise FacetValidationError(f"{where} needs legend.position top or bottom: the spectrum lies across the facet")
    import matplotlib

    name = str(raw.get("name") or "viridis").strip()
    if name not in matplotlib.colormaps:
        raise FacetValidationError(f"{where}.name={name!r} is not a Matplotlib colormap (e.g. turbo, viridis)")
    limits = raw.get("limits")
    if (not isinstance(limits, list) or len(limits) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in limits) or not limits[0] < limits[1]):
        raise FacetValidationError(f"{where}.limits is [low, high] with low < high, got {limits!r}")
    return {"name": name, "limits": [float(limits[0]), float(limits[1])], "label": str(raw.get("label") or "").strip()}


def validate_facet(facet: Any, jobs: Any = None) -> dict[str, Any]:
    """Validate one faceting entry and return its normalized form (legend mode auto reads jobs)."""
    if not isinstance(facet, Mapping):
        raise FacetValidationError(f"faceting entries must be mappings, got {type(facet).__name__}")

    facet_id = _as_int(facet.get("id"), name="id", facet_id=facet.get("id"), minimum=0)
    name = str(facet.get("name", f"facet_{facet_id}")).strip() or f"facet_{facet_id}"
    if "style" in facet:
        raise FacetValidationError(f"facet id={facet_id} style moved to {MOVED_KEYS['style']}")

    raw_panels = facet.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        raise FacetValidationError(f"facet id={facet_id} requires a non-empty panels list")

    panels: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_panels, start=1):
        if not isinstance(raw, Mapping):
            raise FacetValidationError(f"facet id={facet_id} panel {index} must be a mapping")
        if raw.get("file_id") is None:
            raise FacetValidationError(f"facet id={facet_id} panel {index} has no file_id")
        file_id = _as_int(raw.get("file_id"), name=f"panel {index} file_id", facet_id=facet_id, minimum=0)
        if any(panel["file_id"] == file_id for panel in panels):
            raise FacetValidationError(f"facet id={facet_id}: file_id {file_id} is used twice")
        address = str(raw.get("address") or "").strip()
        if not address:
            raise FacetValidationError(f"facet id={facet_id} file_id {file_id} has no address")
        path = Path(address)
        if path.is_absolute() or ".." in path.parts:
            raise FacetValidationError(
                f"facet id={facet_id} file_id {file_id}: address must be relative, inside "
                f"the project's graphs directory (or artifacts/... into a run's graphs folder), got {address!r}"
            )
        orientation = str(raw.get("orientation", "landscape")).strip().lower()
        if orientation not in VALID_ORIENTATIONS:
            raise FacetValidationError(
                f"facet id={facet_id} file_id {file_id} orientation={orientation!r} invalid. "
                f"Allowed: {sorted(VALID_ORIENTATIONS)}"
            )
        label_position = str(raw.get("label_position", "bottom")).strip().lower()
        if label_position not in VALID_LABEL_POSITIONS:
            raise FacetValidationError(
                f"facet id={facet_id} file_id {file_id} label_position={label_position!r} invalid. "
                f"Allowed: {list(VALID_LABEL_POSITIONS)}"
            )
        label = raw.get("label")
        panels.append({"file_id": file_id, "address": address, "orientation": orientation,
                       "label": None if label is None else str(label).strip(), "label_position": label_position})

    arrangement = facet.get("arrangement", {}) or {}
    if not isinstance(arrangement, Mapping):
        raise FacetValidationError(f"facet id={facet_id} arrangement must be a mapping")
    # fit: height (default) every row panel_height tall; width every row scaled to the widest row's width, so
    # stacked figures meet both margins (owner 2026-10-27); natural every panel at its own size in inches, so
    # the text of every panel prints at one size (owner 2026-10-27: Fig. 5.ii's six heat maps)
    fit = str(arrangement.get("fit") or "height").strip().lower()
    if fit not in ("height", "width", "natural"):
        raise FacetValidationError(
            f"facet id={facet_id} arrangement.fit={fit!r} invalid. Allowed: ['height', 'natural', 'width']")
    mode = str(arrangement.get("mode", "horizontal")).strip().lower()
    if mode not in VALID_ARRANGEMENTS:
        raise FacetValidationError(
            f"facet id={facet_id} arrangement.mode={mode!r} invalid. "
            f"Allowed: {sorted(VALID_ARRANGEMENTS)}"
        )
    known = [panel["file_id"] for panel in panels]
    order_raw = arrangement.get("order")
    if order_raw is None:
        order_raw = list(known)
    if not isinstance(order_raw, list) or not order_raw:
        raise FacetValidationError(
            f"facet id={facet_id} arrangement.order must be a non-empty list of file_ids"
        )
    nested = any(isinstance(item, list) for item in order_raw)
    if nested and not all(isinstance(item, list) and item for item in order_raw):
        raise FacetValidationError(
            f"facet id={facet_id} arrangement.order mixes file_ids and rows; write "
            "either [1, 2, 3] or [[1, 2], [3]]"
        )
    if nested and mode != "grid":
        raise FacetValidationError(
            f"facet id={facet_id} arrangement.order with rows needs mode: grid, not {mode!r}"
        )

    def ids(items: list[Any]) -> list[int]:
        return [
            _as_int(item, name="arrangement.order entry", facet_id=facet_id, minimum=0)
            for item in items
        ]

    if nested:
        if arrangement.get("columns") is not None:
            raise FacetValidationError(
                f"facet id={facet_id}: arrangement.order already names the rows; remove columns"
            )
        rows = [ids(row) for row in order_raw]
    else:
        flat = ids(order_raw)
        if mode == "grid":
            columns = _as_int(
                arrangement.get("columns", 2), name="arrangement.columns", facet_id=facet_id
            )
            rows = [flat[i:i + columns] for i in range(0, len(flat), columns)]
        elif arrangement.get("columns") is not None:
            raise FacetValidationError(
                f"facet id={facet_id} arrangement.columns belongs to mode: grid, not {mode!r}"
            )
        elif mode == "horizontal":
            rows = [flat]
        else:
            rows = [[file_id] for file_id in flat]

    order = [file_id for row in rows for file_id in row]
    unknown = [file_id for file_id in order if file_id not in known]
    if unknown:
        raise FacetValidationError(
            f"facet id={facet_id} arrangement.order names file_ids {unknown} "
            f"that no panel has; panels have {known}"
        )
    if len(set(order)) != len(order):
        raise FacetValidationError(
            f"facet id={facet_id} arrangement.order lists a file_id twice: {order}"
        )

    by_id = {panel["file_id"]: panel["address"] for panel in panels}
    legend = _legend(facet.get("legend"), facet_id=facet_id, first_address=by_id[order[0]], jobs=jobs)

    export = facet.get("export", {}) or {}
    if not isinstance(export, Mapping):
        raise FacetValidationError(f"facet id={facet_id} export must be a mapping")
    formats = export.get("formats", ["svg"])     # owner 2026-10-27: SVG only by default
    if isinstance(formats, str):
        formats = [formats]
    formats = list(dict.fromkeys(str(f).strip().lower().lstrip(".") for f in formats or []))
    invalid = [f for f in formats if f not in VALID_FACET_FORMATS]
    if invalid or not formats:
        raise FacetValidationError(
            f"facet id={facet_id} export.formats={formats} invalid. "
            f"Allowed: {sorted(VALID_FACET_FORMATS)}"
        )

    if "dpi" in export:
        raise FacetValidationError(f"facet id={facet_id} export.dpi moved to {MOVED_KEYS['export.dpi']}")
    png = export.get("png", {}) or {}
    if not isinstance(png, Mapping):
        raise FacetValidationError(f"facet id={facet_id} export.png must be a mapping")
    svg = export.get("svg", {}) or {}
    if not isinstance(svg, Mapping):
        raise FacetValidationError(f"facet id={facet_id} export.svg must be a mapping")
    on_error = str(svg.get("on_error", "warn")).strip().lower()
    if on_error not in VALID_SVG_ON_ERROR:
        raise FacetValidationError(
            f"facet id={facet_id} export.svg.on_error={on_error!r} invalid. Allowed: {sorted(VALID_SVG_ON_ERROR)}"
        )
    rules = facet.get("text_rules", []) or []
    if not isinstance(rules, list):
        raise FacetValidationError(f"facet id={facet_id} text_rules must be a list")

    return {
        "id": facet_id,
        "name": name,
        "enabled": bool(facet.get("enabled", True)),
        "output_file_name": str(facet.get("output_file_name", name)).strip() or name,
        "arrangement": {"mode": mode, "order": order, "rows": rows, "fit": fit},
        "legend": legend,
        "panel_height": _as_float(facet.get("panel_height"), name="panel_height", facet_id=facet_id, minimum=0.5, default=3.2),
        "spacing": _as_float(facet.get("spacing"), name="spacing", facet_id=facet_id, minimum=0.0, default=0.15),
        "panels": panels,
        "export": {
            "formats": formats,
            "png": {"dpi": _as_int(png.get("dpi", 300), name="export.png.dpi", facet_id=facet_id, minimum=72)},
            "svg": {"sanitize": bool(svg.get("sanitize", True)), "validate_xml": bool(svg.get("validate_xml", True)),
                    "on_error": on_error},
            "margin": _as_float(export.get("margin"), name="export.margin", facet_id=facet_id, minimum=0.0, default=0.05),
        },
        "text_rules": [_text_rule(rule, facet_id=facet_id) for rule in rules],
    }


def get_facet_by_id(facets: Any, face_id: int, jobs: Any = None) -> dict[str, Any]:
    if not isinstance(facets, list) or not facets:
        raise FacetValidationError(
            "graph_config.yaml has no faceting entries; add a top-level 'faceting:' list"
        )
    matches = [f for f in facets if isinstance(f, Mapping) and str(f.get("id")) == str(face_id)]
    if not matches:
        known = [f.get("id") for f in facets if isinstance(f, Mapping)]
        raise FacetValidationError(f"no facet with id={face_id}; defined ids: {known}")
    if len(matches) > 1:
        raise FacetValidationError(f"facet id={face_id} is defined {len(matches)} times")
    return validate_facet(matches[0], jobs)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def place_panels(facet: Mapping[str, Any]) -> list[Panel]:
    """Give the panels their cells: rows from top to bottom, each from left to right."""
    arrangement = facet["arrangement"]
    by_id = {spec["file_id"]: spec for spec in facet["panels"]}

    left_out = [file_id for file_id in by_id if file_id not in arrangement["order"]]
    if left_out:
        print(
            f"[Faceting][WARN] facet id={facet['id']}: file_ids {left_out} are not in "
            "arrangement.order and are not drawn."
        )

    panels: list[Panel] = []
    for row_number, row in enumerate(arrangement["rows"], start=1):
        for column_number, file_id in enumerate(row, start=1):
            spec = by_id[file_id]
            panels.append(Panel(
                file_id=file_id,
                file=spec["address"],
                orientation=spec["orientation"],
                row=row_number,
                column=column_number,
                label=spec.get("label"),
                label_position=spec.get("label_position", "bottom"),
            ))
    return panels


def _stem(graph_root: Path, project_root: Path, file: str) -> Path | None:
    """Address of the panel without its extension, or None if no such file exists."""
    relative = Path(file)
    if relative.suffix.lower() in PANEL_SUFFIXES:
        relative = relative.with_suffix("")
    for base in (graph_root, project_root):
        stem = base / relative
        if any(stem.with_name(stem.name + suffix).is_file() for suffix in PANEL_SUFFIXES):
            return stem
    return None


def _sibling(stem: Path, suffix: str) -> Path:
    return stem.with_name(stem.name + suffix)


def _pdf_reader():
    try:
        import pypdf
    except Exception:
        return None
    return pypdf


def _aspect(stem: Path) -> float:
    png = _sibling(stem, ".png")
    if png.is_file():
        with Image.open(png) as image:
            width, height = image.size
        return width / height
    svg = _sibling(stem, ".svg")
    if svg.is_file():
        width, height = _svg_size(svg.read_text(encoding="utf-8"))
        return width / height
    pypdf = _pdf_reader()
    pdf = _sibling(stem, ".pdf")
    if pypdf is not None and pdf.is_file():
        box = pypdf.PdfReader(str(pdf)).pages[0].mediabox
        return float(box.width) / float(box.height)
    raise FacetValidationError(
        f"cannot read the size of {stem.name}: export the graph as svg or png "
        "(or install pypdf to read the pdf)"
    )


_SVG_ROOT = re.compile(r"<svg\b[^>]*>", re.S)


def _natural_size(stem: Path) -> tuple[float, float] | None:
    """The panel's own (width, height) in inches: an SVG's or a PDF's page; None for a PNG (pixels only)."""
    svg = _sibling(stem, ".svg")
    if svg.is_file():
        width, height = _svg_size(svg.read_text(encoding="utf-8"))
        return width / 72.0, height / 72.0
    pypdf = _pdf_reader()
    pdf = _sibling(stem, ".pdf")
    if pypdf is not None and pdf.is_file():
        box = pypdf.PdfReader(str(pdf)).pages[0].mediabox
        return float(box.width) / 72.0, float(box.height) / 72.0
    return None


def _svg_size(text: str) -> tuple[float, float]:
    """(width, height) of an SVG in pt, from its root's width / height (pt or unitless), else its viewBox."""
    root = _SVG_ROOT.search(text)
    if not root:
        raise FacetValidationError("an SVG panel without an <svg> root")
    tag = root.group(0)
    size = [re.search(rf'\b{key}="([0-9.]+)(pt|px)?"', tag) for key in ("width", "height")]
    if all(size):
        return float(size[0].group(1)), float(size[1].group(1))
    box = re.search(r'viewBox="[-0-9.]+ [-0-9.]+ ([0-9.]+) ([0-9.]+)"', tag)
    if not box:
        raise FacetValidationError("an SVG panel without width / height or a viewBox")
    return float(box.group(1)), float(box.group(2))


def _nested_svg(text: str, prefix: str, x: float, y: float, w: float, h: float) -> str:
    """One panel's SVG as an <svg> element of the facet, at (x, y) pt from the top left, w x h pt: its
    declaration and DOCTYPE dropped, every id prefixed (two panels' glyph and clip-path ids never clash)."""
    text = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>", "", text, flags=re.S).strip()
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    text = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', text)
    text = re.sub(r'((?:xlink:)?href)="#([^"]+)"', lambda m: f'{m.group(1)}="#{prefix + m.group(2) if m.group(2) in ids else m.group(2)}"', text)
    text = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{prefix + m.group(1) if m.group(1) in ids else m.group(1)})", text)
    root = _SVG_ROOT.search(text)
    tag = re.sub(r'\s(?:width|height|x|y|preserveAspectRatio)="[^"]*"', "", root.group(0))
    if "viewBox=" not in tag:
        pw, ph = _svg_size(root.group(0))
        tag = tag[:-1] + f' viewBox="0 0 {pw:g} {ph:g}">'
    tag = tag[:-1] + f' x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" preserveAspectRatio="xMidYMid meet">'
    return text[:root.start()] + tag + text[root.end():]


def _vector_svg(layout: Layout, panels: list[Panel], path: Path) -> bool:
    """Write an SVG whose panels are the panels' own SVGs, nested (owner 2026-10-27: SVG only, all vector).
    False when a panel has no .svg."""
    if not all(_sibling(p.stem, ".svg").is_file() for p in panels):
        return False
    fig = _figure(layout)
    buffer = io.StringIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches=None)
    finally:
        plt.close(fig)
    page = buffer.getvalue()
    nested = []
    for k, (panel, (x, y, w, h)) in enumerate(zip(panels, layout.boxes), start=1):
        top = layout.height - y - h                         # the layout counts from the bottom, SVG from the top
        nested.append(_nested_svg(_sibling(panel.stem, ".svg").read_text(encoding="utf-8"), f"panel{k}_",
                                  x * 72.0, top * 72.0, w * 72.0, h * 72.0))
    end = page.rindex("</svg>")
    path.write_text(page[:end] + "\n".join(nested) + "\n" + page[end:], encoding="utf-8")
    return True


def _text_kwargs(rule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "color": rule["color"],
        "fontfamily": rule["family"],
        "fontsize": rule["size"],
        "fontweight": "bold" if rule["bold"] else "normal",
        "fontstyle": "italic" if rule["italic"] else "normal",
    }


@dataclass
class Layout:
    width: float
    height: float
    boxes: list[tuple[float, float, float, float]]  # x, y, w, h in inches, origin bottom-left
    texts: list[tuple[float, float, str, dict[str, Any]]]  # x, y in inches
    legend: dict[str, Any] | None = None  # entries, anchor (x, y in inches), columns, ...


def build_layout(facet: Mapping[str, Any], panels: list[Panel]) -> Layout:
    height = float(facet["panel_height"])
    spacing = float(facet["spacing"])
    margin = float(facet["export"]["margin"])
    rules = {rule["target"]: rule for rule in facet["text_rules"] if rule["text"]}

    def line(target: str) -> float:
        rule = rules.get(target)
        return 0.0 if rule is None else rule["size"] / 72.0 * 1.6

    for panel in panels:
        fits = (panel.aspect >= 1.0) == (panel.orientation == "landscape")
        if not fits:
            print(
                f"[Faceting][WARN] {panel.file}: the file is "
                f"{'landscape' if panel.aspect >= 1.0 else 'portrait'} but the panel is "
                f"declared {panel.orientation}; it is fitted inside a {panel.orientation} cell."
            )
        panel.cell_width = height * (panel.aspect if fits else FALLBACK_ASPECT[panel.orientation])
    own_size = (facet.get("arrangement") or {}).get("fit") == "natural"
    if own_size:
        unsized = [panel.file for panel in panels if panel.size is None]
        if unsized:
            raise FacetValidationError(
                f"facet id={facet.get('id')} arrangement.fit natural needs SVG or PDF panels (a size in inches): {unsized}")
        for panel in panels:
            panel.cell_width = panel.size[0]

    n_rows = max(panel.row for panel in panels)
    rows = {r: sorted((p for p in panels if p.row == r), key=lambda p: p.column) for r in range(1, n_rows + 1)}
    # Rows of equal length share column widths; ragged rows are centred.
    regular = len({len(row) for row in rows.values()}) == 1
    if regular:
        n_columns = len(rows[1])
        column_width = [max(row[c].cell_width for row in rows.values()) for c in range(n_columns)]
        widths = {r: list(column_width) for r in rows}
    else:
        widths = {r: [p.cell_width for p in row] for r, row in rows.items()}
    row_width = {r: sum(w) + spacing * (len(w) - 1) for r, w in widths.items()}
    heights = {r: height for r in rows}
    if own_size:
        # each row as tall as its tallest panel; the columns already the widest panel's width
        heights = {r: max(p.size[1] for p in row) for r, row in rows.items()}
    if (facet.get("arrangement") or {}).get("fit") == "width":
        # every row scaled to the widest row's own width (its panels' widths, not the shared columns)
        natural = {r: sum(p.cell_width for p in row) for r, row in rows.items()}
        target = max(natural[r] + spacing * (len(row) - 1) for r, row in rows.items())
        for r, row in rows.items():
            scale = (target - spacing * (len(row) - 1)) / natural[r]
            widths[r] = [p.cell_width * scale for p in row]
            heights[r] = height * scale
            row_width[r] = target

    legend_cfg = facet.get("legend") or {}
    legend_on = bool(legend_cfg.get("enabled")) and bool(legend_cfg.get("entries"))
    legend_rule = rules.get("legend_text") or next(
        (rule for rule in facet["text_rules"] if rule["target"] == "legend_text"), None
    )
    legend_size = legend_rule["size"] if legend_rule else 8
    legend_line = legend_size / 72.0 * 1.9
    legend_top = legend_bottom = legend_right = spectrum = 0.0
    if legend_on:
        entries = legend_cfg["entries"]
        caption_lines = legend_cfg["caption"].count("\n") + 1 if legend_cfg.get("caption") else 0
        n_lines = -(-len(entries) // int(legend_cfg["columns"])) + (1 if legend_cfg["title"] else 0) + caption_lines
        colormap = legend_cfg.get("colormap")
        # legend.colormap: the spectrum (0.13 in), its ticks and its label, above the entries
        spectrum = (0.13 + legend_line * (2 if colormap["label"] else 1) + 0.06) if colormap else 0.0
        band = n_lines * legend_line + 0.12 + spectrum
        if legend_cfg["position"] == "top":
            legend_top = band
        elif legend_cfg["position"] == "bottom":
            legend_bottom = band
        else:
            longest = max([len(entry["label"]) for entry in entries] + [len(legend_cfg.get("caption") or "")])
            legend_right = (longest * legend_size * 0.62 / 72.0 + 0.55) * int(legend_cfg["columns"]) + 0.15

    left = margin + line("y_label")
    bottom = margin + line("x_label") + legend_bottom
    # Panel labels: each panel's own (label, label_position), else the panel_label rule's text below
    # it; their font is the panel_label rule's. A band above / below every row holds them.
    label_rule = next((rule for rule in facet["text_rules"] if rule["target"] == "panel_label"), None)
    label_kwargs = _text_kwargs(label_rule) if label_rule else {
        "color": "black", "fontfamily": "serif", "fontsize": 10, "fontweight": "normal", "fontstyle": "normal"}

    def panel_label(panel: Panel) -> tuple[str, str] | None:
        if panel.label:
            return panel.label, panel.label_position
        if label_rule and label_rule["text"] and panel.label is None:
            return label_rule["text"], "bottom"
        return None

    labelled = [panel_label(panel) for panel in panels]
    band = label_kwargs["fontsize"] / 72.0 * 1.6
    top_band = band if any(x and x[1].startswith("top") for x in labelled) else 0.0
    bottom_band = band if any(x and x[1].startswith("bottom") for x in labelled) else 0.0
    # label_position left (owner 2026-10-27: the stacked figures indexed I, II, ... on the left): a band
    # left of the grid, 1.8 times the label size (owner 2026-10-27: the index close to its row); each label beside
    # its own panel's left edge, centred on it
    left_band = label_kwargs["fontsize"] / 72.0 * 1.8 if any(x and x[1] == "left" for x in labelled) else 0.0
    left += left_band
    row_heights = {r: heights[r] + top_band + bottom_band for r in rows}
    top = margin + line("title") + legend_top

    grid_width = max(row_width.values())
    grid_height = sum(row_heights.values()) + spacing * (n_rows - 1)
    total_width = left + grid_width + legend_right + margin
    total_height = bottom + grid_height + top

    boxes: list[tuple[float, float, float, float]] = []
    texts: list[tuple[float, float, str, dict[str, Any]]] = []

    for index, panel in enumerate(panels):
        row_widths = widths[panel.row]
        row_left = left + (grid_width - row_width[panel.row]) / 2.0
        cell_x = row_left + sum(row_widths[: panel.column - 1]) + spacing * (panel.column - 1)
        cell_top = total_height - top - sum(row_heights[r] + spacing for r in range(1, panel.row))
        image_top = cell_top - top_band
        cell_w = row_widths[panel.column - 1]
        row_h = heights[panel.row]
        # Largest size of the file that fits its own cell, centred in the column.
        width = min(panel.cell_width * row_h / height, row_h * panel.aspect)     # fit width scales the row
        if own_size:
            width = panel.size[0]                                                # its own size
        drawn_height = width / panel.aspect
        x = cell_x + (cell_w - width) / 2.0
        # natural: the panels of a row top-aligned, their titles on one line; else centred in the row
        y = image_top - drawn_height if own_size else image_top - row_h + (row_h - drawn_height) / 2.0
        boxes.append((x, y, width, drawn_height))

        if labelled[index] is not None:
            template, position = labelled[index]
            text = template.format(
                letter=string.ascii_lowercase[index % 26],
                index=index + 1,
                roman=_roman(index + 1),
                file_id=panel.file_id,
                file=Path(panel.file).stem,
            )
            if position == "left":
                texts.append((x - left_band / 2.0, y + drawn_height / 2.0, text, {**label_kwargs, "ha": "center", "va": "center"}))
                continue
            side = position.partition("-")[2] or "center"
            text_x = {"left": cell_x, "center": cell_x + cell_w / 2.0, "right": cell_x + cell_w}[side]
            text_y = cell_top - top_band / 2.0 if position.startswith("top") else image_top - row_h - bottom_band / 2.0
            texts.append((text_x, text_y, text, {**label_kwargs, "ha": side, "va": "center"}))

    if "title" in rules:
        texts.append((
            left + grid_width / 2.0, total_height - margin - line("title") / 2.0,
            rules["title"]["text"],
            {**_text_kwargs(rules["title"]), "ha": "center", "va": "center"},
        ))
    if "x_label" in rules:
        texts.append((
            left + grid_width / 2.0, margin + legend_bottom + line("x_label") / 2.0,
            rules["x_label"]["text"],
            {**_text_kwargs(rules["x_label"]), "ha": "center", "va": "center"},
        ))
    if "y_label" in rules:
        texts.append((
            margin + line("y_label") / 2.0, bottom + grid_height / 2.0,
            rules["y_label"]["text"],
            {**_text_kwargs(rules["y_label"]), "ha": "center", "va": "center", "rotation": 90},
        ))

    legend = None
    if legend_on:
        # the entries and the caption under them centred together in the band (owner 2026-10-28: Fig. 5.ii's caption
        # met the heat maps' titles)
        lift = legend_line * caption_lines / 2.0
        if legend_cfg["position"] == "top":
            band_top = total_height - margin - line("title")
            anchor = (left + grid_width / 2.0, band_top - spectrum - (legend_top - spectrum) / 2.0 + lift)
        elif legend_cfg["position"] == "bottom":
            band_top = margin + legend_bottom
            anchor = (left + grid_width / 2.0, margin + (legend_bottom - spectrum) / 2.0 + lift)
        else:
            band_top = 0.0
            anchor = (left + grid_width + legend_right / 2.0, bottom + grid_height / 2.0)
        legend = {**legend_cfg, "anchor": anchor, "size": legend_size, "rule": legend_rule,
                  "spectrum_top": band_top, "spectrum_width": min(grid_width * 0.6, 4.5)}

    return Layout(total_width, total_height, boxes, texts, legend)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _figure(layout: Layout) -> Any:
    fig = plt.figure(figsize=(layout.width, layout.height))
    for x, y, text, kwargs in layout.texts:
        fig.text(x / layout.width, y / layout.height, text, **kwargs)
    _draw_legend(fig, layout)
    return fig


def _draw_legend(fig: Any, layout: Layout) -> None:
    legend = layout.legend
    if not legend:
        return

    handle = legend.get("handle") or {"shape": "line", "marker": None, "size": 1.5}
    handles = []
    for entry in legend["entries"]:
        if handle["shape"] == "patch":
            handles.append(Patch(facecolor=entry["color"], edgecolor="none", label=entry["label"]))
        elif handle["shape"] == "marker":
            handles.append(Line2D([0], [0], color=entry["color"], linestyle="none", marker=handle["marker"],
                                  markersize=handle["size"], label=entry["label"]))
        else:
            handles.append(Line2D([0], [0], color=entry["color"], linewidth=handle["size"], label=entry["label"]))

    rule = legend.get("rule")
    prop: dict[str, Any] = {"size": legend["size"]}
    if rule:
        prop.update({
            "family": rule["family"],
            "weight": "bold" if rule["bold"] else "normal",
            "style": "italic" if rule["italic"] else "normal",
        })
    x, y = legend["anchor"]
    if legend.get("colormap"):
        _draw_spectrum(fig, layout, legend, prop)
    drawn = fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(x / layout.width, y / layout.height),
        ncol=int(legend["columns"]),
        frameon=bool(legend["frame"]),
        title=legend["title"] or None,
        prop=prop,
        handlelength=1.8,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    if rule:
        for text in drawn.get_texts():
            text.set_color(rule["color"])
    if legend.get("caption"):
        # under the legend, centred on it, in the legend's font (not bold); **bold** and *italic* pieces in that font
        # too (owner 2026-10-28: Fig. 5.ii "the number in a cell", "in colour", "in grey")
        # a newline starts the next line, each centred
        renderer = fig.canvas.get_renderer()
        box = drawn.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        gap = legend["size"] / 72.0 * 0.6 / layout.height
        step = legend["size"] / 72.0 * 1.35 / layout.height
        for n, line in enumerate(legend["caption"].split("\n")):
            pieces = [(t[2:-2], "bold", "normal") if t.startswith("**") else (t[1:-1], "normal", "italic")
                      if t.startswith("*") else (t, "normal", "normal")
                      for t in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", line) if t]
            texts = [fig.text(0.0, box.y0 - gap - n * step, t, ha="left", va="top", fontsize=legend["size"],
                              fontweight=weight, fontstyle=style, color=rule["color"] if rule else "black",
                              fontfamily=prop.get("family", "serif")) for t, weight, style in pieces]
            widths = [t.get_window_extent(renderer).width / fig.dpi / layout.width for t in texts]
            x = (box.x0 + box.x1) / 2.0 - sum(widths) / 2.0
            for t, w in zip(texts, widths):
                t.set_x(x)
                x += w


def _draw_spectrum(fig: Any, layout: Layout, legend: Mapping[str, Any], prop: Mapping[str, Any]) -> None:
    """legend.colormap: a horizontal colour bar centred over the legend's entries, its ticks and label under it."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    spec = legend["colormap"]
    width = legend["spectrum_width"]
    x = legend["anchor"][0] - width / 2.0
    y = legend["spectrum_top"] - 0.03 - 0.13
    ax = fig.add_axes([x / layout.width, y / layout.height, width / layout.width, 0.13 / layout.height])
    mappable = ScalarMappable(norm=Normalize(*spec["limits"]), cmap=plt.get_cmap(spec["name"]))
    bar = fig.colorbar(mappable, cax=ax, orientation="horizontal")
    bar.ax.tick_params(labelsize=legend["size"], length=2.5, width=0.6)
    for tick in bar.ax.get_xticklabels():
        tick.set_fontfamily(prop.get("family", "serif"))
    bar.outline.set_linewidth(0.6)
    if spec["label"]:
        bar.set_label(spec["label"], fontsize=legend["size"], fontfamily=prop.get("family", "serif"),
                      fontweight=prop.get("weight", "normal"), labelpad=2.0)


def _raster_figure(layout: Layout, panels: list[Panel]) -> Any:
    fig = _figure(layout)
    for panel, (x, y, w, h) in zip(panels, layout.boxes):
        png = _sibling(panel.stem, ".png")
        if not png.is_file():
            plt.close(fig)
            raise FacetValidationError(
                f"{panel.file}: no .png next to it for a png / pdf facet; add png to the export formats of its "
                "graph job, or export the facet as svg (its panels' SVGs nested)"
            )
        ax = fig.add_axes([x / layout.width, y / layout.height, w / layout.width, h / layout.height])
        with Image.open(png) as image:
            ax.imshow(image.convert("RGBA"), interpolation="lanczos", aspect="auto")
        ax.set_axis_off()
    return fig


def _vector_pdf(layout: Layout, panels: list[Panel], path: Path) -> bool:
    """Write a PDF whose panels are the vector PDFs. False when that is not possible."""
    pypdf = _pdf_reader()
    if pypdf is None or not all(_sibling(p.stem, ".pdf").is_file() for p in panels):
        return False

    fig = _figure(layout)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="pdf", bbox_inches=None)
    finally:
        plt.close(fig)
    buffer.seek(0)

    page = pypdf.PdfReader(buffer).pages[0]
    for panel, (x, y, w, h) in zip(panels, layout.boxes):
        source = pypdf.PdfReader(str(_sibling(panel.stem, ".pdf"))).pages[0]
        box = source.mediabox
        scale = min(w * 72.0 / float(box.width), h * 72.0 / float(box.height))
        shift_x = x * 72.0 - float(box.left) * scale
        shift_y = y * 72.0 - float(box.bottom) * scale
        page.merge_transformed_page(
            source,
            pypdf.Transformation().scale(scale, scale).translate(shift_x, shift_y),
        )

    writer = pypdf.PdfWriter()
    writer.add_page(page)
    with path.open("wb") as handle:
        writer.write(handle)
    return True


def render_facet(
    facet: Mapping[str, Any],
    *,
    graph_root: Path,
    project_root: Path,
) -> list[Path]:
    """Compose one facet and return the files written."""
    panels = place_panels(facet)

    missing = []
    for panel in panels:
        panel.stem = _stem(Path(graph_root), Path(project_root), panel.file)
        if panel.stem is None:
            missing.append(panel.file)
    if missing:
        raise FacetValidationError(
            f"facet id={facet['id']}: panel address(es) not found under {graph_root}: {missing}. "
            "Run their graph jobs first (make GRAPH_ID=<id> graph)."
        )
    for panel in panels:
        panel.aspect = _aspect(panel.stem)
        panel.size = _natural_size(panel.stem)

    register_tex_fonts()   # so that font.family: "Latin Modern Roman" of a text rule resolves
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "savefig.bbox": None})
    layout = build_layout(facet, panels)

    out_dir = Path(graph_root) / "facets" / f"facet_{int(facet['id']):03d}_{_safe(facet['name'])}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / _safe(facet["output_file_name"])

    written: list[Path] = []
    for fmt in facet["export"]["formats"]:
        path = out_base.with_name(out_base.name + f".{fmt}")
        if fmt == "pdf" and _vector_pdf(layout, panels, path):
            written.append(path)
            continue
        if fmt == "svg" and _vector_svg(layout, panels, path):
            if sanitize_and_validate_svg(path, facet):
                written.append(path)
            continue
        if fmt == "pdf":
            print(
                "[Faceting] pypdf is not installed (or a panel has no .pdf): the PDF "
                "embeds the PNG panels. 'pip install pypdf' gives vector panels."
            )
        fig = _raster_figure(layout, panels)
        try:
            fig.savefig(path, format=fmt, dpi=facet["export"]["png"]["dpi"], bbox_inches=None)
        finally:
            plt.close(fig)
        if fmt == "svg" and not sanitize_and_validate_svg(path, facet):
            continue
        written.append(path)

    return written
