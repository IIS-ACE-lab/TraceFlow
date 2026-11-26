# services/graph_manager/raw_exporter.py
"""
Raw data exporter for TraceFlow GraphManager.

Purpose:
- Save the exact data used to generate graphs.
- Keep graph outputs reproducible.
- Support scalar history, curve arrays, tables, and metadata summaries.
- Never crash graph generation because raw export failed.

Controlled by graph_config.yaml:

export:
  save_raw_plot_data: true

Current schema:
  output_file_name: "..."

Backward compatibility:
  output_name is still accepted as fallback.

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


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

    out = "".join(clean).strip("_")
    return out or "unnamed"


def should_export_raw(job: Mapping[str, Any]) -> bool:
    export_cfg = job.get("export", {}) or {}

    if not isinstance(export_cfg, Mapping):
        return False

    value = export_cfg.get("save_raw_plot_data", False)

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_output_file_name(job: Mapping[str, Any]) -> str:
    """
    Official field:
        output_file_name

    Backward-compatible fallback:
        output_name
    """

    return safe_stem(job.get("output_file_name", job.get("output_name", "graph")))


def ensure_raw_dir(job_output_dir: Path) -> Path:
    raw_dir = Path(job_output_dir) / "raw_plot_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, pd.Series):
        return obj.to_list()

    return str(obj)


def export_json(data: Any, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)

    return out_path


def export_dataframe(df: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()

    if suffix == ".json":
        df.to_json(out_path, orient="records", indent=2)
    elif suffix == ".jsonl":
        df.to_json(out_path, orient="records", lines=True)
    else:
        df.to_csv(out_path, index=False)

    return out_path


def export_array(array: Any, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(array)
    suffix = out_path.suffix.lower()

    if suffix == ".npy":
        np.save(out_path, arr)
    elif suffix == ".json":
        export_json(arr.tolist(), out_path)
    else:
        pd.DataFrame(arr).to_csv(out_path, index=False)

    return out_path


def export_series(
    *,
    job: Mapping[str, Any],
    job_output_dir: Path,
    data: Any,
    model: str | None = None,
    process: str | None = None,
    metric: str | None = None,
    label: str | None = None,
) -> list[Path]:
    """
    Export one plot series.

    Supported:
    - pandas DataFrame -> csv + json
    - numpy array/list/tuple -> csv + npy
    - scalar/dict metadata -> json

    Returns:
        List of written paths. Empty if disabled or failed.
    """

    if not should_export_raw(job):
        return []

    written: list[Path] = []

    try:
        raw_dir = ensure_raw_dir(job_output_dir)

        parts = [
            get_output_file_name(job),
            safe_stem(metric or job.get("metric", "metric")),
        ]

        if model:
            parts.append(safe_stem(model))

        if process:
            parts.append(safe_stem(process))

        if label:
            parts.append(safe_stem(label))

        stem = "__".join(parts)

        if isinstance(data, pd.DataFrame):
            written.append(export_dataframe(data, raw_dir / f"{stem}.csv"))
            written.append(export_dataframe(data, raw_dir / f"{stem}.json"))

        elif isinstance(data, np.ndarray):
            written.append(export_array(data, raw_dir / f"{stem}.csv"))
            written.append(export_array(data, raw_dir / f"{stem}.npy"))

        elif isinstance(data, (list, tuple)):
            arr = np.asarray(data)

            if arr.ndim >= 1 and arr.dtype != object:
                written.append(export_array(arr, raw_dir / f"{stem}.csv"))
                written.append(export_array(arr, raw_dir / f"{stem}.npy"))
            else:
                written.append(export_json(list(data), raw_dir / f"{stem}.json"))

        elif isinstance(data, Mapping):
            written.append(export_json(dict(data), raw_dir / f"{stem}.json"))

        else:
            written.append(export_json({"value": data}, raw_dir / f"{stem}.json"))

    except Exception as exc:
        print(f"[RawExporter][WARN] Failed to export raw plot data: {exc}")
        return []

    return written


def export_figure_data(fig: Any, out_base: Path, job: Mapping[str, Any]) -> Path | None:
    """
    raw_plot_data/<figure>.json beside the figure: exactly what it draws, read from the figure itself (every
    line's x and y, every bar, the legend, axis labels, limits and scales), so a number in a paper
    figure can be read back without the run folders. Called by plotters.save_figure for every figure
    of a job with export.save_raw_plot_data: true.
    """
    if not should_export_raw(job):
        return None
    try:
        from matplotlib.colors import to_hex

        axes = []
        for ax in fig.get_axes():
            legend = ax.get_legend()
            axes.append({
                "title": ax.get_title(), "x_label": ax.get_xlabel(), "y_label": ax.get_ylabel(),
                "x_scale": ax.get_xscale(), "y_scale": ax.get_yscale(),
                "x_limits": list(ax.get_xlim()), "y_limits": list(ax.get_ylim()),
                "legend": [t.get_text() for t in legend.get_texts()] if legend else [],
                "lines": [{"label": line.get_label(), "color": to_hex(line.get_color()),
                           "x": np.asarray(line.get_xdata(), dtype=float).tolist(),
                           "y": np.asarray(line.get_ydata(), dtype=float).tolist()}
                          for line in ax.get_lines() if not str(line.get_label()).startswith("_")],
                "bars": [{"x": float(p.get_x()), "y": float(p.get_y()), "width": float(p.get_width()),
                          "height": float(p.get_height()), "color": to_hex(p.get_facecolor())}
                         for p in ax.patches if hasattr(p, "get_width") and hasattr(p, "get_height")],
            })
        out_base = Path(out_base)
        return export_json({"figure": out_base.name, "job_id": job.get("id"), "axes": axes},
                           ensure_raw_dir(out_base.parent) / f"{safe_stem(out_base.name)}.json")
    except Exception as exc:
        print(f"[RawExporter][WARN] Failed to export the figure's data: {exc}")
        return None


def export_job_manifest(
    *,
    job: Mapping[str, Any],
    job_output_dir: Path,
    created_graphs: list[str | Path],
    missing_items: list[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path | None:
    """
    Export a manifest describing what the graph job produced.
    """

    if not should_export_raw(job):
        return None

    try:
        raw_dir = ensure_raw_dir(job_output_dir)

        export_cfg = job.get("export", {}) or {}
        legend_cfg = job.get("legend", {}) or {}
        style_cfg = job.get("style", {}) or {}

        manifest = {
            "job_id": job.get("id"),
            "job_name": job.get("name"),
            "job_type": job.get("type"),
            "mode": job.get("mode", "auto"),
            "plot_type": job.get("plot_type", "auto"),
            "metric": job.get("metric"),

            # New official field.
            "output_file_name": job.get("output_file_name", job.get("output_name")),

            # Backward-compatible alias for old consumers.
            "output_name": job.get("output_file_name", job.get("output_name")),

            "export_formats": list(export_cfg.get("formats", []))
            if isinstance(export_cfg, Mapping)
            else [],

            "created_graphs": [str(p) for p in created_graphs],
            "missing_items": list(missing_items or []),

            "legend": dict(legend_cfg) if isinstance(legend_cfg, Mapping) else {},
            "style": dict(style_cfg) if isinstance(style_cfg, Mapping) else {},

            "job": dict(job),
        }

        active_context = job.get("active_context")
        if isinstance(active_context, Mapping):
            manifest["active_context"] = dict(active_context)

        if extra:
            manifest["extra"] = dict(extra)

        return export_json(manifest, raw_dir / "manifest.json")

    except Exception as exc:
        print(f"[RawExporter][WARN] Failed to export graph manifest: {exc}")
        return None
