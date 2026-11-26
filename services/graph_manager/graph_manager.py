#!/usr/bin/env python3
# services/graph_manager/graph_manager.py
"""
Job-driven GraphManager for TraceFlow / Side Channel Core.

Reads graph jobs from:
    cfg["graph"]["jobs"]

Make:
    make ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=1 graph

Design:
- No separate cli.py
- Job-level mode only:
    mode: "auto"
- Job-level plot_type:
    plot_type: "line" | "scatter" | "line_scatter" | ...
- Missing metrics issue terminal warnings and continue
- model_overlay uses:
    models: [...]
    processes: [...]
- source picks the metrics directory (default attack_metrics):
    attack_metrics    metrics/attack_metrics            train | validation | test
    cross_validation  metrics/cross_validation/fold_NN  train | validation (fold mean +/- std)
    key_recovery      metrics/key_recovery              validation | test
- type: model_explanation draws what a trained model looks at, from its final
  checkpoint and test traces (services/graph_manager/explain.py): method
  weight_visualization | gradient | gradient_x_input | heatmap.
- job 0 is the only automated job: the model judge rewrites its top_models.
  Every other job lists its models itself; the names of all generated models
  are collected in <project>/artifacts/done_model_names.yaml.
- a models entry is a model_name or a model_hash. Before plotting, each is
  bound to its run of this ENV through artifacts/run_registry.json and
  artifacts/model_name_map.json (both must agree); the series are named by
  model_name.

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from services.side_channel_core.utils.config_loader import get_project_config
from services.side_channel_core.utils.logger import get_logger
from services.side_channel_core.utils.model_naming import _load_mapping
from services.side_channel_core.utils.path_utils import get_metrics_dir
from services.side_channel_core.utils.sweep_engine import load_registry

from services.graph_manager.faceting import PANEL_SUFFIXES, _stem, get_facet_by_id, place_panels, render_facet
from services.graph_manager.sources import facet_sources, job_sources, write_sources
from services.side_channel_core.utils.seal import recording
from services.graph_manager.graph_job_validator import (
    get_graph_job_by_id,
    normalize_process_name,
    resolve_model_refs,
)
from services.graph_manager.metric_loader import (
    LoadedMetric,
    combine_folds,
    fold_metric_dirs,
    load_metric,
    load_source_metric,
)
from services.graph_manager.missing_plotter import MissingMetric
from services.graph_manager.metric_router import route_metric
from services.graph_manager.plotters import (
    ge_average_curves,
    get_plot_type,
    plot_fold_heatmap,
    plot_scalar_line,
    plot_scalar_overlay,
    plot_curve,
    plot_curve_overlay,
    plot_distribution,
    plot_strip,
)
from services.graph_manager.attack_plotters import (
    plot_attack_metric,
    plot_fold_rank_curves,
    plot_key_candidate_scores,
)
from services.graph_manager.metric_catalog import print_metric_choices, print_model_choices
from services.graph_manager.missing_plotter import missing_to_dict
from services.graph_manager.plot_style import get_std_band, render_template
from services.graph_manager.job_paths import safe_name
from services.graph_manager.trace_budget import TraceBudgetError, cut_curve, final_ranks_at, test_metrics_dir


LOGGER = get_logger("GraphManager")


def combo_key(items: Iterable[str]) -> str:
    return "_".join(sorted(str(x) for x in items))


def output_file_name(job: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> str:
    """
    Official field:
        output_file_name

    Backward-compatible fallback:
        output_name
    """

    raw = job.get("output_file_name", job.get("output_name", "graph"))
    return safe_name(render_template(str(raw), context or {}))


def with_context(job: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a shallow job copy carrying active runtime context.

    Plotters can read:
        job["process"]
        job["active_context"]

    without mutating the validated job object.
    """

    out = dict(job)
    out["active_context"] = dict(context)

    if "process" in context:
        out["process"] = context["process"]

    if "model" in context:
        out["model"] = context["model"]

    if "model_name" in context:
        out["model_name"] = context["model_name"]

    return out


# Graph metric -> column recorded by every run.
FOLD_MEAN_METRICS = {
    "mean_accuracy": "accuracy",
    "mean_fit_time_s": "fit_time_s",
    "mean_fit_time_h": "fit_time_h",
    "mean_fit_time_per_epoch": "fit_time_per_epoch",
}


def relative_series(
    series: Mapping[str, pd.DataFrame],
    reference: Mapping[str, pd.DataFrame],
    column: str,
    *,
    label: str = "",
    job_id: Any = None,
) -> dict[str, pd.DataFrame]:
    """
    Every series divided by the reference run's series of the same metric
    (graph job relative:), read as the job reads its models: the reference
    becomes 1.0 and the values become ratios. For a time metric the hardware
    cancels when every run, the reference included, trained on one device.

    series keys are model names, or "model:metric" when the job draws several
    metrics; reference is keyed by metric ("" for a one-metric job). A
    reference with one value (read at its last or a selected epoch) divides
    every value; otherwise the values are divided epoch by epoch. The std over
    folds is divided by the same reference mean.
    """
    out: dict[str, pd.DataFrame] = {}
    for key, df in series.items():
        _, sep, metric = key.partition(":")
        ref = reference.get(metric if sep else "")
        if ref is None or column not in ref.columns or ref.empty:
            raise ValueError(f"graph job id={job_id}: relative reference ({label}) has no {metric or column} data")
        ref_values = pd.to_numeric(ref[column], errors="coerce")
        df = df.copy()
        if len(ref) == 1:
            denominator = float(ref_values.iloc[0])
        else:
            by_epoch = dict(zip(ref["epoch"], ref_values))
            denominator = df["epoch"].map(by_epoch)
            if denominator.isna().any():
                missing = sorted(df.loc[denominator.isna(), "epoch"].tolist())[:5]
                raise ValueError(
                    f"graph job id={job_id}: relative reference ({label}) has no value at epoch(s) {missing}"
                )
        if np.any(np.asarray(denominator, dtype=float) == 0):
            raise ValueError(f"graph job id={job_id}: relative reference ({label}) is zero, cannot divide")
        for col in (column, f"{column}_std"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") / denominator
        out[key] = df
    return out


def time_metric_seconds(metric: str) -> float | None:
    """Seconds per unit of a time metric (fit_time_*, training_time_s), None for any other metric."""
    name = str(metric).lower().replace("{", "").replace("}", "")
    if "fit_time" not in name and name not in ("training_time_s",):
        return None
    return 3600.0 if name.endswith("_h") else 1.0


def probe_relative_series(
    series: Mapping[str, pd.DataFrame],
    probes: Mapping[str, float],
    column: str,
    metric_of: Mapping[str, str],
    *,
    job_id: Any = None,
) -> dict[str, pd.DataFrame]:
    """
    relative: {rule: probe}. Every time series divided by the probe of its own
    run (seconds per step of utils/device_record.PROBE_SPEC on the device it
    trained on), so the value is in probe steps: a run of a fast and of a slow
    device that cost the same work read alike. series keys are model names or
    "model:metric"; metric_of gives each key's metric.
    """
    out: dict[str, pd.DataFrame] = {}
    for key, df in series.items():
        model = key.partition(":")[0]
        unit = time_metric_seconds(metric_of[key])
        if unit is None:
            raise ValueError(
                f"graph job id={job_id}: relative.rule probe divides time metrics; {metric_of[key]} is not one"
            )
        df = df.copy()
        for col in (column, f"{column}_std"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * unit / probes[model]
        out[key] = df
    return out


class GraphManager:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.cfg = get_project_config(project_name, validate_data=False)

        self.graph_cfg = self.cfg.get("graph", {}) or {}
        self.graph_runtime = (
            (self.cfg.get("visualization", {}) or {})
            .get("graph_manager", {}) or {}
        )

        if not self.graph_runtime.get("enabled", False):
            raise RuntimeError(
                "graph_manager.enabled=false in master.yaml. "
                "Set visualization.graph_manager.enabled=true."
            )

        self.jobs = self.graph_cfg.get("jobs", []) or []

        LOGGER.info("GraphManager initialized for project=%s", project_name)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def project_root(self) -> Path:
        return Path(self.cfg["project"]["root"]).resolve()

    def graph_root(self) -> Path:
        raw = (self.cfg.get("paths", {}) or {}).get("graphs", "graphs")
        p = Path(str(raw))

        if not p.is_absolute():
            p = self.project_root() / p

        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    def job_output_dir(self, job: Mapping[str, Any]) -> Path:
        """The job's figure folder (services/graph_manager/job_paths.py): a job of one run draws into
        that run's paths.model_graphs folder, a job of several runs into <project>/graphs/jobs/."""
        from services.graph_manager import job_paths

        out = job_paths.job_dir(
            job,
            project_root=self.project_root(),
            graphs_root=self.graph_root(),
            env=self.env(),
            model_graphs=(self.cfg.get("paths") or {}).get("model_graphs"),
            name_map=_load_mapping(self.project_root()),
        )
        try:
            out.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            # A job of one run draws into the run's folder, which the training container wrote as root.
            # The panel's buttons take projects/ back first; from a terminal it is this one line.
            raise PermissionError(
                f"graph job id={job.get('id')}: cannot write {out} (the training container wrote the run's folder as "
                f"root). Take it back, then draw again: sudo chown -R \"$(id -u):$(id -g)\" {self.project_root()}/artifacts"
            ) from exc
        return out

    def env(self) -> str:
        return str(os.environ.get("ENV", self.cfg.get("env", "dev")))

    def model_root(self, model: str) -> Path:
        return self.project_root() / "artifacts" / self.env() / "generated_models" / str(model)

    def available_models(self) -> list[str]:
        root = self.project_root() / "artifacts" / self.env() / "generated_models"
        return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []

    def metrics_dir(self, model: str) -> Path:
        """
        Resolve the metrics dir (paths.metrics) of a model.

        GraphManager must not crash if a model has no metrics directory;
        missing metrics are reported by metric_loader.py.
        """
        cfg = {**self.cfg, "model_name": model}

        try:
            return Path(get_metrics_dir(cfg))
        except Exception:
            # Same layout as paths.metrics, so the sibling sources
            # (cross_validation, key_recovery) still resolve.
            return self.model_root(model) / "metrics" / "attack_metrics"

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    def execute(self, graph_id: int) -> None:
        job = get_graph_job_by_id(self.jobs, graph_id)

        if not job.get("enabled", True):
            raise RuntimeError(f"Graph job disabled: id={job['id']} name={job['name']}")

        created_paths: list[Path] = []
        missing_items: list[dict[str, Any]] = []

        # Every artifact the job reads, the registries binding its models included, is verified
        # (utils/seal.py) and recorded for sources.json.
        with recording() as reads:
            job = self.bind_models(job)

            print(
                f"[GraphManager] Running graph job "
                f"id={job['id']} name={job['name']} type={job['type']} "
                f"mode={job.get('mode', 'auto')} plot_type={job.get('plot_type', 'auto')}"
            )

            if job["type"] == "solo":
                created_paths, missing_items = self.run_solo(job)

            elif job["type"] == "process_overlay":
                created_paths, missing_items = self.run_process_overlay(job)

            elif job["type"] == "model_overlay":
                created_paths, missing_items = self.run_model_overlay(job)

            elif job["type"] == "model_explanation":
                created_paths, missing_items = self.run_model_explanation(job)

            else:
                raise RuntimeError(f"Unsupported graph job type={job['type']!r}")

        print(f"[GraphManager] Created graph count: {len(created_paths)}")

        if missing_items:
            print(f"[GraphManager] Missing metric count: {len(missing_items)}")

        # Always shown: it is the menu for the next job, not only an error aid.
        if job["type"] != "model_explanation":
            self.report_choices(job)

        if len(created_paths) == 0:
            raise RuntimeError(
                f"Graph job id={job['id']} created 0 graphs. Check the job's source, "
                "metric, processes and models against the table above."
            )

        # The figures' provenance: the runs (name, hash, key), the files read with their seal
        # state, the figures with their SHA-256; signed (services/graph_manager/sources.py).
        folder = self.job_output_dir(job)
        # export.save_raw_plot_data: the job's manifest beside each figure's raw data (raw_exporter.py)
        from services.graph_manager.raw_exporter import export_job_manifest
        export_job_manifest(job=job, job_output_dir=folder, created_graphs=created_paths, missing_items=missing_items)
        written = write_sources(
            folder,
            job_sources(job=job, bound=self._bound, reads=reads, outputs=created_paths,
                        project_root=self.project_root(), env=self.env(), folder=folder),
        )
        print(f"[GraphManager] Sources written: {written}")

    def execute_facet(self, face_id: int) -> None:
        """Compose the graph files of one faceting entry into a single figure."""
        facet = get_facet_by_id(self.graph_cfg.get("faceting", []), face_id, jobs=self.jobs)

        if not facet["enabled"]:
            raise RuntimeError(f"Facet disabled: id={facet['id']} name={facet['name']}")

        print(
            f"[GraphManager] Running facet id={facet['id']} name={facet['name']} "
            f"arrangement={facet['arrangement']['mode']} order={facet['arrangement']['order']}"
        )
        written = render_facet(
            facet,
            graph_root=self.graph_root(),
            project_root=self.project_root(),
        )
        for path in written:
            print(f"[GraphManager] Facet written: {path}")
        panels = []
        for panel in place_panels(facet):
            stem = _stem(self.graph_root(), self.project_root(), panel.file)
            panels += [stem.with_name(stem.name + suffix) for suffix in sorted(PANEL_SUFFIXES)
                       if stem is not None and stem.with_name(stem.name + suffix).is_file()]
        if written:
            folder = Path(written[0]).parent
            sources = write_sources(folder, facet_sources(
                facet=facet, panels=panels, outputs=written, graph_root=self.graph_root(), folder=folder))
            print(f"[GraphManager] Sources written: {sources}")
        print(f"[GraphManager] Created graph count: {len(written)}")

    def bind_models(self, job: Mapping[str, Any]) -> dict[str, Any]:
        """
        Bind the job's models (a model_name or a model_hash each) to their
        runs of this env, read from artifacts/run_registry.json and
        artifacts/model_name_map.json; a damaged registry raises.

        The returned job lists the model names, so series, legends and output
        paths use names; a legend label written for a hash moves to its name.
        """
        root = self.project_root()
        bound = resolve_model_refs(
            list(job.get("models") or []),
            registry=load_registry(root, heal=False),
            name_map=_load_mapping(root),
            env=self.env(),
            job_id=job.get("id"),
        )

        for run in bound:
            print(
                f"[GraphManager] series model_name={run['model_name']} "
                f"model_hash={run['model_hash']} status={run['status']} (listed as {run['ref']})"
            )
            if run["status"] != "DONE":
                print(
                    f"[GraphManager][WARN] model={run['model_name']} ({run['model_hash']}) "
                    f"has training status {run['status']}, not DONE: its records may be incomplete"
                )

        self._bound = bound      # name, hash and status of every run, for sources.json
        out = dict(job)
        out["models"] = [run["model_name"] for run in bound]

        legend = dict(out.get("legend") or {})
        labels = dict(legend.get("labels") or {})
        for run in bound:
            if run["model_hash"] in labels:
                labels.setdefault(run["model_name"], labels[run["model_hash"]])
        legend["labels"] = labels
        out["legend"] = legend
        return out

    def run_model_explanation(self, job: Mapping[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
        """One curve per model along the trace; a heatmap draws one figure per conv layer."""
        import matplotlib.pyplot as plt

        from services.graph_manager import explain
        from services.graph_manager.plot_style import apply_all_text_styles, apply_axes_style, apply_matplotlib_style
        from services.graph_manager.plotters import legend_label, save_figure

        method, aggregate = job["method"], job["aggregate"]
        registry = load_registry(self.project_root(), heal=False)
        curves: dict[Any, dict[str, np.ndarray]] = {}      # figure key (layer or None) -> model -> curve
        lengths: dict[str, int] = {}
        for run in self._bound:
            model, run_cfg, _ = explain.load_trained(self.cfg, run, registry[run["model_hash"]])
            # Weights need one trace only (to read the flatten's shape); the others use job.traces.
            traces, labels = explain.test_traces(run_cfg, 1 if method == "weight_visualization" else job.get("traces"))
            lengths[run["model_name"]] = int(traces.shape[-1])
            if method == "weight_visualization":
                curves.setdefault(None, {})[run["model_name"]] = explain.weight_visualization(model, traces)
            elif method == "heatmap":
                for layer, curve in explain.heatmaps(model, traces, job.get("layers"), aggregate).items():
                    curves.setdefault(layer, {})[run["model_name"]] = curve
            else:
                curves.setdefault(None, {})[run["model_name"]] = explain.attribution(model, traces, labels, method, aggregate)
            over = "its weights" if method == "weight_visualization" else f"{len(traces)} test traces ({aggregate})"
            print(f"[GraphManager] {method} of {run['model_name']}: {over}")

        titles = {"weight_visualization": "Weight visualisation", "gradient": "Gradient visualisation",
                  "gradient_x_input": "Gradient x input", "heatmap": "Heatmap"}
        apply_matplotlib_style(job)
        created: list[Path] = []
        folder = self.job_output_dir(job)
        for key, by_model in curves.items():
            fig, ax = plt.subplots()
            for name, curve in by_model.items():
                # Pooled curves (weights after the last pooling, a deep conv layer's output) are drawn
                # on the input's sample axis, so the models and layers line up with the trace.
                x = np.arange(len(curve)) * (lengths[name] / len(curve))
                ax.plot(x, curve, label=legend_label(job, name, name))
            ax.set_xlabel("Time sample")
            ax.set_ylabel({"weight_visualization": "Mean |weight|", "heatmap": "Mean filter output"}.get(
                method, "Mean |attribution|" if aggregate == "mean_abs" else "Mean attribution"))
            title = titles[method] + (f", conv layer {key}" if key is not None else "")
            ax.set_title(str((job.get("style") or {}).get("title") or title))
            ax.legend()
            apply_axes_style(ax, job)
            apply_all_text_styles(ax, job)
            stem = job["output_file_name"] + (f"_layer{key}" if key is not None else "")
            created.extend(save_figure(fig, folder / stem, job))
            plt.close(fig)
        return created, []

    def report_choices(self, job: Mapping[str, Any]) -> None:
        """Show which source/process/metric combinations the models recorded."""
        models = [str(m).strip() for m in job.get("models", []) if str(m).strip()]
        known = self.available_models()
        unknown = [m for m in models if not self.model_root(m).is_dir()]

        if unknown:
            print_model_choices(
                missing=unknown,
                available=known,
                env=self.env(),
                model_names_file=self.project_root() / "artifacts" / "done_model_names.yaml",
            )

        inspected = [m for m in models if m not in unknown] or known
        print_metric_choices({m: self.metrics_dir(m).parent for m in inspected})

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_one(
        self,
        *,
        model: str,
        process: str,
        metric: str,
        mode: str,
        job: Mapping[str, Any],
    ) -> LoadedMetric:
        # Whether a metric exists for a process is decided by the recorded
        # data (e.g. the CV final training records GE on train); a missing
        # combination is reported by the loader.
        # mean_accuracy and mean_fit_time_* are read from the column of every
        # run: one run has its own value, source: cross_validation the mean
        # over the folds.
        fold_mean = metric in FOLD_MEAN_METRICS
        stored = FOLD_MEAN_METRICS.get(metric, metric)

        combine = (job.get("combines") or {}).get(process)
        if combine is not None:
            loaded = self._load_combined(
                model=model, metric=stored, mode=mode, job=job, combine=combine
            )
        else:
            process_name = normalize_process_name(process)
            try:
                metrics_dir = self._metrics_dir_for(model, process_name, job)
            except FileNotFoundError as exc:
                route = route_metric(stored, mode=mode)
                return LoadedMetric(stored, route.canonical_metric, model, process_name, route, None, None,
                                    MissingMetric(metric=stored, model=model, process=process_name, reason=str(exc)))
            loaded = load_source_metric(
                source=str(job.get("source", "attack_metrics")),
                metrics_dir=metrics_dir,
                model=model,
                process=process_name,
                metric=stored,
                mode=mode,
                array_epoch=job.get("array_epoch", {"mode": "latest"}),
                with_std=get_std_band(job)["enabled"],
            )
            if job.get("trace_budget") is not None and loaded.ok:
                loaded = self._at_trace_budget(loaded, job, metrics_dir, mode)
        if fold_mean:
            loaded.metric = loaded.canonical_metric = metric
            loaded.route = route_metric(metric, mode=mode)
            if isinstance(loaded.data, pd.DataFrame):
                loaded.data = loaded.data.rename(
                    columns={stored: metric, f"{stored}_std": f"{metric}_std"}
                )
        return self._select_scalar_epoch(loaded, job)

    def _load_combined(
        self,
        *,
        model: str,
        metric: str,
        mode: str,
        job: Mapping[str, Any],
        combine: Mapping[str, Any],
    ) -> LoadedMetric:
        """
        One series from several processes: their values added or averaged per
        epoch. With source: cross_validation this is done inside every fold
        first, so the mean and the spread over folds are those of the combined
        quantity.
        """
        label = combine["label"]
        route = route_metric(metric, mode=mode)
        canonical = route.canonical_metric
        source = str(job.get("source", "attack_metrics"))
        metrics_dir = self.metrics_dir(model)

        def missing(reason: str) -> LoadedMetric:
            print(f"[GraphManager][WARN] combine {label} model={model} metric={metric}: {reason}")
            return LoadedMetric(
                metric, canonical, model, label, route, None, None,
                MissingMetric(metric=metric, model=model, process=label, reason=reason),
            )

        def combined(directory: Path) -> pd.DataFrame | str:
            frames = []
            for process in combine["processes"]:
                part = load_metric(
                    metrics_dir=directory, model=model, process=process,
                    metric=metric, mode=mode, array_epoch={"mode": "latest"},
                )
                if not part.ok or not isinstance(part.data, pd.DataFrame):
                    return f"{canonical!r} is not recorded per epoch for process {process}"
                frame = part.data[["epoch", canonical]].copy()
                frame[canonical] = pd.to_numeric(frame[canonical], errors="coerce")
                frames.append(frame.set_index("epoch")[canonical].rename(process))
            table = pd.concat(frames, axis=1, join="inner")
            if table.empty:
                return "the processes share no epoch"
            values = table.sum(axis=1, skipna=False) if combine["mode"] == "sum" else table.mean(axis=1, skipna=False)
            return values.rename(canonical).reset_index()

        if source == "cross_validation":
            folds = fold_metric_dirs(metrics_dir.parent)
            if not folds:
                return missing("no fold directories (cross_validation disabled for this model?)")
            parts: dict[str, pd.DataFrame] = {}
            for fold in folds:
                result = combined(fold)
                if isinstance(result, str):
                    return missing(f"{fold.name}: {result}")
                parts[fold.name] = result
            data = combine_folds(canonical, parts, with_std=get_std_band(job)["enabled"])
        else:
            data = combined(metrics_dir)
            if isinstance(data, str):
                return missing(data)

        return LoadedMetric(metric, canonical, model, label, route, data, metrics_dir.parent)

    def _metrics_dir_for(self, model: str, process: str, job: Mapping[str, Any]) -> Path:
        """The run's metrics dir; for the test process, the test the job's `test` chooses (trace_budget.py): the
        latest All test by default, never an earlier one."""
        if process == "test":
            return test_metrics_dir(self.model_root(model), self._test_entry(model), job.get("test"))
        return self.metrics_dir(model)

    def _test_entry(self, model: str) -> dict:
        """The run's test_registry.json entry (its tests), found by its name through model_name_map.json."""
        from services.side_channel_core.utils.seal import load_json

        root = self.project_root() / "artifacts"
        try:
            names = _load_mapping(self.project_root())
            registry = load_json(root / "test_registry.json") if (root / "test_registry.json").is_file() else {}
        except Exception:                                  # unreadable: the recorded test is read
            return {}
        hashes = [h for h, n in (names or {}).items() if n == model]
        entry = registry.get(hashes[0]) if hashes else registry.get(model)
        return entry if isinstance(entry, dict) else {}

    @staticmethod
    def _at_trace_budget(loaded: LoadedMetric, job: Mapping[str, Any], metrics_dir: Path, mode: str) -> LoadedMetric:
        """The loaded attack curve cut at the job's trace_budget; final_ranks read at it (trace_budget.py)."""
        budget = int(job["trace_budget"])
        where = f"graph job id={job.get('id')} {loaded.model} {loaded.process} {loaded.canonical_metric}"
        if loaded.canonical_metric == "final_ranks":
            curves = load_source_metric(
                source="attack_metrics", metrics_dir=metrics_dir, model=loaded.model, process=loaded.process,
                metric="per_attack_curves", mode=mode, array_epoch=job.get("array_epoch", {"mode": "latest"}),
            )
            if not curves.ok:
                raise TraceBudgetError(f"{where}: final_ranks at a budget needs the run's per_attack_curves")
            loaded.data = final_ranks_at(curves.data, budget, where)
        else:
            loaded.data = cut_curve(loaded.data, budget, where)
        return loaded

    @staticmethod
    def _select_scalar_epoch(loaded: LoadedMetric, job: Mapping[str, Any]) -> LoadedMetric:
        """
        array_epoch.mode: selected keeps one epoch of a per-epoch scalar, so a
        model overlay of it becomes one bar per model (e.g. fit time at epoch 100).
        With plot_type: barh, which draws one bar per model, mode: latest keeps
        the last epoch of each model (its final value, e.g. the fit time of
        the whole training, whatever its number of epochs).
        """
        array_epoch = job.get("array_epoch") or {}
        mode = str(array_epoch.get("mode", "latest"))
        if not loaded.ok or not isinstance(loaded.data, pd.DataFrame):
            return loaded
        if "epoch" not in loaded.data.columns:
            return loaded
        if mode == "latest" and get_plot_type(job) == "barh" and not loaded.data.empty:
            last = loaded.data["epoch"].max()
            loaded.data = loaded.data[loaded.data["epoch"] == last].reset_index(drop=True)
            return loaded
        if mode != "selected":
            return loaded
        digits = "".join(ch for ch in str(array_epoch.get("epoch", "")) if ch.isdigit())
        if not digits:
            return loaded
        loaded.data = loaded.data[loaded.data["epoch"] == int(digits)].reset_index(drop=True)
        if loaded.data.empty:
            print(
                f"[GraphManager][WARN] model={loaded.model} process={loaded.process} "
                f"has no {loaded.canonical_metric!r} at epoch {int(digits)}"
            )
        return loaded

    # ------------------------------------------------------------------
    # Plot dispatch
    # ------------------------------------------------------------------

    def _plot_loaded_metric(
        self,
        *,
        loaded: LoadedMetric,
        job: Mapping[str, Any],
        out_base: Path,
    ) -> list[Path]:
        if not loaded.ok:
            return []

        route = loaded.route
        metric = loaded.canonical_metric

        if route.plotter == "fold_curves":
            return plot_fold_rank_curves(
                folds=loaded.data,
                job=job,
                out_base=out_base,
                model=loaded.model,
                process=loaded.process,
            )

        if route.plotter == "key_scores":
            # CV validation: one record per fold -> one figure per fold.
            payloads = loaded.data if "score" not in loaded.data else {"": loaded.data}
            paths: list[Path] = []
            for fold, payload in payloads.items():
                paths += plot_key_candidate_scores(
                    payload=payload,
                    job=job,
                    out_base=Path(f"{out_base}__{fold}") if fold else out_base,
                    model=loaded.model,
                    process=f"{loaded.process} {fold}".strip(),
                )
            return paths

        if route.plotter == "scalar_line":
            if not isinstance(loaded.data, pd.DataFrame):
                print(
                    f"[GraphManager][WARN] Expected DataFrame for scalar metric={metric}, "
                    f"got {type(loaded.data)}"
                )
                return []

            return plot_scalar_line(
                df=loaded.data,
                job=job,
                out_base=out_base,
                metric=metric,
                model=loaded.model,
                process=loaded.process,
                x_col=route.x_axis,
                y_col=route.y_axis,
            )
        # Epoch-overlay mode for one model:
        #
        # metric_loader returns:
        #   {"ep0001": curve, "ep0002": curve, ...}
        #
        # This must be plotted as a curve overlay, not as one single curve.
        if route.plotter == "attack_curve" and isinstance(loaded.data, dict):
            return plot_curve_overlay(
                series=loaded.data,
                job=job,
                out_base=out_base,
                metric=metric,
                x_label=route.x_axis,
                y_label=route.y_axis,
            )
        if route.plotter == "attack_curve" and route.family in {"rank", "repeated_attack"}:
            return plot_attack_metric(
                metric=metric,
                data=loaded.data,
                job=job,
                out_base=out_base,
                model=loaded.model,
                process=loaded.process,
            )

        if route.plotter == "distribution" and route.family in {"repeated_attack", "research"}:
            return plot_attack_metric(
                metric=metric,
                data=loaded.data,
                job=job,
                out_base=out_base,
                model=loaded.model,
                process=loaded.process,
            )

        if route.plotter == "attack_curve":
            return plot_curve(
                data=loaded.data,
                job=job,
                out_base=out_base,
                metric=metric,
                model=loaded.model,
                process=loaded.process,
                x_label=route.x_axis,
                y_label=route.y_axis,
            )

        if route.plotter == "distribution":
            return plot_distribution(
                data=loaded.data,
                job=job,
                out_base=out_base,
                metric=metric,
                model=loaded.model,
                process=loaded.process,
                x_label=route.x_axis,
                y_label=route.y_axis,
            )

        print(f"[GraphManager][WARN] No plotter route for metric={metric}: {route}")
        return []

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    def run_solo(self, job: Mapping[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
        created: list[Path] = []
        missing: list[dict[str, Any]] = []

        metric = str(job["metric"])
        mode = str(job.get("mode", "auto"))

        for model in job.get("models", []):
            model = str(model).strip()
            if not model:
                continue

            for process_raw in job.get("processes", []):
                process = normalize_process_name(process_raw)

                loaded = self._load_one(
                    model=model,
                    process=process,
                    metric=metric,
                    mode=mode,
                    job=job,
                )

                if loaded.missing:
                    missing.append(missing_to_dict(loaded.missing))
                    continue

                context = {
                    "metric": loaded.canonical_metric,
                    "model": model,
                    "model_name": model,
                    "process": process,
                    "job_id": job.get("id"),
                    "job_name": job.get("name"),
                    "source": job.get("source", "attack_metrics"),
                }
                job_ctx = with_context(job, context)

                out_dir = (
                    self.job_output_dir(job_ctx)
                    / "solo"
                    / safe_name(model)
                    / safe_name(process)
                )
                out_base = (
                    out_dir
                    / f"{output_file_name(job_ctx, context)}__"
                    f"{safe_name(model)}__{safe_name(process)}__"
                    f"{safe_name(loaded.canonical_metric)}"
                )

                paths = self._plot_loaded_metric(
                    loaded=loaded,
                    job=job_ctx,
                    out_base=out_base,
                )
                created.extend(paths)

        return created, missing

    def run_process_overlay(
        self,
        job: Mapping[str, Any],
    ) -> tuple[list[Path], list[dict[str, Any]]]:
        """
        Overlay processes for each model.

        New schema:
            type: process_overlay
            models: [...]
            processes: [...]

        Behavior:
            for each model:
                overlay all selected processes in one figure
        """

        created: list[Path] = []
        missing: list[dict[str, Any]] = []

        metric = str(job["metric"])
        mode = str(job.get("mode", "auto"))
        route = route_metric(metric, mode=mode)

        processes = [normalize_process_name(p) for p in job.get("processes", [])]

        for model in job.get("models", []):
            model = str(model).strip()
            if not model:
                continue

            scalar_series: dict[str, pd.DataFrame] = {}
            curve_series: dict[str, Any] = {}
            dropped: dict[str, str] = {}             # a process that recorded nothing keeps its slot

            valid_processes: list[str] = []

            for process in processes:
                loaded = self._load_one(
                    model=model,
                    process=process,
                    metric=metric,
                    mode=mode,
                    job=job,
                )

                if loaded.missing:
                    missing.append(missing_to_dict(loaded.missing))
                    dropped[process] = "not recorded"
                    continue

                label_key = process
                valid_processes.append(process)

                if loaded.route.storage == "scalar_history":
                    scalar_series[label_key] = loaded.data
                else:
                    curve_series[label_key] = loaded.data

            if not scalar_series and not curve_series:
                print(
                    f"[GraphManager][WARN] No valid process_overlay data for "
                    f"model={model} processes={processes}"
                )
                continue

            process_key = combo_key(valid_processes or processes)

            context = {
                "metric": route.canonical_metric,
                "model": model,
                "model_name": model,
                "process": process_key,
                "processes": process_key,
                "job_id": job.get("id"),
                "job_name": job.get("name"),
                "source": job.get("source", "attack_metrics"),
            }
            job_ctx = with_context(job, context)

            out_dir = self.job_output_dir(job_ctx) / "process_overlay" / safe_name(model)
            out_base = (
                out_dir
                / f"{output_file_name(job_ctx, context)}__"
                f"{safe_name(model)}__{safe_name(process_key)}__"
                f"{safe_name(route.canonical_metric)}"
            )

            if scalar_series:
                created.extend(
                    plot_scalar_overlay(
                        series=scalar_series,
                        job=job_ctx,
                        out_base=out_base,
                        metric=route.canonical_metric,
                        x_col=route.x_axis,
                        y_col=route.y_axis,
                        order=processes,
                        dropped=dropped,
                    )
                )

            if curve_series:
                created.extend(
                    plot_curve_overlay(
                        series=curve_series,
                        job=job_ctx,
                        out_base=out_base,
                        metric=route.canonical_metric,
                        x_label=route.x_axis,
                        y_label=route.y_axis,
                        order=processes,
                        dropped=dropped,
                        marks=job.get("marks"),
                    )
                )

        return created, missing

    def relative_reference(self, job: Mapping[str, Any], models: list[str]) -> dict[str, str]:
        """
        The reference run of the job's relative: block, found by its `where`
        conditions (model_filter): rule run among the job's models, rule anchor
        among the finished runs of this ENV. Exactly one run must match.
        """
        from services.model_filter.model_filter import ModelFilterError, find_runs

        relative = job["relative"]
        rule, label = relative["rule"], relative["label"]
        if rule == "probe":
            print(f"[GraphManager] relative (rule probe): every run divided by its own device probe, {label}")
            return {"rule": rule, "label": label}
        try:
            runs = find_runs(
                self.project_root(), relative["where"], env=self.env(),
                status="DONE" if rule == "anchor" else None,
                model_names=models if rule == "run" else None,
            )
        except ModelFilterError as exc:
            raise ValueError(f"graph job id={job.get('id')}: relative.where: {exc}") from exc
        if len(runs) != 1:
            among = "the job's models" if rule == "run" else f"the finished runs of ENV={self.env()}"
            found = ", ".join(f"{r.model_name} ({r.model_hash})" for r in runs) or "none"
            raise ValueError(
                f"graph job id={job.get('id')}: relative.where ({label}) must match exactly one run among "
                f"{among}; it matches {len(runs)}: {found}"
            )
        run = runs[0]
        print(
            f"[GraphManager] relative (rule {rule}): reference = {run.model_name} ({run.model_hash}), "
            f"{label}; its value is 1.0"
        )
        return {"rule": rule, "model_name": run.model_name, "model_hash": run.model_hash, "label": label}

    def _device_record(self, model: str) -> dict[str, Any] | None:
        from utils.device_record import read_device_record

        return read_device_record(self.model_root(model))

    def _run_probes(self, models: Iterable[str], job: Mapping[str, Any]) -> dict[str, float]:
        """Seconds per probe step of every run; a run without a probe stops the job."""
        from utils.device_record import PROBE_SPEC, probe_seconds

        probes, lacking = {}, []
        for model in models:
            value = probe_seconds(self._device_record(model))
            if value is None:
                lacking.append(model)
            else:
                probes[model] = value
        if lacking:
            raise ValueError(
                f"graph job id={job.get('id')}: relative.rule probe needs a device probe (version "
                f"{PROBE_SPEC['version']}) in <run>/device.yaml (paths.device); none for {lacking} (trained before device "
                "records, or the probe failed)"
            )
        return probes

    def _check_same_device(self, reference: Mapping[str, str], models: Iterable[str], job: Mapping[str, Any]) -> None:
        """
        A time is divided by the reference's only when both trained on one
        device (GPU, torch, CUDA, cuDNN, TF32 switches). A run without a device
        record is named in a warning; the ratio then assumes one device.
        """
        from utils.device_record import SAME_DEVICE_FIELDS, device_fingerprint

        ref_print = device_fingerprint(self._device_record(reference["model_name"]))
        unknown, differ = [], []
        for model in models:
            fingerprint = device_fingerprint(self._device_record(model))
            if fingerprint is None or ref_print is None:
                unknown.append(model)
            elif fingerprint != ref_print:
                fields = [".".join(path) for path, a, b in zip(SAME_DEVICE_FIELDS, fingerprint, ref_print) if a != b]
                differ.append(f"{model} ({', '.join(fields)})")
        if differ:
            raise ValueError(
                f"graph job id={job.get('id')}: relative compares training times only on one device; "
                f"these runs differ from the reference {reference['model_name']}: {'; '.join(differ)}. "
                "Use relative.rule probe to compare across devices"
            )
        if unknown:
            print(
                f"[GraphManager][WARN] no device record for {sorted(set(unknown + ([reference['model_name']] if ref_print is None else [])))}: "
                "the relative times assume they trained on one device"
            )

    def _relative_reference_frames(
        self,
        reference: Mapping[str, str],
        job: Mapping[str, Any],
        process: str,
        metrics: list[str],
        mode: str,
        several: bool,
    ) -> dict[str, pd.DataFrame]:
        """The reference run's series of every metric, read as the job reads its models (metric_process too)."""
        frames: dict[str, pd.DataFrame] = {}
        for one in metrics:
            read = (job.get("metric_process") or {}).get(one, process)
            loaded = self._load_one(model=reference["model_name"], process=read, metric=one, mode=mode, job=job)
            if loaded.missing:
                raise ValueError(
                    f"graph job id={job.get('id')}: relative reference {reference['model_name']} "
                    f"({reference['label']}) has no {one} for process {read}"
                )
            if several:
                column = loaded.route.y_axis
                frames[one] = loaded.data.rename(columns={column: "value", f"{column}_std": "value_std"})
            else:
                frames[""] = loaded.data
        return frames

    def run_fold_heatmap(self, job: Mapping[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
        """
        plot_type heatmap (graph_job_validator.check_heatmap_job; owner 2026-10-27, Fig. 20 c): a model per row, a
        cross-validation fold per column, each cell the job's metric in that fold's key-recovery report
        (cross_validation/fold_NN/key_recovery/validation_history.jsonl) at the job's epoch (array_epoch latest:
        the final one; selected: its epoch). A fold whose value is empty (stable_rank1_trace: the key never stays at
        rank 1) shows its final key rank instead, on grey.
        """
        import re

        from services.graph_manager.metric_loader import _load_key_recovery_dir
        from services.graph_manager.missing_plotter import report_missing_metric

        metric, process = str(job["metric"]), "validation"
        policy = job.get("array_epoch") or {}
        selected = policy.get("epoch") if policy.get("mode") == "selected" else None
        selected = int(re.sub(r"\D", "", str(selected))) if selected not in (None, "") else None
        labels = (job.get("legend") or {}).get("labels") or {}
        models = [str(m).strip() for m in job.get("models", []) if str(m).strip()]
        rows: list[str] = []
        values: list[list[float]] = []
        notes: list[list[str]] = []
        missing: list[dict[str, Any]] = []
        folds_of: list[int] = []
        for model in models:
            folds = sorted((self.metrics_dir(model).parent / "cross_validation").glob("fold_*"))
            line, line_notes, checked = [], [], []
            for fold in folds:
                kr = fold / "key_recovery"
                data, paths = _load_key_recovery_dir(kr, process, metric)
                ranks, _ = _load_key_recovery_dir(kr, process, "final_true_key_rank")
                checked += paths
                if data is None or data.empty:
                    line = None
                    break
                at = data[data["epoch"] == selected] if selected is not None else data.tail(1)
                if at.empty:
                    line = None
                    break
                value = pd.to_numeric(at[metric], errors="coerce").iloc[0]
                rank = None
                if ranks is not None:
                    hit = ranks[ranks["epoch"] == at["epoch"].iloc[0]]
                    rank = None if hit.empty else hit["final_true_key_rank"].iloc[0]
                line.append(float(value) if pd.notna(value) else float("nan"))
                line_notes.append(f"rank\n{int(rank)}" if rank is not None and pd.notna(rank) else "none")
            if not folds or line is None:
                missing.append(missing_to_dict(report_missing_metric(
                    metric=metric, model=model, process=process,
                    reason="no cross-validation fold records it" if not folds else
                    f"a fold's key-recovery report has no {metric}" + (f" at epoch {selected}" if selected else ""),
                    expected_paths=[str(p) for p in checked])))
                continue
            rows.append(model)
            values.append(line)
            notes.append(line_notes)
            folds_of.append(len(line))
        if not rows:
            return [], missing
        if len(set(folds_of)) != 1:
            raise ValueError(f"graph job id={job.get('id')}: heatmap rows need the same folds, got {dict(zip(rows, folds_of))}")
        context = {"metric": metric, "process": process, "models": combo_key(rows), "model": combo_key(rows),
                   "model_name": combo_key(rows), "job_id": job.get("id"), "job_name": job.get("name"),
                   "source": job.get("source")}
        job_ctx = with_context(job, context)
        out_dir = self.job_output_dir(job_ctx) / "model_overlay" / process
        out_base = out_dir / f"{output_file_name(job_ctx, context)}__{process}__{safe_name(metric)}"
        created = plot_fold_heatmap(
            values=values, notes=notes, rows=[str(labels.get(m, m)) for m in rows],
            colours=[((job.get("style") or {}).get("series") or {}).get(m, {}).get("color") or f"C{models.index(m) % 10}"
                     for m in rows],                         # style.series colours, else the run's slot's
            folds=list(range(1, folds_of[0] + 1)),
            job=job_ctx, out_base=out_base, metric=metric)
        return created, missing

    def run_model_overlay(
        self,
        job: Mapping[str, Any],
    ) -> tuple[list[Path], list[dict[str, Any]]]:
        """
        Overlay models for each process.

        New schema:
            type: model_overlay
            models: [...]
            processes: [...]

        Behavior:
            for each process:
                overlay all selected models in one figure

        This is the correct mode for ablation graphs:
            each model is one ablation setting.
        """

        if get_plot_type(job) == "heatmap":
            return self.run_fold_heatmap(job)

        created: list[Path] = []
        missing: list[dict[str, Any]] = []

        metric = str(job["metric"])
        metrics = [str(m) for m in (job.get("metrics") or [metric])]
        mode = str(job.get("mode", "auto"))
        route = route_metric(metric, mode=mode)
        several = len(metrics) > 1
        if several and any(route_metric(m, mode=mode).storage != "scalar_history" for m in metrics):
            raise ValueError(
                f"graph job id={job.get('id')}: several metrics can be overlaid only when all are "
                f"per-epoch scalars, got {metrics}"
            )

        models = [str(m).strip() for m in job.get("models", []) if str(m).strip()]
        processes = [normalize_process_name(p) for p in job.get("processes", [])]
        reference = self.relative_reference(job, models) if job.get("relative") else None
        strip = get_plot_type(job) == "strip"           # a dot per run, a row per group (check_strip_job)

        # metric_process (graph_job_validator.normalize_metric_process): a metric read from its own process
        metric_process = job.get("metric_process") or {}
        for process in processes:
            read = {one: metric_process.get(one, process) for one in metrics}
            # the figure's process: the job's, or every process its metrics are read from ("validation+train")
            shown = "+".join(dict.fromkeys(read.values())) if metric_process else process
            scalar_series: dict[str, pd.DataFrame] = {}
            curve_series: dict[str, Any] = {}
            # Every model keeps its slot (colour i of the job's order) in the overlay:
            # one that recorded nothing is drawn as a legend entry with the reason.
            order = [f"{m}:{one}" for m in models for one in metrics] if several else list(models)
            dropped: dict[str, str] = {}

            valid_models: list[str] = []

            for model in models:
                for one in metrics:
                    loaded = self._load_one(
                        model=model,
                        process=read[one],
                        metric=one,
                        mode=mode,
                        job=job,
                    )

                    if loaded.missing:
                        missing.append(missing_to_dict(loaded.missing))
                        dropped[f"{model}:{one}" if several else model] = "not recorded"
                        continue

                    if model not in valid_models:
                        valid_models.append(model)

                    if not several:
                        if loaded.route.storage == "scalar_history":
                            scalar_series[model] = loaded.data
                        else:
                            curve_series[model] = loaded.data
                        continue
                    # One curve per (model, metric): the metric's column (and its
                    # std over folds) under one name, so all share the y axis.
                    column = loaded.route.y_axis
                    scalar_series[f"{model}:{one}"] = loaded.data.rename(
                        columns={column: "value", f"{column}_std": "value_std"}
                    )

            if not scalar_series and not curve_series:
                print(
                    f"[GraphManager][WARN] No valid model_overlay data for "
                    f"models={models} process={process}"
                )
                continue

            model_key = combo_key(valid_models or models)

            context = {
                "metric": route.canonical_metric,
                "process": shown,
                "models": model_key,
                "model": model_key,
                "model_name": model_key,
                "job_id": job.get("id"),
                "job_name": job.get("name"),
                "source": job.get("source", "attack_metrics"),
                # text_rules can name the reference: "fit time relative to {relative_to}"
                "relative_to": reference["label"] if reference else "",
            }
            job_ctx = with_context(job, context)

            out_dir = self.job_output_dir(job_ctx) / "model_overlay" / safe_name(shown)
            if reference is not None and reference["rule"] == "probe":
                if curve_series:
                    raise ValueError(
                        f"graph job id={job.get('id')}: relative divides scalar metrics; "
                        f"{route.canonical_metric} is a curve"
                    )
                metric_of = {key: (key.partition(":")[2] if several else metric) for key in scalar_series}
                scalar_series = probe_relative_series(
                    scalar_series,
                    self._run_probes({key.partition(":")[0] for key in scalar_series}, job),
                    "value" if several else route.y_axis,
                    metric_of,
                    job_id=job.get("id"),
                )
            elif reference is not None:
                if curve_series:
                    raise ValueError(
                        f"graph job id={job.get('id')}: relative divides scalar metrics; "
                        f"{route.canonical_metric} is a curve"
                    )
                if any(time_metric_seconds(m) is not None for m in metrics):
                    self._check_same_device(reference, {key.partition(":")[0] for key in scalar_series}, job)
                scalar_series = relative_series(
                    scalar_series,
                    self._relative_reference_frames(reference, job, process, metrics, mode, several),
                    "value" if several else route.y_axis,
                    label=reference["label"],
                    job_id=job.get("id"),
                )

            metric_part = "+".join(metrics) if several else route.canonical_metric
            out_base = (
                out_dir
                / f"{output_file_name(job_ctx, context)}__"
                f"{safe_name(shown)}__{safe_name(metric_part)}"
            )

            if scalar_series and strip:
                created.extend(
                    plot_strip(
                        series=scalar_series,
                        job=job_ctx,
                        out_base=out_base,
                        metric=route.canonical_metric,
                        groups=job.get("groups"),
                        order=order,
                        dropped=dropped,
                    )
                )
            elif scalar_series:
                created.extend(
                    plot_scalar_overlay(
                        series=scalar_series,
                        job=job_ctx,
                        out_base=out_base,
                        metric=metric_part,
                        x_col=route.x_axis,
                        y_col="value" if several else route.y_axis,
                        order=order,
                        dropped=dropped,
                        groups=job.get("groups"),
                    )
                )

            if curve_series:
                created.extend(
                    plot_curve_overlay(
                        series=curve_series,
                        job=job_ctx,
                        out_base=out_base,
                        metric=route.canonical_metric,
                        x_label=route.x_axis,
                        y_label=route.y_axis,
                        order=order,
                        dropped=dropped,
                        groups=job.get("groups"),
                        overlays=self._overlay_data(list(curve_series), process, job, mode) if job.get("overlays") else None,
                    )
                )

        return created, missing

    def _overlay_data(self, models: list[str], process: str, job: Mapping[str, Any], mode: str) -> dict[str, Any]:
        """
        The values of the job's overlays (graph_job_validator.normalize_overlays_config) for the runs whose GE curve
        is drawn, read from the curve's own record (_load_one: the job's process, `test`, `trace_budget` and epoch):
        ge_averages from per_attack_curves, ntge from the last row's per_attack_traces_to_ge_1, trace_lines from
        the run's snapshot. A run without what an overlay reads is left out of it with a warning; plotters._draw_overlays
        draws the result.
        """
        overlays = job["overlays"]
        out: dict[str, Any] = {}
        warn = lambda what, model, reason: print(                                   # noqa: E731
            f"[GraphManager][WARN] overlays.{what}: model={model} process={process} {reason}: left out of it")
        if overlays.get("ge_averages"):
            averages: dict[str, dict[str, np.ndarray]] = {name: {} for name in overlays["ge_averages"]}
            for model in models:
                loaded = self._load_one(model=model, process=process, metric="per_attack_curves", mode=mode, job=job)
                if not loaded.ok:
                    warn("ge_averages", model, "has no per_attack_curves")
                    continue
                try:
                    curves = ge_average_curves(loaded.data)
                except ValueError as exc:                                # not [attacks, traces, 2]
                    warn("ge_averages", model, str(exc))
                    continue
                for name, curve in curves.items():
                    if name in averages:
                        averages[name][model] = curve
            out["ge_averages"] = averages
        if overlays.get("ntge"):
            values: dict[str, float] = {}
            for model in models:
                loaded = self._load_one(model=model, process=process, metric="per_attack_traces_to_ge_1",
                                        mode=mode, job=job)
                if not loaded.ok or not isinstance(loaded.data, pd.DataFrame) or loaded.data.empty:
                    warn("ntge", model, "records no per_attack_traces_to_ge_1")
                    continue
                # NaN: no attack ever stably at rank 1 (the recorded null)
                values[model] = float(pd.to_numeric(loaded.data["per_attack_traces_to_ge_1"], errors="coerce").iloc[-1])
            out["ntge"] = values
        if overlays.get("trace_lines"):
            lines: dict[str, list[float]] = {name: [] for name in overlays["trace_lines"]}
            for model in models:
                n_val, budget = self._trace_counts(model)
                if "n_val" in lines:
                    if n_val is None:
                        warn("trace_lines", model, "has neither data.validation_range nor a validation GE curve")
                    else:
                        lines["n_val"].append(float(n_val))
                if "budget" in lines and budget is not None:
                    lines["budget"].append(float(budget))
            out["trace_lines"] = lines
        return out

    def _trace_counts(self, model: str) -> tuple[int | None, int | None]:
        """(N_val, N_budget) of a run: the length of its snapshot's data.validation_range (else the last trace count
        of its validation mean_attack_rank_curve) and its data entry's attack_trace_budget (None: none)."""
        from services.side_channel_core.utils.seal import load_yaml

        snapshot = self.model_root(model) / "config" / f"{model}_params.yaml"
        data: Any = {}
        if snapshot.is_file():
            data = (load_yaml(snapshot) or {}).get("data") or {}
            data = data[0] if isinstance(data, list) and len(data) == 1 else data
        data = data if isinstance(data, Mapping) else {}
        window = data.get("validation_range")
        n_val = int(window[1]) - int(window[0]) if isinstance(window, (list, tuple)) and len(window) == 2 else None
        if n_val is None:
            curve = load_source_metric(source="attack_metrics", metrics_dir=self.metrics_dir(model), model=model,
                                       process="validation", metric="mean_attack_rank_curve",
                                       array_epoch={"mode": "latest"})
            if curve.ok and isinstance(curve.data, np.ndarray) and curve.data.ndim == 2 and len(curve.data):
                n_val = int(curve.data[-1, 0])
        budget = data.get("attack_trace_budget")
        return n_val, (int(budget) if budget not in (None, "") else None)

    @staticmethod
    def dedupe_combinations(combos: Iterable[Iterable[str]]) -> list[list[str]]:
        """
        Kept for backward compatibility and potential future use.
        Current schema does not require model_combinations/process_combinations.
        """

        seen = set()
        unique: list[list[str]] = []

        for combo in combos:
            combo_list = [str(x) for x in combo]
            key = tuple(sorted(combo_list))

            if key in seen:
                continue

            seen.add(key)
            unique.append(combo_list)

        return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph jobs from graph_config.yaml")
    parser.add_argument("--project", required=True, help="Project name from master.yaml")
    parser.add_argument("--env", required=True, help="Runtime environment: dev/staging/prod")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--graph-id", type=int, help="Graph job ID from graph_config.yaml (jobs)")
    target.add_argument("--face-id", type=int, help="Facet ID from graph_config.yaml (faceting)")
    args = parser.parse_args()

    os.environ["ENV"] = args.env

    manager = GraphManager(project_name=args.project)
    if args.face_id is not None:
        manager.execute_facet(face_id=args.face_id)
    else:
        manager.execute(graph_id=args.graph_id)


if __name__ == "__main__":
    main()
