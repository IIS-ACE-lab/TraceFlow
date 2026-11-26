# services/mlops/core/benchmark_runner.py
# ============================================================
# Benchmark Runner (PURE LOGIC)
# - Builds a deterministic benchmarking plan from compiled context
# - Defines expected artifacts contract (paths + filenames) without writing files
# - NO execution: execution delegated later in Phase 3+
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence
from services.side_channel_core.utils.timestamps import now_iso


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class BenchmarkTestPlan:
    """
    A single benchmark test (deterministic plan).
    """
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkPlan:
    """
    Full benchmark plan for a trial/model candidate.
    """
    ts: str
    project_name: str
    hypothesis_id: str
    trial_id: Optional[int]
    sota_trial: Optional[str]

    enabled: bool
    tests: List[BenchmarkTestPlan]

    # thresholds that define "qualified"
    qualify_thresholds: Dict[str, Any] = field(default_factory=dict)

    # expected artifact destinations (no writes), relative templates or resolved paths
    artifacts: Dict[str, str] = field(default_factory=dict)

    # machine-readable metadata for downstream executor
    meta: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Public API
# ============================================================

def build_benchmark_plan(
    *,
    compiled: Mapping[str, Any],
    hypothesis_id: str,
    trial_id: Optional[int] = None,
    sota_trial: Optional[str] = None,
    resolved_paths: Optional[Mapping[str, str]] = None,
) -> BenchmarkPlan:
    """
    Build deterministic benchmarking plan and artifacts contract.

    Inputs:
      compiled: output of core/context.py (merged view)
      hypothesis_id: selected hypothesis
      trial_id/sota_trial: identifiers (optional)
      resolved_paths: optional mapping of resolved path strings (from AutoConfig.get_map(..., resolve_as_path=True))
                     If not provided, contract will be built using templates where possible.

    Expected compiled keys (best-effort):
      - compiled["benchmarking"] may include:
          enabled: bool
          tests: list
          qualify_thresholds: dict
      - compiled["paths"] should include benchmark roots:
          auto_run_map_benchmark, leaderboard_benchmark, pareto_front_benchmark
      - compiled["registry"] may include filenames:
          auto_run_map.filename, leaderboard.filename, pareto_front.filename
    """
    ts = _now_iso()
    proj = _project_name(compiled)

    bench_block = compiled.get("benchmarking", {}) or {}
    enabled = bool(bench_block.get("enabled", True))

    tests = _parse_tests(bench_block.get("tests", []))
    qualify_thresholds = _normalize_dict(bench_block.get("qualify_thresholds", {}))

    artifacts = build_expected_benchmark_artifacts(
        compiled=compiled,
        hypothesis_id=hypothesis_id,
        trial_id=trial_id,
        sota_trial=sota_trial,
        resolved_paths=resolved_paths,
    )

    meta = {
        "project": proj,
        "hypothesis_id": hypothesis_id,
        "trial_id": trial_id,
        "sota_trial": sota_trial,
        "registry_filenames": _registry_filenames(compiled),
        "bench_block": _safe_shallow_copy(bench_block),
    }

    return BenchmarkPlan(
        ts=ts,
        project_name=proj,
        hypothesis_id=hypothesis_id,
        trial_id=trial_id,
        sota_trial=sota_trial,
        enabled=enabled,
        tests=tests,
        qualify_thresholds=qualify_thresholds,
        artifacts=artifacts,
        meta=meta,
    )


def build_expected_benchmark_artifacts(
    *,
    compiled: Mapping[str, Any],
    hypothesis_id: str,
    trial_id: Optional[int] = None,
    sota_trial: Optional[str] = None,
    resolved_paths: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """
    Expected artifacts contract for benchmarking.

    Returns mapping of artifact keys -> path strings (no IO).
    Keys are stable, executor can rely on them.

    Artifacts include:
      - benchmark_root (dir)
      - run_report_json
      - robustness_report_json
      - calibration_report_json (optional placeholder)
      - leaderboard_csv (shared)
      - pareto_front_json (shared)
      - auto_run_map_json (shared)

    If resolved_paths is provided, uses it as authoritative for benchmark roots.
    Otherwise uses compiled["paths"] templates as-is.
    """
    paths_block = compiled.get("paths", {}) or {}
    reg_files = _registry_filenames(compiled)

    # benchmark roots (prefer resolved)
    auto_run_map_root = _get_path_value(resolved_paths, paths_block, "auto_run_map_benchmark")
    leaderboard_root = _get_path_value(resolved_paths, paths_block, "leaderboard_benchmark")
    pareto_root = _get_path_value(resolved_paths, paths_block, "pareto_front_benchmark")

    # shared files
    auto_run_map_file = _join(auto_run_map_root, reg_files["auto_run_map"])
    leaderboard_file = _join(leaderboard_root, reg_files["leaderboard"])
    pareto_file = _join(pareto_root, reg_files["pareto_front"])

    # per-trial benchmark outputs (deterministic naming)
    trial_tag = _trial_tag(hypothesis_id=hypothesis_id, trial_id=trial_id, sota_trial=sota_trial)

    # per-trial outputs under the benchmark root (use auto_run_map_root as canonical)
    trial_dir = _join(auto_run_map_root, "trials", trial_tag)

    # standard outputs
    run_report = _join(trial_dir, "benchmark_report.json")
    robustness_report = _join(trial_dir, "robustness_report.json")
    calibration_report = _join(trial_dir, "calibration_report.json")
    metrics_table = _join(trial_dir, "metrics_table.csv")

    # optional plot exports (executor may write)
    plots_dir = _join(trial_dir, "plots")
    pareto_plot = _join(plots_dir, "pareto_front.png")
    robustness_plot = _join(plots_dir, "robustness_matrix.png")

    return {
        # roots
        "benchmark_root": str(auto_run_map_root),
        "trial_benchmark_dir": str(trial_dir),
        "plots_dir": str(plots_dir),

        # per-trial files
        "benchmark_report_json": str(run_report),
        "robustness_report_json": str(robustness_report),
        "calibration_report_json": str(calibration_report),
        "metrics_table_csv": str(metrics_table),

        # shared registries
        "auto_run_map_json": str(auto_run_map_file),
        "leaderboard_csv": str(leaderboard_file),
        "pareto_front_json": str(pareto_file),

        # optional plot destinations (contract only)
        "pareto_front_png": str(pareto_plot),
        "robustness_matrix_png": str(robustness_plot),
    }


# ============================================================
# Internals
# ============================================================

def _parse_tests(raw: Any) -> List[BenchmarkTestPlan]:
    """
    Accepts:
      tests:
        - name: standard_attack
          enabled: true
        - name: desync_robustness
          enabled: true
          desync_values: [0,25,50]
    """
    out: List[BenchmarkTestPlan] = []
    if not raw:
        return out
    if not isinstance(raw, list):
        raise TypeError("[mlops/core/benchmark_runner] benchmarking.tests must be a list")
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TypeError(f"[mlops/core/benchmark_runner] benchmarking.tests[{i}] must be a mapping")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"[mlops/core/benchmark_runner] benchmarking.tests[{i}].name must be non-empty")
        enabled = bool(item.get("enabled", True))
        params = {k: v for k, v in item.items() if k not in {"name", "enabled"}}
        out.append(BenchmarkTestPlan(name=name, enabled=enabled, params=params))
    return out


def _registry_filenames(compiled: Mapping[str, Any]) -> Dict[str, str]:
    """
    Pull filenames from compiled.registry.*.filename if present.
    Else use stable defaults.
    """
    reg = compiled.get("registry", {}) or {}

    def _filename(path: Sequence[str], default: str) -> str:
        cur: Any = reg
        for k in path:
            if not isinstance(cur, Mapping):
                return default
            cur = cur.get(k)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
        return default

    return {
        "auto_run_map": _filename(["auto_run_map", "filename"], "auto_run_map.json"),
        "leaderboard": _filename(["leaderboard", "filename"], "leaderboard.csv"),
        "pareto_front": _filename(["pareto_front", "filename"], "pareto_front.json"),
    }


def _trial_tag(*, hypothesis_id: str, trial_id: Optional[int], sota_trial: Optional[str]) -> str:
    """
    Deterministic tag for artifacts. Stable across modules.
    Priority:
      sota_trial if provided (already formatted by your naming policy)
      else hypothesis_id + trial_id
    """
    h = str(hypothesis_id).strip() or "H?"
    if sota_trial and str(sota_trial).strip():
        return str(sota_trial).strip()
    if trial_id is None:
        return f"{h}_trial_unknown"
    return f"{h}_trial_{int(trial_id):04d}"


def _project_name(compiled: Mapping[str, Any]) -> str:
    # compiled might have project.name or project_name
    p = compiled.get("project", {}) or {}
    if isinstance(p, Mapping):
        name = p.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    pn = compiled.get("project_name")
    if isinstance(pn, str) and pn.strip():
        return pn.strip()
    return "unknown_project"


def _get_path_value(
    resolved_paths: Optional[Mapping[str, str]],
    paths_block: Mapping[str, Any],
    key: str,
) -> str:
    """
    Prefer resolved_paths[key] else paths_block[key].
    """
    if resolved_paths and isinstance(resolved_paths.get(key), str) and str(resolved_paths[key]).strip():
        return str(resolved_paths[key]).strip()
    v = paths_block.get(key, "")
    if isinstance(v, str) and v.strip():
        return v.strip()
    # fallback: empty means "executor decides", but we return stable placeholder
    return f"<missing:{key}>"


def _join(root: str, *parts: str) -> str:
    """
    Pure string join (no Path resolution side effects).
    Uses "/" to keep deterministic across OS. Executor can normalize later.
    """
    r = (root or "").rstrip("/")
    xs = [p.strip("/").strip() for p in parts if isinstance(p, str) and p.strip()]
    if not r:
        return "/".join(xs)
    if not xs:
        return r
    return r + "/" + "/".join(xs)


def _normalize_dict(v: Any) -> Dict[str, Any]:
    if v is None:
        return {}
    if isinstance(v, dict):
        return dict(v)
    raise TypeError("[mlops/core/benchmark_runner] expected mapping/dict")


def _safe_shallow_copy(v: Any) -> Any:
    if isinstance(v, dict):
        return dict(v)
    if isinstance(v, list):
        return list(v)
    return v


def _now_iso() -> str:
    return now_iso()
