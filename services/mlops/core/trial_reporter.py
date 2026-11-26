# services/mlops/core/trial_reporter.py
"""
Trial Reporter (SCC outputs -> TrialResult)
==========================================

This module is **NOT** an orchestrator.

It is a pure-ish reporting layer that:
1) Accepts SCC execution output (from scc_bridge.run_trial_via_scc)
2) Locates SCC-written metrics (CSV) in a defensive way
3) Extracts objective metric values (based on objective_names)
4) Returns a structured TrialResult for OptunaRunner to "tell"

Design rules
------------
- No Optuna engine calls here (no ask/tell).
- No SCC execution here.
- No config compilation here.
- Best-effort file IO only; never crashes the whole HPO loop unless caller requests strict.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional,  Tuple
from services.side_channel_core.utils.timestamps import now_str


JSONDict = Dict[str, Any]


# -----------------------------------------------------------------------------
# Light logging
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = now_str()
    print(f"[mlops/trial_reporter {ts}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Data contracts
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialMeta:
    """
    Minimal metadata required to interpret SCC outputs.
    The runner/compiler can add richer fields elsewhere; reporter only needs these.
    """
    project_name: str
    env: str
    trial_id: str
    hypothesis_id: str
    objective_names: List[str]  # expected objective columns (ordered)


@dataclass(frozen=True)
class TrialResult:
    trial_id: int
    model_hash: str
    model_name: str
    status: str  # COMPLETE | PRUNED | FAIL
    objectives: Dict[str, float]
    run_dir: Optional[str] = None

    # extra diagnostics (safe defaults; backwards compatible)
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    metrics_csv: Optional[str] = None
    metrics_row: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _read_last_csv_row(csv_path: Path) -> Dict[str, Any]:
    """
    Minimal CSV reader (no pandas). Returns last non-empty row as dict.
    """
    last: Optional[Dict[str, Any]] = None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and any((v is not None and str(v).strip() != "") for v in row.values()):
                last = row
    return last or {}


def _score_csv_candidate(p: Path) -> Tuple[int, float]:
    """
    Higher is better.
    Prefer epoch/metrics/val-like filenames and newest mtime.
    """
    name = p.name.lower()
    s = 0
    if "epoch" in name:
        s += 3
    if "metric" in name:
        s += 3
    if "train" in name:
        s += 1
    if "val" in name:
        s += 1
    try:
        mt = p.stat().st_mtime
    except Exception:
        mt = 0.0
    return (s, mt)


def _find_metrics_csv(metrics_dir: Path) -> Optional[Path]:
    if not metrics_dir.exists() or not metrics_dir.is_dir():
        return None
    csvs = list(metrics_dir.glob("*.csv"))
    if not csvs:
        return None
    csvs_sorted = sorted(csvs, key=_score_csv_candidate, reverse=True)
    return csvs_sorted[0]


def _candidate_objective_keys(obj: str) -> List[str]:
    """
    SCC may log slightly different column labels.
    Try a small, deterministic alias list.
    """
    base = obj.strip()
    if not base:
        return []
    cand = [base]

    # common normalizations
    if "@" in base:
        cand.append(base.replace("@", "_at_"))
        cand.append(base.replace("@", ""))
        cand.append(base.replace("@", "_"))
    else:
        # sometimes people store "rank_k_success@5" but objective was "rank_k_success_5"
        cand.append(base.replace("_at_", "@"))
        cand.append(base.replace("_", "@"))

    # also try lower/upper variants (csv headers are case-sensitive but users aren't)
    extra = []
    for k in cand:
        extra.extend([k.lower(), k.upper()])
    cand.extend(extra)

    # unique preserve order
    seen = set()
    out: List[str] = []
    for k in cand:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _infer_metrics_dir(meta: TrialMeta, scc_out: Mapping[str, Any]) -> Optional[Path]:
    """
    Best-effort inference.
    Priority:
      1) scc_out["metrics_dir"]
      2) scc_out["run_dir"]/metrics
      3) None (caller decides)
    """
    md = scc_out.get("metrics_dir")
    if isinstance(md, str) and md.strip():
        return Path(md)

    rd = scc_out.get("run_dir")
    if isinstance(rd, str) and rd.strip():
        return Path(rd) / "metrics"

    return None


# -----------------------------------------------------------------------------
# Reporter
# -----------------------------------------------------------------------------

class TrialReporter:
    """
    Converts SCC outputs into a TrialResult by extracting objective metrics.

    Usage (inside optuna_runner):
        result = reporter.report_final(meta=meta, scc_out=out)
        # engine.tell(trial_id, values=result.objectives, state=result.status)
    """

    def __init__(self, *, strict: bool = False) -> None:
        """
        strict=False:
          - never raises if metrics/objectives missing; returns FAIL TrialResult.
        strict=True:
          - raises RuntimeError when it cannot produce objectives.
        """
        self.strict = bool(strict)

    def report_final(self, *, meta: TrialMeta, scc_out: Mapping[str, Any]) -> TrialResult:
        # 0) basic identity from SCC
        trial_id_int = int(str(meta.trial_id).strip())
        model_name = str(scc_out.get("model_name") or "").strip()
        model_hash = str(scc_out.get("model_hash") or "").strip()
        run_dir = scc_out.get("run_dir")
        run_dir_s = str(run_dir).strip() if isinstance(run_dir, str) and run_dir.strip() else None

        # If SCC already marked status (rare), honor it if PRUNED/FAILED
        scc_status = str(scc_out.get("status") or "").upper().strip()
        if scc_status in {"PRUNED", "FAILED", "FAIL"}:
            return TrialResult(
                trial_id=trial_id_int,
                model_hash=model_hash,
                model_name=model_name,
                status="PRUNED" if scc_status == "PRUNED" else "FAIL",
                objectives={},
                run_dir=run_dir_s,
                reason_code=scc_status,
                reason="reported_by_scc_bridge",
            )

        # 1) locate metrics csv
        metrics_dir = _infer_metrics_dir(meta, scc_out)
        csv_path: Optional[Path] = None
        if metrics_dir is not None:
            csv_path = _find_metrics_csv(metrics_dir)

        if csv_path is None:
            msg = f"no metrics CSV found (metrics_dir={metrics_dir})"
            if self.strict:
                raise RuntimeError(f"[trial_reporter] {msg}")
            _log(f"{msg}")
            return TrialResult(
                trial_id=trial_id_int,
                model_hash=model_hash,
                model_name=model_name,
                status="FAIL",
                objectives={},
                run_dir=run_dir_s,
                reason_code="METRICS_NOT_FOUND",
                reason=msg,
            )

        # 2) read last row
        row = _read_last_csv_row(csv_path)
        if not row:
            msg = f"metrics CSV has no readable rows: {csv_path}"
            if self.strict:
                raise RuntimeError(f"[trial_reporter] {msg}")
            _log(f"{msg}")
            return TrialResult(
                trial_id=trial_id_int,
                model_hash=model_hash,
                model_name=model_name,
                status="FAIL",
                objectives={},
                run_dir=run_dir_s,
                reason_code="METRICS_EMPTY",
                reason=msg,
                metrics_csv=str(csv_path),
                metrics_row={},
            )

        # 3) extract objectives
        objectives: Dict[str, float] = {}
        missing: List[str] = []

        for obj in meta.objective_names:
            got: Optional[float] = None
            for key in _candidate_objective_keys(obj):
                if key in row:
                    got = _to_float(row.get(key))
                    if got is not None:
                        break
            if got is None:
                missing.append(obj)
            else:
                objectives[obj] = float(got)

        if missing:
            msg = f"missing objectives in metrics CSV: {missing} (csv={csv_path.name})"
            if self.strict:
                raise RuntimeError(f"[trial_reporter] {msg}")
            _log(f"{msg}")
            return TrialResult(
                trial_id=trial_id_int,
                model_hash=model_hash,
                model_name=model_name,
                status="FAIL",
                objectives=objectives,  # keep any we did find
                run_dir=run_dir_s,
                reason_code="OBJECTIVE_MISSING",
                reason=msg,
                metrics_csv=str(csv_path),
                metrics_row=row,
            )

        # 4) success
        return TrialResult(
            trial_id=trial_id_int,
            model_hash=model_hash,
            model_name=model_name,
            status="COMPLETE",
            objectives=objectives,
            run_dir=run_dir_s,
            reason_code="OK",
            reason="objectives_extracted",
            metrics_csv=str(csv_path),
            metrics_row=row,
        )
