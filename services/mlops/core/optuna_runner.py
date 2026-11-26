# services/mlops/core/optuna_runner.py
"""
Optuna Runner (MLOps Orchestration Layer)
========================================

This module orchestrates AutoMLOps HPO runs by:
  1) loading + compiling AutoMLOps config (via MLOpsConfig)
  2) sampling trial params (via core/optuna_engine.py — ask/tell)
  3) compiling an SCC-safe run_cfg (via core/trial_compiler.py)
  4) executing SCC training ONLY when allowed (via utils/scc_bridge.py)
  5) reading SCC-produced metrics (via core/trial_reporter.py)
  6) judging outcomes (via core/trial_judge.py) and reporting to OptunaEngine

Key contracts
-------------
- SCC owns: naming/hash/run_dir/registry for *model* runs.
- MLOps owns: Optuna loops, trial registry, audit snapshots, decision logic.

PRUNED “no execution” contract (important)
------------------------------------------
If a trial is already known as PRUNED (e.g., persisted in the MLOps trial registry),
**we do not call SCC**. We only:
  - tell the OptunaEngine the trial is PRUNED
  - record the result in the runner output

If SCC execution raises a pruned signal (optuna.TrialPruned), we still mark it PRUNED,
but that is a “halted after start” case (not “no execution”).

This runner deliberately does NOT import or depend on SCC sweep_engine logic.
The only SCC coupling is the thin execution bridge (scc_bridge.py), which returns:
  {model_hash, model_name, run_dir?, metrics_dir? ...}

"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.mlops.utils.auto_config import MLOpsConfig
from services.mlops.core.optuna_engine import OptunaEngine
from services.mlops.core.trial_compiler import TrialCompiler, CompileResult
from services.mlops.core.trial_reporter import TrialReporter
from services.mlops.core.trial_judge import (
    TrialStage,
    TrialDecision,
    judge_trial,
    RC_PRUNED,
    RC_FAILED,
)
from services.mlops.utils.scc_bridge import run_trial_via_scc
from services.side_channel_core.utils.timestamps import now_str
from services.side_channel_core.utils.seal import load_json

Json = Dict[str, Any]

# Optional: Optuna pruned exception type (only for catching raised prunes from SCC)
try:  # pragma: no cover
    import optuna  # type: ignore

    _OPTUNA_TRIAL_PRUNED = optuna.TrialPruned
except Exception:  # pragma: no cover
    _OPTUNA_TRIAL_PRUNED = type("TrialPruned", (Exception,), {})


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = now_str()
    print(f"[mlops/optuna_runner {ts}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _select_project_block(auto_cfg: Json, project_name: str) -> Json:
    projects = auto_cfg.get("project") or []
    if not isinstance(projects, list):
        raise ValueError("[optuna_runner] auto_mlops.yaml 'project' must be a list")
    for p in projects:
        if isinstance(p, dict) and str(p.get("name", "")).strip() == project_name:
            return p
    raise KeyError(f"[optuna_runner] project '{project_name}' not found in auto_mlops.yaml")


def _select_hypothesis_block(project_block: Json, hypothesis_id: str) -> Json:
    hyps = project_block.get("hypotheses") or []
    if not isinstance(hyps, list):
        raise ValueError("[optuna_runner] project.hypotheses must be a list")
    for h in hyps:
        if isinstance(h, dict) and str(h.get("hypothesis_id", "")).strip() == hypothesis_id:
            return h
    raise KeyError(f"[optuna_runner] hypothesis '{hypothesis_id}' not found under project")


def _build_trial_space(hypothesis_block: Json) -> Json:
    """
    Convert a hypothesis block into a generic search-space dict for OptunaEngine.ask().

    Convention:
      - hypothesis.model.search_space -> params["model"][...]
      - hypothesis.trainer -> params["trainer"][...]
    """
    space: Json = {}

    model = hypothesis_block.get("model") or {}
    if isinstance(model, dict):
        ss = model.get("search_space")
        if isinstance(ss, dict):
            space["model"] = ss
        elif ss is not None:
            space["model"] = {"value": ss}

    trainer = hypothesis_block.get("trainer")
    if isinstance(trainer, dict):
        space["trainer"] = trainer

    # (optional) seed space if hypothesis provides it
    if "seed" in hypothesis_block:
        space["seed"] = hypothesis_block["seed"]

    return space


def _read_trial_registry_status(registry_path: Path, trial_id: str) -> Optional[str]:
    """
    Best-effort: check if this trial_id already exists and has a status.

    We keep this read-only and defensive; scc_bridge owns the writes.
    """
    if not registry_path.exists():
        return None
    try:
        reg = load_json(registry_path)  # utils/seal.py: verified, seal stripped
        if not isinstance(reg, dict):
            return None
        entry = reg.get(str(trial_id))
        if isinstance(entry, dict):
            st = entry.get("status")
            return str(st) if st is not None else None
        return None
    except Exception:
        return None


def _worst_values(objective_metrics: List[str]) -> Dict[str, float]:
    # A deterministic placeholder; engine.tell(FAIL/PRUNED) can ignore values,
    # but we provide something stable.
    return {m: 0.0 for m in objective_metrics}

def _find_metrics_csv(metrics_dir: Path) -> Optional[Path]:
    """
    Defensive: pick most likely metrics CSV inside metrics_dir.
    Prefers filenames containing 'epoch'/'metric', else newest *.csv.
    """
    if not metrics_dir.exists() or not metrics_dir.is_dir():
        return None
    csvs = list(metrics_dir.glob("*.csv"))
    if not csvs:
        return None

    def _score(p: Path) -> Tuple[int, float]:
        name = p.name.lower()
        s = 0
        if "epoch" in name:
            s += 2
        if "metric" in name:
            s += 2
        if "val" in name:
            s += 1
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (s, mtime)

    csvs.sort(key=_score, reverse=True)
    return csvs[0]


def _read_last_csv_row(csv_path: Path) -> Dict[str, Any]:
    """
    Minimal CSV reader (no pandas dependency). Returns last non-empty row.
    """
    import csv

    last: Optional[Dict[str, Any]] = None
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and any((v is not None and str(v).strip() != "") for v in row.values()):
                last = row
    return last or {}


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _read_objectives_fallback(*, run_cfg: Json, model_name: str, metrics_dir: Optional[str], objective_metrics: List[str]) -> Dict[str, float]:
    """
    Fallback objective reader if TrialReporter does not provide read_final_objectives.
    """
    # Prefer metrics_dir returned by bridge
    mdir = Path(metrics_dir) if isinstance(metrics_dir, str) and metrics_dir.strip() else None

    # Try SCC path template if available in cfg
    if mdir is None:
        try:
            from services.side_channel_core.utils.path_utils import resolve_path as scc_resolve_path  # type: ignore
            mdir = Path(scc_resolve_path(run_cfg["paths"]["metrics"], {**run_cfg, "model_name": model_name}))
        except Exception:
            mdir = None

    if mdir is None:
        raise RuntimeError("[optuna_runner] cannot determine metrics_dir for objective reading")

    csv_path = _find_metrics_csv(mdir)
    if csv_path is None:
        raise RuntimeError(f"[optuna_runner] no metrics CSV found in: {mdir}")

    last = _read_last_csv_row(csv_path)
    out: Dict[str, float] = {}
    for m in objective_metrics:
        raw = last.get(m)
        if raw is None:
            raw = last.get(m.replace("@", "_at_"))
        if raw is None:
            raw = last.get(m.replace("@", ""))
        val = _coerce_float(raw)
        if val is not None:
            out[m] = float(val)

    if not out:
        raise RuntimeError(f"[optuna_runner] could not read objectives {objective_metrics} from {csv_path.name}; cols={list(last.keys())}")

    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunnerTrialResult:
    trial_id: int
    hypothesis_id: str
    status: str  # DONE|PRUNED|FAIL
    model_name: str = ""
    model_hash: str = ""
    run_dir: Optional[str] = None
    objectives: Optional[Dict[str, float]] = None
    reason_code: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class OptunaRunSummary:
    attempted: int
    done: int
    pruned: int
    failed: int
    study_name: str


class OptunaRunner:
    """
    Orchestrates OptunaEngine ask/tell and SCC execution through scc_bridge.
    """

    def __init__(self, *, env: Optional[str] = None) -> None:
        self.env = str(env or os.environ.get("ENV", "dev")).strip() or "dev"
        self.cfg = MLOpsConfig()

    def run(
        self,
        *,
        project_name: str,
        hypothesis_id: str,
        mode: str = "train",
        max_trials_override: Optional[int] = None,
    ) -> Tuple[List[RunnerTrialResult], OptunaRunSummary]:
        """
        Run a single (project, hypothesis) HPO loop.

        Returns:
          (results, summary)
        """
        # 1) Load compiled AutoMLOps config (resolved + env aware)
        auto_cfg = self.cfg.get_map(
            "auto_mlops",
            project_name=project_name,
            resolve_as_path=False,
            mkdir=True,
            strict=True,
            extra={"env": self.env},
        )
        if not isinstance(auto_cfg, dict):
            raise TypeError("[optuna_runner] MLOpsConfig.get_map('auto_mlops') must return dict")

        project_block = _select_project_block(auto_cfg, project_name)
        hypothesis_block = _select_hypothesis_block(project_block, hypothesis_id)

        # 2) Build trial search space from hypothesis
        space = _build_trial_space(hypothesis_block)

        # 3) Engine (pure core)
        compiled_for_engine: Json = {**auto_cfg, "env": self.env, "project": {"name": project_name}}
        engine = OptunaEngine(compiled=compiled_for_engine)

        # 4) Compiler + Reporter (MLOps-owned)
        compiler = TrialCompiler(cfg=self.cfg)
        reporter = TrialReporter()

        # 5) Registry path (same key used by scc_bridge)
        registry_path_str = self.cfg.get_map(
            "auto_mlops",
            key_map="paths.auto_trial_registry",
            project_name=project_name,
            resolve_as_path=True,
            mkdir=True,
            strict=True,
        )
        registry_path = Path(registry_path_str)

        # Runner-level max_trials override
        if max_trials_override is not None:
            try:
                max_trials_override = int(max_trials_override)
            except Exception:
                raise ValueError("[optuna_runner] max_trials_override must be int")

        objective_metrics = list(engine.objective_metrics)

        results: List[RunnerTrialResult] = []
        attempted = 0

        _log(f"start: project={project_name} hypothesis={hypothesis_id} env={self.env} study={engine.study_name}")

        while True:
            if engine.should_stop():
                break
            if max_trials_override is not None and attempted >= max_trials_override:
                break

            # Ask for next trial parameters
            try:
                ask = engine.ask(
                    space=space,
                    trial_user_attrs={
                        "project_name": project_name,
                        "hypothesis_id": hypothesis_id,
                        "env": self.env,
                    },
                )
            except RuntimeError as e:
                _log(f"stop gate: {e}")
                break

            attempted += 1
            trial_id_i = int(ask.trial_id)
            trial_id_s = str(trial_id_i)

            # -----------------------------------------------------------------
            # PRUNED “no execution” gate:
            # if trial already marked PRUNED in registry, do not call SCC.
            # -----------------------------------------------------------------
            prev_status = _read_trial_registry_status(registry_path, trial_id_s)
            if prev_status and prev_status.upper() == "PRUNED":
                _log(f"Trial {trial_id_i} SKIP (already PRUNED in registry; no SCC execution)")
                try:
                    engine.tell(trial_id=trial_id_i, values=_worst_values(objective_metrics), state="PRUNED")
                except Exception:
                    pass
                results.append(
                    RunnerTrialResult(
                        trial_id=trial_id_i,
                        hypothesis_id=str(hypothesis_id),
                        status="PRUNED",
                        reason_code=RC_PRUNED,
                        reason="registry_pruned_skip",
                    )
                )
                continue

            # 1) Compile SCC run cfg (SCC-safe + uniqueness injection)
            try:
                comp: CompileResult = compiler.compile(
                    project_name=project_name,
                    hypothesis=hypothesis_block,
                    sampled_params=ask.params,
                    trial_id=trial_id_s,
                    hypothesis_id=str(hypothesis_id),
                    env=self.env,
                )
            except Exception as e:
                _log(f"Trial {trial_id_i} FAIL during compile: {e}")
                try:
                    engine.tell(trial_id=trial_id_i, values=_worst_values(objective_metrics), state="FAIL")
                except Exception:
                    pass
                results.append(
                    RunnerTrialResult(
                        trial_id=trial_id_i,
                        hypothesis_id=str(hypothesis_id),
                        status="FAIL",
                        reason_code=RC_FAILED,
                        reason=f"compile_error:{e}",
                    )
                )
                continue

            run_cfg = comp.run_cfg

            # 2) Execute SCC (unless pruned-before-run)
            try:
                _log(f"▶  Trial {trial_id_i}: SCC execute (hypothesis={hypothesis_id})")
                out = run_trial_via_scc(
                    run_cfg=run_cfg,
                    mode=mode,
                    trial_id=trial_id_s,
                    hypothesis_id=str(hypothesis_id),
                    sota_trial=None,
                )
            except _OPTUNA_TRIAL_PRUNED as e:
                # This is a prune-after-start (SCC raised pruned)
                _log(f"Trial {trial_id_i} PRUNED (raised during SCC execution): {e}")
                try:
                    engine.tell(trial_id=trial_id_i, values=_worst_values(objective_metrics), state="PRUNED")
                except Exception:
                    pass
                results.append(
                    RunnerTrialResult(
                        trial_id=trial_id_i,
                        hypothesis_id=str(hypothesis_id),
                        status="PRUNED",
                        reason_code=RC_PRUNED,
                        reason="scc_raised_trialpruned",
                    )
                )
                continue
            except Exception as e:
                _log(f"Trial {trial_id_i} FAIL during SCC execution: {e}")
                try:
                    engine.tell(trial_id=trial_id_i, values=_worst_values(objective_metrics), state="FAIL")
                except Exception:
                    pass
                results.append(
                    RunnerTrialResult(
                        trial_id=trial_id_i,
                        hypothesis_id=str(hypothesis_id),
                        status="FAIL",
                        reason_code=RC_FAILED,
                        reason=f"scc_exec_error:{e}",
                    )
                )
                continue

            model_name = str(out.get("model_name") or "")
            model_hash = str(out.get("model_hash") or "")
            run_dir = out.get("run_dir") or None
            metrics_dir = out.get("metrics_dir") or None

            # 3) Read metrics + objectives (final)
            try:
                if hasattr(reporter, "read_final_objectives"):
                    snapshot, objectives = reporter.read_final_objectives(
                        run_cfg=run_cfg,
                        model_name=model_name,
                        metrics_dir=metrics_dir,
                        objective_metrics=objective_metrics,
                    )
                else:
                    # Fallback: infer objectives from metrics CSV; build a minimal snapshot-like object
                    objectives = _read_objectives_fallback(
                        run_cfg=run_cfg,
                        model_name=model_name,
                        metrics_dir=metrics_dir,
                        objective_metrics=objective_metrics,
                    )
                    # Minimal snapshot object for judge_trial compatibility:
                    # - Must have .metrics dict
                    class _Snap:
                        def __init__(self, metrics: Dict[str, Any]):
                            self.metrics = metrics
                    snapshot = _Snap(metrics=objectives)  # type: ignore
            except Exception as e:
                _log(f"Trial {trial_id_i} FAIL reading objectives: {e}")
                try:
                    engine.tell(trial_id=trial_id_i, values=_worst_values(objective_metrics), state="FAIL")
                except Exception:
                    pass
                results.append(
                    RunnerTrialResult(
                        trial_id=trial_id_i,
                        hypothesis_id=str(hypothesis_id),
                        status="FAIL",
                        model_name=model_name,
                        model_hash=model_hash,
                        run_dir=str(run_dir) if run_dir else None,
                        reason_code=RC_FAILED,
                        reason=f"metrics_read_error:{e}",
                    )
                )
                continue

            # 4) Judge trial (final stage)
            try:
                verdict = judge_trial(
                    snapshot=snapshot,
                    pruning=None,  # v1: pruning may be injected later (epoch streaming)
                    compiled=compiled_for_engine,
                    stage=TrialStage.FINAL,
                )
            except Exception as e:
                _log(f"Trial {trial_id_i} FAIL in trial_judge: {e}")
                try:
                    engine.tell(trial_id=trial_id_i, values=_worst_values(objective_metrics), state="FAIL")
                except Exception:
                    pass
                results.append(
                    RunnerTrialResult(
                        trial_id=trial_id_i,
                        hypothesis_id=str(hypothesis_id),
                        status="FAIL",
                        model_name=model_name,
                        model_hash=model_hash,
                        run_dir=str(run_dir) if run_dir else None,
                        objectives=objectives,
                        reason_code=RC_FAILED,
                        reason=f"judge_error:{e}",
                    )
                )
                continue

            # 5) Tell engine based on verdict
            if verdict.decision in (TrialDecision.HALT,) and verdict.reason_code == RC_PRUNED:
                # Note: in FINAL stage, HALT usually means something went wrong or pruned late.
                # We reflect as PRUNED.
                _log(f"Trial {trial_id_i} PRUNED by judge: {verdict.reason}")
                try:
                    engine.tell(trial_id=trial_id_i, values=objectives, state="PRUNED")
                except Exception:
                    pass
                results.append(
                    RunnerTrialResult(
                        trial_id=trial_id_i,
                        hypothesis_id=str(hypothesis_id),
                        status="PRUNED",
                        model_name=model_name,
                        model_hash=model_hash,
                        run_dir=str(run_dir) if run_dir else None,
                        objectives=objectives,
                        reason_code=verdict.reason_code,
                        reason=verdict.reason,
                    )
                )
                continue

            # Default: DONE (even if REJECT/QUALIFY/ACCEPT — those are your internal semantics)
            _log(f"Trial {trial_id_i} DONE: {model_hash} ({model_name}) objectives={json.dumps(objectives)}")
            try:
                engine.tell(trial_id=trial_id_i, values=objectives, state="DONE")
            except Exception:
                # never crash runner because tell failed
                pass

            results.append(
                RunnerTrialResult(
                    trial_id=trial_id_i,
                    hypothesis_id=str(hypothesis_id),
                    status="DONE",
                    model_name=model_name,
                    model_hash=model_hash,
                    run_dir=str(run_dir) if run_dir else None,
                    objectives=objectives,
                    reason_code=verdict.reason_code,
                    reason=verdict.reason,
                )
            )

        summary = OptunaRunSummary(
            attempted=attempted,
            done=sum(1 for r in results if r.status == "DONE"),
            pruned=sum(1 for r in results if r.status == "PRUNED"),
            failed=sum(1 for r in results if r.status == "FAIL"),
            study_name=engine.study_name,
        )
        _log(f"done: attempted={summary.attempted} done ={summary.done} pruned={summary.pruned} fail={summary.failed}")
        return results, summary
