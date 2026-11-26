# services/mlops/core/auto_mlops.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from services.mlops.utils.auto_config import AutoConfig
from services.mlops.utils.logger import mlops_logger, MLOpsLogger

from services.mlops.core.context import compile_context, CompiledContext
from services.mlops.core.registry import HypothesisRegistry, TrialRegistry, TrialRecord
from services.mlops.core.hypothesis_selector import HypothesisSelector
from services.mlops.core.auto_params_emitter import AutoParamsEmitter
from services.mlops.core.optuna_engine import OptunaEngine, OptunaTrialHandle
from services.mlops.core.metrics_reader import MetricsReader, MetricsSnapshot
from services.mlops.core.pruning import PruningEngine, PruningOutcome
from services.mlops.core.trial_judge import TrialJudge, TrialDecision
from services.mlops.core.benchmark_runner import BenchmarkPlanner, BenchmarkPlan

from services.mlops.core.training_launcher import (
    TrainingBackend,
    TrainRequest,
    TrainResult,
    BenchmarkRequest,
    BenchmarkResult,
)

from services.mlops.adapters.disk_registry_backend import DiskRegistryBackend


# ============================================================
# Runner runtime options (NOT part of auto_mlops.yaml)
# ============================================================

@dataclass(frozen=True)
class RunnerOptions:
    strict: bool = True
    resolve_paths: bool = True
    mkdir: bool = True
    touch_files: bool = False

    resume: bool = True

    hard_stop_on_exception: bool = False
    sleep_between_trials_sec: float = 0.0

    # Registry save cadence
    save_after_each_trial: bool = True


# ============================================================
# Runner
# ============================================================

class AutoMLOpsRunner:
    """
    Phase-3 Orchestrator (transaction coordinator).

    Owns side-effects:
      - logger writes
      - backend dispatch (train/benchmark)
      - registry persistence (via DiskRegistryBackend)

    Must NOT write artifacts itself.
    """

    def __init__(
        self,
        *,
        cfg: AutoConfig,
        training_backend: TrainingBackend,
        options: Optional[RunnerOptions] = None,
    ) -> None:
        self.cfg = cfg
        self.backend = training_backend
        self.opt = options or RunnerOptions()

        self.lg: Optional[MLOpsLogger] = None
        self.optuna: Optional[OptunaEngine] = None

        # Core engines (pure logic)
        self.selector = HypothesisSelector()
        self.emitter = AutoParamsEmitter()
        self.metrics_reader = MetricsReader()
        self.pruning_engine = PruningEngine()
        self.judge = TrialJudge()
        self.benchmark_planner = BenchmarkPlanner()

        # Registries + backend (wired in run() after paths are resolved)
        self.registry_backend: Optional[DiskRegistryBackend] = None
        self.hyp_reg: Optional[HypothesisRegistry] = None
        self.trial_reg: Optional[TrialRegistry] = None

    # ============================================================
    # Config helpers
    # ============================================================

    def _load_auto_mlops_cfg(self) -> Dict[str, Any]:
        # Preferred: cfg.load("auto_mlops") if you implemented it
        if hasattr(self.cfg, "load") and callable(getattr(self.cfg, "load")):
            return self.cfg.load("auto_mlops")  # type: ignore[attr-defined]
        # Fallback: get_map("auto_mlops") returns dict
        return self.cfg.get_map("auto_mlops")

    def _load_core_cfg(self, *, project_name: str) -> Dict[str, Any]:
        """
        Core cfg = runtime envelope (env/model_name/dataset_alias/etc).

        If you later want this to be project params.yaml, add it to tracking registry and
        load via cfg.load("params_<project>") or cfg.get_map(...).

        For now we keep it minimal but stable.
        """
        # If AutoConfig was built with a preloaded core_cfg, use it.
        core = getattr(self.cfg, "core_cfg", None)
        if isinstance(core, dict) and core:
            return core

        # Minimum required for path_resolver/build_context fallback
        return {
            "env": "dev",
            "project": {"name": project_name},
        }

    def _resolve_registry_base_dir(self, *, project_name: str) -> str:
        """
        Resolve registry base dir:
          base_dir = <auto_run_map_benchmark>/registry/

        This is intentional: registry snapshots live under plugin benchmark root,
        not under per-trial dirs.
        """
        # auto_mlops.paths.auto_run_map_benchmark is a directory template
        base_dir = self.cfg.get_map(
            "auto_mlops",
            "paths.auto_run_map_benchmark",
            project_name=project_name,
            resolve_as_path=True,
            mkdir=True,
            strict=self.opt.strict,
            extra=None,
            touch_file=False,
        )
        # Ensure stable subfolder
        return str(base_dir / "registry")

    # ============================================================
    # Registry wiring + resume semantics
    # ============================================================

    def _init_registries(self, *, project_name: str) -> None:
        base_dir = self._resolve_registry_base_dir(project_name=project_name)
        self.registry_backend = DiskRegistryBackend(base_dir=base_dir)

        self.hyp_reg = HypothesisRegistry(backend=self.registry_backend, namespace="hypotheses")
        self.trial_reg = TrialRegistry(backend=self.registry_backend, namespace="trials")

        if self.opt.resume:
            self.hyp_reg.load()
            self.trial_reg.load()

    def _safe_save_registries(self) -> None:
        if not self.opt.save_after_each_trial:
            return
        if self.hyp_reg is not None:
            self.hyp_reg.save()
        if self.trial_reg is not None:
            self.trial_reg.save()

    def _next_trial_id(self, *, project_name: str) -> int:
        """
        Deterministic next trial_id based on persisted registry.
        """
        assert self.trial_reg is not None

        existing = self.trial_reg.list(project_name=project_name)
        max_id = -1
        for tr in existing:
            if tr.trial_id is None:
                continue
            try:
                max_id = max(max_id, int(tr.trial_id))
            except Exception:
                continue
        return max_id + 1

    def _trial_already_terminal(self, *, trial_key: str) -> bool:
        """
        Skip re-running terminal trials on resume.
        """
        assert self.trial_reg is not None
        try:
            tr = self.trial_reg.get(trial_key=trial_key)
        except Exception:
            return False
        return tr.status in {"completed", "pruned", "failed", "promoted"}

    # ============================================================
    # Public entrypoint
    # ============================================================

    def run(self, *, project_name: str) -> Dict[str, Any]:
        auto_mlops_cfg = self._load_auto_mlops_cfg()
        core_cfg = self._load_core_cfg(project_name=project_name)

        # Compile base (global + project overrides; no hypothesis yet)
        compiled_base: CompiledContext = compile_context(
            auto_mlops_cfg=auto_mlops_cfg,
            project_name=project_name,
            hypothesis_id=None,
        )

        # Init registries (disk backend) BEFORE logger so resume semantics are available
        self._init_registries(project_name=project_name)
        assert self.hyp_reg is not None
        assert self.trial_reg is not None

        # Init logger (uses auto_mlops.paths.* file paths)
        self.lg = mlops_logger(core_cfg=core_cfg, mlops_cfg=compiled_base.auto_mlops_raw, project_name=project_name)
        self.lg.audit(code="RUN_START", msg="AutoMLOps run started", data={"project_name": project_name, "resume": self.opt.resume})

        # Init optuna engine (ask/tell)
        self.optuna = OptunaEngine.from_compiled(compiled_base)

        # Ensure hypotheses exist in HypothesisRegistry for counters/selection
        hyp_ids: Sequence[str] = self.selector.list_hypotheses(compiled_base)
        for hid in hyp_ids:
            self.hyp_reg.ensure(project_name=project_name, hypothesis_id=hid)

        # Budget
        max_trials = int(compiled_base.get("search.budget.max_trials") or 0)
        if max_trials <= 0:
            raise ValueError("[mlops/core/auto_mlops] search.budget.max_trials must be > 0")

        # Resume: choose deterministic starting trial_id
        trial_id_cursor = self._next_trial_id(project_name=project_name) if self.opt.resume else 0

        summary: Dict[str, Any] = {
            "project": project_name,
            "max_trials": max_trials,
            "resume": bool(self.opt.resume),
            "start_trial_id": int(trial_id_cursor),
            "completed_trials": 0,
            "promoted": [],
            "qualified": [],
            "pruned": 0,
            "failed": 0,
            "skipped_terminal": 0,
        }

        for _ in range(max_trials):
            try:
                # 1) Select hypothesis (allocation policy)
                hypothesis_id = self.selector.select_next(
                    compiled=compiled_base,
                    project_name=project_name,
                    hypothesis_registry=self.hyp_reg,
                    trial_registry=self.trial_reg,
                )
                self.lg.set_hypothesis(hypothesis_id)

                # 2) Compile hypothesis-specific view
                compiled: CompiledContext = compile_context(
                    auto_mlops_cfg=auto_mlops_cfg,
                    project_name=project_name,
                    hypothesis_id=hypothesis_id,
                )

                # 3) Ask Optuna (trial handle)
                #    We pass the explicit trial_id_cursor so resume remains deterministic.
                trial_handle: OptunaTrialHandle = self.optuna.ask(compiled=compiled, trial_id=int(trial_id_cursor))

                # trial_key must be stable & unique
                trial_key = f"{project_name}::{hypothesis_id}::{trial_handle.trial_name}"

                # Resume: skip terminal trials (if someone restarts mid-loop)
                if self.opt.resume and self._trial_already_terminal(trial_key=trial_key):
                    summary["skipped_terminal"] += 1
                    trial_id_cursor += 1
                    continue

                # 4) Propose params from hypothesis search spaces (with optuna trial)
                proposed_params = self.emitter.propose(
                    compiled=compiled,
                    trial_id=trial_handle.trial_id,
                    optuna_trial=trial_handle.optuna_trial,
                )

                # 5) Create trial record early (so state exists before training)
                self.trial_reg.upsert(
                    TrialRecord(
                        project_name=project_name,
                        hypothesis_id=hypothesis_id,
                        trial_key=trial_key,
                        trial_id=trial_handle.trial_id,
                        sota_trial=trial_handle.trial_name,
                        trial_name=trial_handle.trial_name,
                        status="running",
                        params=dict(proposed_params),
                    )
                )
                self.hyp_reg.bump(project_name=project_name, hypothesis_id=hypothesis_id, field_name="trials_started", delta=1)

                # Safe save (start-of-trial checkpoint)
                self._safe_save_registries()

                # 6) Launch training via backend (runner does NOT write files)
                self.lg.set_trial(trial_id=trial_handle.trial_id, sota_trial=trial_handle.trial_name)
                self.lg.decision(
                    code="TRIAL_START",
                    msg="Launching training trial",
                    data={"trial_key": trial_key, "trial_id": trial_handle.trial_id, "hypothesis_id": hypothesis_id, "params": proposed_params},
                )

                train_req = TrainRequest(
                    project_name=project_name,
                    hypothesis_id=hypothesis_id,
                    trial_id=trial_handle.trial_id,
                    trial_name=trial_handle.trial_name,
                    compiled=compiled.to_dict(),
                    params=dict(proposed_params),
                )
                train_res: TrainResult = self.backend.run_train(train_req)

                # 7) Read/normalize metrics
                metrics: MetricsSnapshot = self.metrics_reader.read(compiled=compiled, train_result=train_res)
                self.lg.metric(code="METRICS_READ", msg="Metrics snapshot acquired", data=metrics.to_dict())

                # 8) Execute pruning rules
                pruning: PruningOutcome = self.pruning_engine.evaluate(compiled=compiled, metrics=metrics)
                if pruning.should_prune:
                    self.lg.prune(code=pruning.code, msg=pruning.reason, data=pruning.to_dict())

                # 9) Judge decision (accept/reject/promote/qualify)
                decision: TrialDecision = self.judge.judge(compiled=compiled, metrics=metrics, pruning=pruning)
                self.lg.audit(code=decision.code, msg=decision.summary, data=decision.to_dict())

                # 10) Tell Optuna
                self.optuna.tell(compiled=compiled, trial_handle=trial_handle, metrics=metrics, decision=decision, pruning=pruning)

                # 11) Optional benchmark (plan + dispatch)
                bench_plan: Optional[BenchmarkPlan] = None
                bench_res: Optional[BenchmarkResult] = None
                if decision.should_benchmark:
                    bench_plan = self.benchmark_planner.plan(compiled=compiled, decision=decision)
                    self.lg.audit(code="BENCH_PLAN", msg="Benchmark plan created", data=bench_plan.to_dict())

                    bench_req = BenchmarkRequest(
                        project_name=project_name,
                        hypothesis_id=hypothesis_id,
                        trial_id=trial_handle.trial_id,
                        trial_name=trial_handle.trial_name,
                        compiled=compiled.to_dict(),
                        plan=bench_plan.to_dict(),
                        artifacts=train_res.artifacts,
                    )
                    bench_res = self.backend.run_benchmark(bench_req)

                # 12) Update registries (terminal transition)
                terminal_status = "completed"
                if pruning.should_prune:
                    terminal_status = "pruned"
                if decision.is_promoted:
                    terminal_status = "promoted"

                self.trial_reg.update_status(
                    trial_key=trial_key,
                    status=terminal_status,
                    reason_code=decision.code if not pruning.should_prune else pruning.code,
                    reason_msg=decision.summary if not pruning.should_prune else pruning.reason,
                    metrics=metrics.to_dict(),
                    objective_values=decision.objective_values,
                )

                # Attach artifacts (strings only; runner does not create paths)
                # `train_res.artifacts` is backend-defined dict[str,str]; we store selected common keys if present.
                self.trial_reg.attach_artifacts(
                    trial_key=trial_key,
                    auto_params_path=train_res.artifacts.get("auto_params_path"),
                    run_dir=train_res.artifacts.get("run_dir"),
                    metrics_path=train_res.artifacts.get("metrics_path"),
                    benchmark_dir=(bench_res.artifacts.get("benchmark_dir") if (bench_res and getattr(bench_res, "artifacts", None)) else train_res.artifacts.get("benchmark_dir")),
                )

                # Hypothesis counters
                if terminal_status == "completed":
                    self.hyp_reg.bump(project_name=project_name, hypothesis_id=hypothesis_id, field_name="trials_completed", delta=1)
                elif terminal_status == "pruned":
                    self.hyp_reg.bump(project_name=project_name, hypothesis_id=hypothesis_id, field_name="trials_pruned", delta=1)
                elif terminal_status == "promoted":
                    self.hyp_reg.bump(project_name=project_name, hypothesis_id=hypothesis_id, field_name="trials_promoted", delta=1)

                # Best tracker (primary objective)
                primary_metric_name = compiled.get("objectives.primary")
                primary_value = None
                if isinstance(primary_metric_name, str) and primary_metric_name:
                    v = metrics.metrics.get(primary_metric_name)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        primary_value = float(v)

                if decision.is_promoted and primary_value is not None:
                    self.hyp_reg.set_best(
                        project_name=project_name,
                        hypothesis_id=hypothesis_id,
                        best_trial_key=trial_key,
                        best_primary_metric=primary_value,
                    )

                # Save registries at safe point (end-of-trial)
                self._safe_save_registries()

                # Summary
                summary["completed_trials"] += 1
                if decision.is_promoted:
                    summary["promoted"].append(trial_handle.trial_name)
                if decision.is_qualified:
                    summary["qualified"].append(trial_handle.trial_name)
                if pruning.should_prune:
                    summary["pruned"] += 1

                if self.opt.sleep_between_trials_sec > 0:
                    time.sleep(self.opt.sleep_between_trials_sec)

                trial_id_cursor += 1

            except Exception as e:
                summary["failed"] += 1
                if self.lg is not None:
                    self.lg.exception(code="TRIAL_EXCEPTION", msg="Trial failed with exception", exc=e, data={"project": project_name})
                # Save registries even on exception (so we can resume deterministically)
                try:
                    self._safe_save_registries()
                except Exception:
                    pass
                if self.opt.hard_stop_on_exception:
                    break
                trial_id_cursor += 1
                continue

        if self.lg is not None:
            self.lg.audit(code="RUN_END", msg="AutoMLOps run finished", data=summary)

        # Final save (end-of-run safe point)
        try:
            if self.hyp_reg is not None:
                self.hyp_reg.save()
            if self.trial_reg is not None:
                self.trial_reg.save()
        except Exception:
            # Never crash on final save
            pass

        return summary


# ============================================================
# Convenience API
# ============================================================

def run_auto_mlops(
    *,
    cfg: AutoConfig,
    training_backend: TrainingBackend,
    project_name: str,
    options: Optional[RunnerOptions] = None,
) -> Dict[str, Any]:
    runner = AutoMLOpsRunner(cfg=cfg, training_backend=training_backend, options=options)
    return runner.run(project_name=project_name)
