# services/mlops/core/optuna_engine.py
# ============================================================
# Optuna engine wrapper (CORE / PURE) — production-grade
#
# PURE: no file I/O, no YAML loading, no path resolution
# No SCC coupling: does NOT import sweep_engine / SCC registry helpers
# Ask/Tell interface:
#       ask(space)  -> params
#       tell(...)   -> COMPLETE | PRUNED | FAIL
#
# Supports objective direction schemas:
#   A) {"accuracy": "maximize", "loss": "minimize"}
#   B) {"minimize": ["loss"], "maximize": ["accuracy", ...]}
#
# Author: Shraddha Mishra
# ============================================================

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union, Literal, Tuple, TYPE_CHECKING

Json = Dict[str, Any]
TrialState = Literal["COMPLETE", "PRUNED", "FAIL"]

try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover
    optuna = None

if TYPE_CHECKING:  # pragma: no cover
    from optuna import Trial
    from optuna.study import Study
else:
    Trial = Any
    Study = Any


# ============================================================
# Data models
# ============================================================

@dataclass(frozen=True)
class Budget:
    """
    Budget contract taken from compiled.search.budget
    """
    max_trials: int
    parallel_trials: int = 1
    max_gpu_hours: Optional[float] = None
    max_wall_hours: Optional[float] = None


@dataclass
class AskResult:
    """
    Returned by engine.ask().
    """
    trial_id: int
    trial: Any
    params: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Helpers (pure)
# ============================================================

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_s() -> float:
    return time.time()


def _safe_str(v: Any) -> str:
    try:
        return str(v)
    except Exception:
        return ""


def _is_nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _as_int(v: Any, *, ctx: str, default: int) -> int:
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"[core/optuna_engine] {ctx} must be int")
    return int(v)


def _as_float(v: Any, *, ctx: str, default: Optional[float]) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"[core/optuna_engine] {ctx} must be float")
    return float(v)


def _as_bool(v: Any, *, ctx: str, default: bool) -> bool:
    if v is None:
        return default
    if not isinstance(v, bool):
        raise ValueError(f"[core/optuna_engine] {ctx} must be bool")
    return bool(v)


def _require_mapping(x: Any, *, ctx: str) -> Json:
    if x is None:
        return {}
    if not isinstance(x, dict):
        raise ValueError(f"[core/optuna_engine] {ctx} must be a mapping")
    return x


# ============================================================
# Search-space suggestion (generic)
# ============================================================

def suggest_from_space(trial: "Trial", space: Any, *, name_prefix: str = "") -> Any:
    """
    Suggest values from a *generic* search-space object.

    Supported:
      - scalar -> scalar (fixed)
      - list   -> categorical by choosing an index (safe for dict payloads)
      - dict   -> recurse keys

    Notes:
      - trainer/SCA variant indices use categorical distributions; keep their
        variant tables fixed within a study (start a new study after edits)
      - list items may be dicts/lists/scalars; dict payloads become nested suggestions
      - we avoid suggest_categorical for dict payloads (unhashable) by using idx trick
    """
    if space is None:
        return None

    # Fixed scalar
    if isinstance(space, (str, int, float, bool)):
        return space

    # List -> choose index then decode
    if isinstance(space, list):
        if not space:
            raise ValueError(f"[core/optuna_engine] empty list space at '{name_prefix}'")

        idx_name = f"{name_prefix}.__idx" if name_prefix else "__idx"
        if name_prefix in {"trainer_variant_index", "sca_variant_index"}:
            # Variant ordering has no numerical meaning. Use a separate name
            # from legacy integer distributions in existing studies.
            idx = trial.suggest_categorical(f"{name_prefix}.__category", list(range(len(space))))
        else:
            idx = trial.suggest_int(idx_name, 0, len(space) - 1)
        chosen = space[int(idx)]

        if isinstance(chosen, dict):
            out: Json = {}
            for k, v in chosen.items():
                if not _is_nonempty_str(k):
                    raise ValueError(f"[core/optuna_engine] invalid dict key in list space at '{name_prefix}'")
                child = f"{name_prefix}.{k}" if name_prefix else str(k)
                out[str(k)] = suggest_from_space(trial, v, name_prefix=child)
            return out

        if isinstance(chosen, list):
            return chosen

        if isinstance(chosen, (str, int, float, bool)) or chosen is None:
            return chosen

        # Unknown payload type -> treat as fixed
        return chosen

    # Dict -> recurse
    if isinstance(space, dict):
        out: Json = {}
        for k, v in space.items():
            if not _is_nonempty_str(k):
                raise ValueError(f"[core/optuna_engine] invalid dict key at '{name_prefix}'")
            child = f"{name_prefix}.{k}" if name_prefix else str(k)
            out[str(k)] = suggest_from_space(trial, v, name_prefix=child)
        return out

    # Unknown type -> keep fixed
    return space


# ============================================================
# Optuna Engine (PURE core)
# ============================================================

class OptunaEngine:
    """
    Ask/Tell engine wrapping an Optuna Study.

    PURE search-core: no SCC coupling, no registry logic, no file I/O.

    - Initializes Optuna study from compiled.search + compiled.objectives
    - ask(space) -> params dict
    - tell(trial_id, values, state) -> finalize trial

    Orchestration (SCC execution, metrics reading, pruning decisions, registries)
    belongs to: services/mlops/core/optuna_runner.py
    """

    def __init__(
        self,
        *,
        compiled: Json,
        study_name: Optional[str] = None,
        seed_salt: str = "MLSCA_OPTUNA_V1",
        start_time_s: Optional[float] = None,
        # optional explicit overrides
        storage: Optional[str] = None,
        load_if_exists: Optional[bool] = None,
    ) -> None:
        if optuna is None:
            raise RuntimeError(
                "[core/optuna_engine] optuna is not installed. Install optuna or disable search."
            )

        if not isinstance(compiled, dict):
            raise ValueError("[core/optuna_engine] compiled must be a dict")

        self.compiled: Json = compiled
        self.seed_salt = str(seed_salt)
        self.start_time_s = float(start_time_s) if start_time_s is not None else _now_s()

        # best-effort identity fields (engine doesn't enforce schema)
        self.project_name = _safe_str((compiled.get("project") or {}).get("name") or "")
        self.env = _safe_str(compiled.get("env") or "")
        self.model_name = _safe_str(compiled.get("model_name") or "")

        self.search_cfg: Json = _require_mapping(compiled.get("search"), ctx="compiled.search")
        self.objectives_cfg: Json = _require_mapping(compiled.get("objectives"), ctx="compiled.objectives")

        # Objectives: stable order + directions
        dir_cfg = self.objectives_cfg.get("directions")
        self.objective_metrics, self.directions = self._parse_directions_and_metrics(dir_cfg)

        if not self.objective_metrics or not self.directions:
            raise ValueError("[core/optuna_engine] objectives.directions is empty/invalid")
        if len(self.objective_metrics) != len(self.directions):
            raise RuntimeError("[core/optuna_engine] internal error: objectives metrics/directions mismatch")

        # Budget
        budget_cfg = _require_mapping(self.search_cfg.get("budget"), ctx="search.budget")
        self.budget = Budget(
            max_trials=_as_int(budget_cfg.get("max_trials"), ctx="search.budget.max_trials", default=1),
            parallel_trials=_as_int(budget_cfg.get("parallel_trials"), ctx="search.budget.parallel_trials", default=1),
            max_gpu_hours=_as_float(budget_cfg.get("max_gpu_hours"), ctx="search.budget.max_gpu_hours", default=None),
            max_wall_hours=_as_float(budget_cfg.get("max_wall_hours"), ctx="search.budget.max_wall_hours", default=None),
        )
        if self.budget.max_trials <= 0:
            raise ValueError("[core/optuna_engine] search.budget.max_trials must be > 0")
        if self.budget.parallel_trials <= 0:
            self.budget = Budget(
                max_trials=self.budget.max_trials,
                parallel_trials=1,
                max_gpu_hours=self.budget.max_gpu_hours,
                max_wall_hours=self.budget.max_wall_hours,
            )

        # Study config
        study_cfg = _require_mapping(self.search_cfg.get("study"), ctx="search.study")

        if storage is None:
            storage = study_cfg.get("storage")
        if storage is not None and not isinstance(storage, str):
            raise ValueError("[core/optuna_engine] search.study.storage must be string or null")

        if load_if_exists is None:
            load_if_exists = True if storage else False
        if not isinstance(load_if_exists, bool):
            raise ValueError("[core/optuna_engine] load_if_exists must be bool")

        # Deterministic study name
        cfg_study_name = study_cfg.get("study_name") or study_cfg.get("name")
        if study_name is None and _is_nonempty_str(cfg_study_name):
            study_name = str(cfg_study_name).strip()

        if not study_name:
            seed = _sha(f"{self.seed_salt}::proj={self.project_name}::env={self.env}::model={self.model_name}")[:16]
            study_name = f"mlsca_{self.project_name or 'project'}_{seed}"
        self.study_name = str(study_name)

        # Sampler + Pruner
        self.sampler = self._build_sampler(_require_mapping(self.search_cfg.get("sampler"), ctx="search.sampler"),
                                           study_name=self.study_name)
        self.pruner = self._build_pruner(_require_mapping(self.search_cfg.get("pruner"), ctx="search.pruner"))

        # Create/load study
        self.storage = str(storage) if storage else None
        self.load_if_exists = bool(load_if_exists)

        try:
            self.study = optuna.create_study(
                study_name=self.study_name,
                directions=self.directions,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.storage,
                load_if_exists=self.load_if_exists if self.storage else False,
            )
        except TypeError:
            # compatibility: older optuna
            self.study = optuna.create_study(
                study_name=self.study_name,
                directions=self.directions,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.storage,
            )

        # Track asked trials (trial_id -> optuna Trial object)
        self._asked: Dict[int, Any] = {}

    # --------------------------
    # Budget / stopping
    # --------------------------
    def _terminal_trial_count(self) -> int:
        """
        Count only finalized trials that should consume budget.

        We intentionally exclude non-terminal asked/running trials so that
        duplicate sampling attempts or other transient ask-side artifacts do
        not prematurely exhaust the search budget.
        """
        terminal_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }

        count = 0
        for t in self.study.trials:
            try:
                if t.state in terminal_states:
                    count += 1
            except Exception:
                continue
        return count
    def should_stop(self, *, now_s: Optional[float] = None, used_gpu_hours: Optional[float] = None) -> bool:
        """
        Pure stop gate. Caller supplies used_gpu_hours if tracked externally.

        IMPORTANT:
        Budget is consumed only by terminal trials (COMPLETE/PRUNED/FAIL),
        not by transient asked/running trials.
        """
        if self._terminal_trial_count() >= self.budget.max_trials:
            return True

        if self.budget.max_wall_hours is not None:
            t = float(now_s) if now_s is not None else _now_s()
            elapsed_h = (t - self.start_time_s) / 3600.0
            if elapsed_h >= float(self.budget.max_wall_hours):
                return True

        if self.budget.max_gpu_hours is not None and used_gpu_hours is not None:
            if float(used_gpu_hours) >= float(self.budget.max_gpu_hours):
                return True

        return False

    # --------------------------
    # Ask / Tell
    # --------------------------

    def ask(
        self,
        *,
        space: Json,
        trial_user_attrs: Optional[Dict[str, Any]] = None,
    ) -> AskResult:
        """
        Ask Optuna for a new trial and sample `space` into concrete params.

        `space` is produced by trial_compiler in runner (engine stays generic).
        """
        if self.should_stop():
            raise RuntimeError("[core/optuna_engine] budget exceeded: should_stop() == True")

        if not isinstance(space, dict):
            raise ValueError("[core/optuna_engine] ask(space=...) must receive a dict")

        trial = self.study.ask()
        trial_id = int(trial.number)

        try:
            params = suggest_from_space(trial, space, name_prefix="")
        except Exception:
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise
        if not isinstance(params, dict):
            # normalize scalar output into dict
            params = {"value": params}

        # Attach attrs (best-effort; engine stays pure w/out logging)
        if trial_user_attrs:
            for k, v in trial_user_attrs.items():
                try:
                    trial.set_user_attr(str(k), v)
                except Exception:
                    pass

        self._asked[trial_id] = trial
        return AskResult(
            trial_id=trial_id,
            trial=trial,
            params=params,
            meta={
                "study_name": self.study_name,
                "storage": self.storage,
                "objective_metrics": list(self.objective_metrics),
                "directions": list(self.directions),
            },
        )

    def tell(
        self,
        *,
        trial_id: int,
        values: Optional[Union[float, Sequence[float], Dict[str, float]]] = None,
        state: TrialState = "COMPLETE",
        # Optional pruning signal: Optuna expects ONE scalar for report()
        intermediate_value: Optional[float] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Report final results for a trial.

        values accepted:
          - float (single objective only)
          - sequence of floats (must match objective order length)
          - dict metric->value (engine orders by objective_metrics deterministically)

        Values are required only for COMPLETE. PRUNED uses intermediate reports;
        FAIL and PRUNED do not pass final values to Study.tell().

        intermediate_value:
          - optional single scalar to report for pruner decisions
          - selection of which metric to report belongs in runner/trial_judge
        """
        if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 0:
            raise ValueError("[core/optuna_engine] trial_id must be non-negative int")

        st = str(state).upper().strip()
        if st not in ("COMPLETE", "PRUNED", "FAIL"):
            raise ValueError("[core/optuna_engine] state must be COMPLETE|PRUNED|FAIL")
        final_vals = self._normalize_values(values) if st == "COMPLETE" else None
        trial = self._asked.get(trial_id)
        if trial is None:
            self._find_trial_by_number(trial_id)
            trial = trial_id

        # Optional pruning report
        if intermediate_value is not None:
            if isinstance(trial, int):
                raise ValueError("Intermediate reporting requires a live trial returned by this engine.ask()")
            if isinstance(intermediate_value, bool) or not isinstance(intermediate_value, (int, float)):
                raise ValueError("[core/optuna_engine] intermediate_value must be numeric")
            s = int(step) if step is not None else 0
            try:
                trial.report(float(intermediate_value), step=s)
            except Exception:
                # keep engine resilient; runner decides pruning anyway
                pass

        if st == "COMPLETE":
            self.study.tell(trial, final_vals, state=optuna.trial.TrialState.COMPLETE)
        elif st == "PRUNED":
            # Optuna derives a pruned trial's value from intermediate reports.
            self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)

        self._asked.pop(trial_id, None)

    # --------------------------
    # Sampler / Pruner builders
    # --------------------------

    def _build_sampler(self, sampler_cfg: Json, *, study_name: str):
        name = str(sampler_cfg.get("name") or "tpe").strip().lower()

        # deterministic seed derived from study_name unless explicit seed in config
        seed_base = _sha(f"{self.seed_salt}::sampler::{study_name}")[:8]
        seed_int = int(seed_base, 16) % (2**31 - 1)

        seed = sampler_cfg.get("seed")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("[core/optuna_engine] search.sampler.seed must be int")
            seed_int = int(seed)

        if name == "random":
            return optuna.samplers.RandomSampler(seed=seed_int)

        if name == "grid":
            # GridSampler requires an explicit global grid search space.
            # Engine stays pure: runner must provide this and/or use dedicated grid mode.
            raise ValueError(
                "[core/optuna_engine] sampler 'grid' requires explicit GridSampler space; "
                "use sampler.name != 'grid' or implement grid space in runner."
            )

        if name == "cmaes":
            # CMA-ES only supports single-objective robustly; fallback to TPE for multi-objective
            if len(self.directions) > 1:
                return optuna.samplers.TPESampler(seed=seed_int)
            return optuna.samplers.CmaEsSampler(seed=seed_int)

        # default: TPE
        startup_trials = sampler_cfg.get("startup_trials", 10)
        if isinstance(startup_trials, bool) or not isinstance(startup_trials, int) or startup_trials < 0:
            raise ValueError("[core/optuna_engine] search.sampler.startup_trials must be non-negative int")

        multivariate = _as_bool(sampler_cfg.get("multivariate"), ctx="search.sampler.multivariate", default=True)
        group = _as_bool(sampler_cfg.get("group"), ctx="search.sampler.group", default=True)

        return optuna.samplers.TPESampler(
            seed=seed_int,
            n_startup_trials=int(startup_trials),
            multivariate=multivariate,
            group=group,
        )

    def _build_pruner(self, pruner_cfg: Json):
        name = str(pruner_cfg.get("name") or "asha").strip().lower()
        if name in ("none", "off", "disabled"):
            return optuna.pruners.NopPruner()

        min_resource = pruner_cfg.get("min_resource", 1)
        reduction_factor = pruner_cfg.get("reduction_factor", 3)
        min_early = pruner_cfg.get("min_early_stopping_rate", 0)

        for k, v in {
            "min_resource": min_resource,
            "reduction_factor": reduction_factor,
            "min_early_stopping_rate": min_early,
        }.items():
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"[core/optuna_engine] search.pruner.{k} must be non-negative int")

        if name == "hyperband":
            rf = int(reduction_factor) if int(reduction_factor) > 0 else 3
            return optuna.pruners.HyperbandPruner(
                min_resource=int(min_resource),
                reduction_factor=rf,
            )

        # default: ASHA / SuccessiveHalving
        rf = int(reduction_factor) if int(reduction_factor) > 0 else 3
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(min_resource),
            reduction_factor=rf,
            min_early_stopping_rate=int(min_early),
        )

    # --------------------------
    # Objectives parsing (stable + strict)
    # --------------------------

    def _parse_directions_and_metrics(self, directions_cfg: Any) -> Tuple[List[str], List[str]]:
        """
        Returns:
          objective_metrics: [metric1, metric2, ...]  (stable order)
          directions:        ["minimize"|"maximize", ...] same length/order

        Supports:
          A) {metric: minimize/maximize, ...}
          B) {minimize: [m1,m2], maximize: [m3,...]}
        """
        if directions_cfg is None:
            return ([], [])
        if not isinstance(directions_cfg, dict) or not directions_cfg:
            return ([], [])

        # Schema B
        if "minimize" in directions_cfg or "maximize" in directions_cfg:
            mins = directions_cfg.get("minimize", []) or []
            maxs = directions_cfg.get("maximize", []) or []
            if not isinstance(mins, list) or not isinstance(maxs, list):
                raise ValueError("[core/optuna_engine] objectives.directions minimize/maximize must be lists")

            metrics: List[str] = []
            dirs: List[str] = []

            for m in mins:
                if not _is_nonempty_str(m):
                    continue
                metrics.append(str(m).strip())
                dirs.append("minimize")

            for m in maxs:
                if not _is_nonempty_str(m):
                    continue
                metrics.append(str(m).strip())
                dirs.append("maximize")

            # de-duplicate while preserving first occurrence
            seen = set()
            metrics2: List[str] = []
            dirs2: List[str] = []
            for m, d in zip(metrics, dirs):
                if m in seen:
                    continue
                seen.add(m)
                metrics2.append(m)
                dirs2.append(d)

            return (metrics2, dirs2)

        # Schema A (enforce deterministic ordering by metric name)
        tmp: List[Tuple[str, str]] = []
        for metric, direction in directions_cfg.items():
            if not _is_nonempty_str(metric):
                continue
            m = str(metric).strip()
            d = str(direction).strip().lower()
            if d not in ("minimize", "maximize"):
                raise ValueError(f"[core/optuna_engine] objectives.directions.{m} must be minimize/maximize")
            tmp.append((m, d))

        # deterministic: sort by metric name, then de-dup by first metric occurrence
        tmp.sort(key=lambda x: x[0])

        seen = set()
        metrics2: List[str] = []
        dirs2: List[str] = []
        for m, d in tmp:
            if m in seen:
                continue
            seen.add(m)
            metrics2.append(m)
            dirs2.append(d)

        return (metrics2, dirs2)

    # --------------------------
    # Final values normalization
    # --------------------------

    def _normalize_values(self, values: Union[float, Sequence[float], Dict[str, float]]) -> List[float]:
        """
        Normalize final objective values to match objective_metrics ordering.
        """
        n_obj = len(self.directions)
        if n_obj <= 0:
            raise RuntimeError("[core/optuna_engine] invalid engine state: no objectives configured")

        # single scalar
        if isinstance(values, (int, float)) and not isinstance(values, bool):
            if n_obj != 1:
                raise ValueError(
                    "[core/optuna_engine] single value provided but multi-objective directions configured"
                )
            return [float(values)]

        # dict metric -> value
        if isinstance(values, dict):
            out: List[float] = []
            for m in self.objective_metrics:
                if m not in values:
                    raise ValueError(f"[core/optuna_engine] missing objective value for '{m}'")
                v = values[m]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"[core/optuna_engine] objective '{m}' must be numeric")
                out.append(float(v))
            return out

        # list/tuple
        if isinstance(values, (list, tuple)):
            if len(values) != n_obj:
                raise ValueError("[core/optuna_engine] values length must match number of objectives")
            out: List[float] = []
            for i, v in enumerate(values):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"[core/optuna_engine] values[{i}] must be numeric")
                out.append(float(v))
            return out

        raise ValueError("[core/optuna_engine] invalid values type for tell()")

    # --------------------------
    # Lookup helpers
    # --------------------------

    def _find_trial_by_number(self, trial_id: int):
        for t in self.study.trials:
            try:
                if int(t.number) == int(trial_id):
                    return t
            except Exception:
                continue
        raise KeyError(f"[core/optuna_engine] trial not found in study: {trial_id}")
