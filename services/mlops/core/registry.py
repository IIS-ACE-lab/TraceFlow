# services/mlops/core/registry.py
# ============================================================
# In-memory registries + persistence adapters (interfaces only)
# - TrialRegistry, HypothesisRegistry
# - RegistryBackend protocol (disk backend implemented later in Phase 3)
#
# Pure core logic: no file I/O here.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
from services.side_channel_core.utils.timestamps import now_iso

Json = Dict[str, Any]


# ============================================================
# Helpers
# ============================================================

def _now_iso() -> str:
    return now_iso()


def require_non_empty(s: Optional[str], *, ctx: str) -> str:
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"[registry] {ctx} must be non-empty string")
    return s.strip()


def stable_key(*parts: str) -> str:
    """
    Deterministic stable composite key for registries.
    """
    return "::".join(parts)


# ============================================================
# Backend Protocol (interfaces only)
# ============================================================

@runtime_checkable
class RegistryBackend(Protocol):
    """
    Persistence adapter contract.

    Implementations (Phase 3) may write to disk (json/csv/sqlite/etc).
    Core must treat this as an optional plug-in: no I/O here.
    """

    def load(self, *, namespace: str) -> Optional[Json]:
        """
        Load persisted state for `namespace`.
        Return None if namespace is missing.
        Must not raise for "not found".
        """
        ...

    def save(self, *, namespace: str, state: Json) -> None:
        """
        Persist full state for `namespace`.
        Implementations may do atomic write, journaling, etc.
        """
        ...


# ============================================================
# Data models
# ============================================================

@dataclass(frozen=True)
class HypothesisRecord:
    """
    Minimal, stable record for a hypothesis within a project.
    """
    project_name: str
    hypothesis_id: str

    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    # Optional metadata and counters
    tags: Tuple[str, ...] = ()
    notes: Optional[str] = None

    # trial counters
    trials_started: int = 0
    trials_completed: int = 0
    trials_pruned: int = 0
    trials_failed: int = 0
    trials_promoted: int = 0

    # Optional current best snapshot (opaque to core)
    best_trial_key: Optional[str] = None
    best_primary_metric: Optional[float] = None

    def key(self) -> str:
        return stable_key(self.project_name, self.hypothesis_id)


@dataclass(frozen=True)
class TrialRecord:
    """
    Trial record is a normalized, schema-safe unit of experimentation.

    trial_key must be unique globally within a project run-map scope.
    A good default is: <project>::<hypothesis_id>::<trial_name>
    """
    project_name: str
    hypothesis_id: str
    trial_key: str

    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    # Trial identity fields
    trial_id: Optional[int] = None
    sota_trial: Optional[str] = None
    trial_name: Optional[str] = None

    # Lifecycle
    status: str = "created"  # created|running|completed|pruned|failed|promoted
    reason_code: Optional[str] = None
    reason_msg: Optional[str] = None

    # Artifacts (paths are strings; resolver in Phase 3)
    auto_params_path: Optional[str] = None
    run_dir: Optional[str] = None
    metrics_path: Optional[str] = None
    benchmark_dir: Optional[str] = None

    # Params + results (opaque payloads)
    params: Json = field(default_factory=dict)     # proposed hyperparams (materialized)
    metrics: Json = field(default_factory=dict)    # last known metrics snapshot
    objective_values: Json = field(default_factory=dict)  # extracted objectives

    def key(self) -> str:
        return self.trial_key


# ============================================================
# Registry Exceptions
# ============================================================

class RegistryError(ValueError):
    pass


class DuplicateKey(RegistryError):
    pass


class NotFound(RegistryError):
    pass


# ============================================================
# HypothesisRegistry (in-memory)
# ============================================================

class HypothesisRegistry:
    """
    In-memory hypothesis registry.
    Optional backend provides load/save hooks.

    This registry stores per-project hypotheses and counters.
    """

    def __init__(self, *, backend: Optional[RegistryBackend] = None, namespace: str = "hypotheses") -> None:
        self._backend = backend
        self._namespace = require_non_empty(namespace, ctx="namespace")
        self._items: Dict[str, HypothesisRecord] = {}

    # ---- backend hooks (no I/O unless backend does it) ----

    def load(self) -> None:
        if not self._backend:
            return
        state = self._backend.load(namespace=self._namespace)
        if not state:
            return
        self._items = _decode_hypothesis_state(state)

    def snapshot(self) -> Json:
        return _encode_hypothesis_state(self._items)

    def save(self) -> None:
        if not self._backend:
            return
        self._backend.save(namespace=self._namespace, state=self.snapshot())

    # ---- CRUD ----

    def upsert(self, record: HypothesisRecord) -> None:
        key = record.key()
        existing = self._items.get(key)
        if existing is None:
            self._items[key] = record
            return
        # merge counters deterministically (record is authoritative for fields except created_at)
        merged = HypothesisRecord(
            project_name=record.project_name,
            hypothesis_id=record.hypothesis_id,
            created_at=existing.created_at,
            updated_at=_now_iso(),
            tags=record.tags or existing.tags,
            notes=record.notes if record.notes is not None else existing.notes,
            trials_started=record.trials_started if record.trials_started else existing.trials_started,
            trials_completed=record.trials_completed if record.trials_completed else existing.trials_completed,
            trials_pruned=record.trials_pruned if record.trials_pruned else existing.trials_pruned,
            trials_failed=record.trials_failed if record.trials_failed else existing.trials_failed,
            trials_promoted=record.trials_promoted if record.trials_promoted else existing.trials_promoted,
            best_trial_key=record.best_trial_key if record.best_trial_key is not None else existing.best_trial_key,
            best_primary_metric=record.best_primary_metric if record.best_primary_metric is not None else existing.best_primary_metric,
        )
        self._items[key] = merged

    def ensure(self, *, project_name: str, hypothesis_id: str, tags: Optional[Sequence[str]] = None) -> HypothesisRecord:
        project_name = require_non_empty(project_name, ctx="project_name")
        hypothesis_id = require_non_empty(hypothesis_id, ctx="hypothesis_id")
        key = stable_key(project_name, hypothesis_id)
        rec = self._items.get(key)
        if rec is not None:
            return rec
        rec = HypothesisRecord(
            project_name=project_name,
            hypothesis_id=hypothesis_id,
            tags=tuple(tags) if tags else (),
        )
        self._items[key] = rec
        return rec

    def get(self, *, project_name: str, hypothesis_id: str) -> HypothesisRecord:
        key = stable_key(require_non_empty(project_name, ctx="project_name"), require_non_empty(hypothesis_id, ctx="hypothesis_id"))
        rec = self._items.get(key)
        if not rec:
            raise NotFound(f"[HypothesisRegistry] not found: {key}")
        return rec

    def list(self, *, project_name: Optional[str] = None) -> List[HypothesisRecord]:
        if project_name is None:
            return list(self._items.values())
        pn = require_non_empty(project_name, ctx="project_name")
        return [r for r in self._items.values() if r.project_name == pn]

    # ---- counters ----

    def bump(self, *, project_name: str, hypothesis_id: str, field_name: str, delta: int = 1) -> HypothesisRecord:
        rec = self.ensure(project_name=project_name, hypothesis_id=hypothesis_id)
        if delta == 0:
            return rec
        if delta < 0:
            raise ValueError("[HypothesisRegistry] delta must be >= 0")
        if field_name not in {
            "trials_started",
            "trials_completed",
            "trials_pruned",
            "trials_failed",
            "trials_promoted",
        }:
            raise ValueError(f"[HypothesisRegistry] invalid counter '{field_name}'")

        new_vals = rec.__dict__.copy()
        new_vals[field_name] = int(new_vals[field_name]) + int(delta)
        new_vals["updated_at"] = _now_iso()
        updated = HypothesisRecord(**new_vals)  # type: ignore[arg-type]
        self._items[rec.key()] = updated
        return updated

    def set_best(
        self,
        *,
        project_name: str,
        hypothesis_id: str,
        best_trial_key: str,
        best_primary_metric: Optional[float],
    ) -> HypothesisRecord:
        rec = self.ensure(project_name=project_name, hypothesis_id=hypothesis_id)
        new_vals = rec.__dict__.copy()
        new_vals["best_trial_key"] = require_non_empty(best_trial_key, ctx="best_trial_key")
        new_vals["best_primary_metric"] = best_primary_metric
        new_vals["updated_at"] = _now_iso()
        updated = HypothesisRecord(**new_vals)  # type: ignore[arg-type]
        self._items[rec.key()] = updated
        return updated


# ============================================================
# TrialRegistry (in-memory)
# ============================================================

class TrialRegistry:
    """
    In-memory trial registry.
    Optional backend provides load/save hooks.

    Stores trials keyed by trial_key.
    """

    def __init__(self, *, backend: Optional[RegistryBackend] = None, namespace: str = "trials") -> None:
        self._backend = backend
        self._namespace = require_non_empty(namespace, ctx="namespace")
        self._items: Dict[str, TrialRecord] = {}

    # ---- backend hooks ----

    def load(self) -> None:
        if not self._backend:
            return
        state = self._backend.load(namespace=self._namespace)
        if not state:
            return
        self._items = _decode_trial_state(state)

    def snapshot(self) -> Json:
        return _encode_trial_state(self._items)

    def save(self) -> None:
        if not self._backend:
            return
        self._backend.save(namespace=self._namespace, state=self.snapshot())

    # ---- CRUD ----

    def add(self, record: TrialRecord) -> None:
        key = require_non_empty(record.trial_key, ctx="trial_key")
        if key in self._items:
            raise DuplicateKey(f"[TrialRegistry] duplicate trial_key: {key}")
        self._items[key] = record

    def upsert(self, record: TrialRecord) -> None:
        key = require_non_empty(record.trial_key, ctx="trial_key")
        existing = self._items.get(key)
        if existing is None:
            self._items[key] = record
            return
        # Merge deterministically:
        # - keep created_at
        # - updated_at -> now
        # - fields from `record` win when not None / not empty
        merged = TrialRecord(
            project_name=record.project_name or existing.project_name,
            hypothesis_id=record.hypothesis_id or existing.hypothesis_id,
            trial_key=key,
            created_at=existing.created_at,
            updated_at=_now_iso(),
            trial_id=record.trial_id if record.trial_id is not None else existing.trial_id,
            sota_trial=record.sota_trial if record.sota_trial is not None else existing.sota_trial,
            trial_name=record.trial_name if record.trial_name is not None else existing.trial_name,
            status=record.status or existing.status,
            reason_code=record.reason_code if record.reason_code is not None else existing.reason_code,
            reason_msg=record.reason_msg if record.reason_msg is not None else existing.reason_msg,
            auto_params_path=record.auto_params_path if record.auto_params_path is not None else existing.auto_params_path,
            run_dir=record.run_dir if record.run_dir is not None else existing.run_dir,
            metrics_path=record.metrics_path if record.metrics_path is not None else existing.metrics_path,
            benchmark_dir=record.benchmark_dir if record.benchmark_dir is not None else existing.benchmark_dir,
            params=record.params if record.params else existing.params,
            metrics=record.metrics if record.metrics else existing.metrics,
            objective_values=record.objective_values if record.objective_values else existing.objective_values,
        )
        self._items[key] = merged

    def get(self, *, trial_key: str) -> TrialRecord:
        key = require_non_empty(trial_key, ctx="trial_key")
        rec = self._items.get(key)
        if not rec:
            raise NotFound(f"[TrialRegistry] not found: {key}")
        return rec

    def list(
        self,
        *,
        project_name: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        status: Optional[Union[str, Sequence[str]]] = None,
    ) -> List[TrialRecord]:
        items = list(self._items.values())

        if project_name is not None:
            pn = require_non_empty(project_name, ctx="project_name")
            items = [t for t in items if t.project_name == pn]

        if hypothesis_id is not None:
            hid = require_non_empty(hypothesis_id, ctx="hypothesis_id")
            items = [t for t in items if t.hypothesis_id == hid]

        if status is not None:
            if isinstance(status, str):
                allowed = {status}
            else:
                allowed = {require_non_empty(s, ctx="status") for s in status}
            items = [t for t in items if t.status in allowed]

        # Deterministic ordering: created_at then trial_key
        items.sort(key=lambda t: (t.created_at, t.trial_key))
        return items

    # ---- lifecycle helpers ----

    def update_status(
        self,
        *,
        trial_key: str,
        status: str,
        reason_code: Optional[str] = None,
        reason_msg: Optional[str] = None,
        metrics: Optional[Json] = None,
        objective_values: Optional[Json] = None,
    ) -> TrialRecord:
        rec = self.get(trial_key=trial_key)
        new_vals = rec.__dict__.copy()
        new_vals["status"] = require_non_empty(status, ctx="status")
        new_vals["updated_at"] = _now_iso()
        if reason_code is not None:
            new_vals["reason_code"] = reason_code
        if reason_msg is not None:
            new_vals["reason_msg"] = reason_msg
        if metrics is not None:
            new_vals["metrics"] = dict(metrics)
        if objective_values is not None:
            new_vals["objective_values"] = dict(objective_values)

        updated = TrialRecord(**new_vals)  # type: ignore[arg-type]
        self._items[rec.trial_key] = updated
        return updated

    def attach_artifacts(
        self,
        *,
        trial_key: str,
        auto_params_path: Optional[str] = None,
        run_dir: Optional[str] = None,
        metrics_path: Optional[str] = None,
        benchmark_dir: Optional[str] = None,
    ) -> TrialRecord:
        rec = self.get(trial_key=trial_key)
        new_vals = rec.__dict__.copy()
        new_vals["updated_at"] = _now_iso()
        if auto_params_path is not None:
            new_vals["auto_params_path"] = auto_params_path
        if run_dir is not None:
            new_vals["run_dir"] = run_dir
        if metrics_path is not None:
            new_vals["metrics_path"] = metrics_path
        if benchmark_dir is not None:
            new_vals["benchmark_dir"] = benchmark_dir

        updated = TrialRecord(**new_vals)  # type: ignore[arg-type]
        self._items[rec.trial_key] = updated
        return updated


# ============================================================
# Serialization (pure dict<->dataclass)
# ============================================================

def _encode_hypothesis_state(items: Dict[str, HypothesisRecord]) -> Json:
    return {
        "version": 1,
        "items": {k: _encode_dataclass(v) for k, v in items.items()},
    }


def _decode_hypothesis_state(state: Json) -> Dict[str, HypothesisRecord]:
    items = state.get("items") or {}
    if not isinstance(items, dict):
        raise RegistryError("[HypothesisRegistry] invalid persisted state: items must be mapping")

    out: Dict[str, HypothesisRecord] = {}
    for k, v in items.items():
        if not isinstance(v, dict):
            continue
        out[str(k)] = HypothesisRecord(
            project_name=str(v.get("project_name", "")),
            hypothesis_id=str(v.get("hypothesis_id", "")),
            created_at=str(v.get("created_at", _now_iso())),
            updated_at=str(v.get("updated_at", _now_iso())),
            tags=tuple(v.get("tags", ()) or ()),
            notes=v.get("notes"),
            trials_started=int(v.get("trials_started", 0)),
            trials_completed=int(v.get("trials_completed", 0)),
            trials_pruned=int(v.get("trials_pruned", 0)),
            trials_failed=int(v.get("trials_failed", 0)),
            trials_promoted=int(v.get("trials_promoted", 0)),
            best_trial_key=v.get("best_trial_key"),
            best_primary_metric=v.get("best_primary_metric"),
        )
    return out


def _encode_trial_state(items: Dict[str, TrialRecord]) -> Json:
    return {
        "version": 1,
        "items": {k: _encode_dataclass(v) for k, v in items.items()},
    }


def _decode_trial_state(state: Json) -> Dict[str, TrialRecord]:
    items = state.get("items") or {}
    if not isinstance(items, dict):
        raise RegistryError("[TrialRegistry] invalid persisted state: items must be mapping")

    out: Dict[str, TrialRecord] = {}
    for k, v in items.items():
        if not isinstance(v, dict):
            continue
        out[str(k)] = TrialRecord(
            project_name=str(v.get("project_name", "")),
            hypothesis_id=str(v.get("hypothesis_id", "")),
            trial_key=str(v.get("trial_key", k)),
            created_at=str(v.get("created_at", _now_iso())),
            updated_at=str(v.get("updated_at", _now_iso())),
            trial_id=v.get("trial_id"),
            sota_trial=v.get("sota_trial"),
            trial_name=v.get("trial_name"),
            status=str(v.get("status", "created")),
            reason_code=v.get("reason_code"),
            reason_msg=v.get("reason_msg"),
            auto_params_path=v.get("auto_params_path"),
            run_dir=v.get("run_dir"),
            metrics_path=v.get("metrics_path"),
            benchmark_dir=v.get("benchmark_dir"),
            params=dict(v.get("params", {}) or {}),
            metrics=dict(v.get("metrics", {}) or {}),
            objective_values=dict(v.get("objective_values", {}) or {}),
        )
    return out


def _encode_dataclass(obj: Any) -> Json:
    # dataclass -> dict (no recursion to avoid surprises)
    if hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
        # Ensure JSON-friendly tuples
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d
    raise TypeError(f"[registry] cannot encode object of type {type(obj)}")
