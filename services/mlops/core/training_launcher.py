# services/mlops/core/training_launcher.py
# ============================================================
# Training Launcher (INTERFACE ONLY)
# - Defines the contract between AutoMLOps core and the external training system
# - NO implementation coupling, NO side effects here
# - Phase 3/4: implement adapters for side_channel_core orchestrator/train/test
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

Json = Dict[str, Any]


# ============================================================
# Domain types
# ============================================================

@dataclass(frozen=True)
class TrainRequest:
    """
    One trial training request produced by AutoMLOps.

    compiled: deterministic view from core/context.py (merged config)
    params:   proposed hyperparams / knobs for this trial
    """
    project_name: str
    hypothesis_id: str

    trial_id: Optional[int] = None
    sota_trial: Optional[str] = None

    compiled: Json = field(default_factory=dict)
    params: Json = field(default_factory=dict)

    # optional metadata for routing/adapters (no behavior here)
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrainResult:
    """
    Minimal output contract of a training run.

    status:
      - "completed"
      - "pruned"     (early halted by pruning logic)
      - "failed"     (exception / crash)
      - "skipped"    (backend decided not to run)
    """
    status: str

    # if backend knows these
    trial_id: Optional[int] = None
    sota_trial: Optional[str] = None

    # where training wrote artifacts (paths, not enforced here)
    artifacts: Json = field(default_factory=dict)

    # optional snapshot of final metrics (raw, will be normalized by core/metrics_reader.py)
    metrics: Optional[Json] = None

    # backend error reporting
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None

    # backend-specific extras (clearml task id, gpu, host, etc.)
    meta: Json = field(default_factory=dict)


@dataclass(frozen=True)
class TestRequest:
    """
    Evaluation/benchmark request.

    Use for:
      - validation-only mode
      - benchmarking execution (Phase 3/4)
    """
    project_name: str
    hypothesis_id: str

    trial_id: Optional[int] = None
    sota_trial: Optional[str] = None

    compiled: Json = field(default_factory=dict)
    params: Json = field(default_factory=dict)

    # identify which tests to run (if applicable)
    tests: List[Json] = field(default_factory=list)


@dataclass(frozen=True)
class TestResult:
    status: str  # "completed" | "failed" | "skipped"
    artifacts: Json = field(default_factory=dict)
    metrics: Optional[Json] = None

    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None

    meta: Json = field(default_factory=dict)


# ============================================================
# Protocols (interfaces)
# ============================================================

@runtime_checkable
class TrainingBackend(Protocol):
    """
    Adapter contract for executing training and evaluation.

    Phase 3/4:
      - SideChannelCoreBackend:
          calls services/side_channel_core/src/core/orchestrator.py
          or direct sca/train.py and sca/test.py
      - Future backends: Kubernetes job launcher, Ray, Slurm, etc.
    """

    name: str

    def can_run(self, request: TrainRequest) -> bool:
        """
        Quick predicate: is backend able to run this request?
        Must not throw; return False if not supported.
        """
        ...

    def run_train(self, request: TrainRequest) -> TrainResult:
        """
        Execute one trial training.

        Must be blocking/synchronous in the sense that it returns only when
        the run is finished / pruned / failed. (No async contract here.)
        """
        ...

    def run_test(self, request: TestRequest) -> TestResult:
        """
        Execute evaluation/benchmark tests for a trained candidate.
        """
        ...


@runtime_checkable
class TrainingLauncher(Protocol):
    """
    Higher-level helper that selects a backend, but still interface-only.
    """
    def select_backend(self, request: TrainRequest) -> TrainingBackend:
        ...

    def launch_train(self, request: TrainRequest) -> TrainResult:
        ...

    def launch_test(self, request: TestRequest) -> TestResult:
        ...


# ============================================================
# Optional: minimal pure dispatcher (still no side-effects)
# ============================================================

@dataclass(frozen=True)
class SimpleTrainingLauncher:
    """
    Deterministic backend selection: first backend that can_run().

    Note: backend implementations will have side effects (training), but this
    dispatcher itself is pure control flow.
    """
    backends: Sequence[TrainingBackend]

    def select_backend(self, request: TrainRequest) -> TrainingBackend:
        for b in self.backends:
            try:
                if b.can_run(request):
                    return b
            except Exception:
                # selection must not crash the system
                continue
        raise RuntimeError(
            f"[mlops/core/training_launcher] no backend can_run() for request "
            f"(project={request.project_name}, hypothesis={request.hypothesis_id}, "
            f"trial_id={request.trial_id}, sota_trial={request.sota_trial})"
        )

    def launch_train(self, request: TrainRequest) -> TrainResult:
        backend = self.select_backend(request)
        return backend.run_train(request)

    def launch_test(self, request: TestRequest) -> TestResult:
        # use same backend selection rule (by wrapping TestRequest -> TrainRequest-ish)
        # simplest approach: choose backend by project/hypothesis compatibility
        train_like = TrainRequest(
            project_name=request.project_name,
            hypothesis_id=request.hypothesis_id,
            trial_id=request.trial_id,
            sota_trial=request.sota_trial,
            compiled=request.compiled,
            params=request.params,
        )
        backend = self.select_backend(train_like)
        return backend.run_test(request)


# ============================================================
# Helpers for status enums (optional)
# ============================================================

TRAIN_STATUSES = {"completed", "pruned", "failed", "skipped"}
TEST_STATUSES = {"completed", "failed", "skipped"}
