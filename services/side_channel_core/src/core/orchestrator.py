# services/side_channel_core/src/core/orchestrator.py
"""
orchestrator.py – Side-Channel Analysis Model Train- Tester
==================================================
One-shot orchestrator: train → test for the same model_name


This module runs a one‐shot training and test pass on a trained & validated SCA model.
It combines all the features of train.py and test.py to generate artifacts for the master.yaml and params.yaml decided schema.

Handles: --model-name, --retry-failed, --resume-interrupted, --force, --alias, --model-id

Config source contract
----------------------
Once a model_hash exists in run_registry.json, every repeat of that experiment
(retry-failed, resume-interrupted, force, --model-name, run_map recovery) runs
with the run's paths.params_out snapshot: the exact resolved config it was
trained with, model_hash and model_name frozen. Only when the snapshot does not
exist yet (the run failed before it was written) is the hash re-planned through
the sweep engine, and make_experiment_hash(run_cfg) must then equal the registry
hash. New hashes use the planner's config (first creation). The chosen source is
logged per run as cfg_source=snapshot|replanned|planner.

--alias names a model at its FIRST creation only; an established hash keeps its
registered model_name (identity is frozen) and the alias is logged as ignored.

Bounded retry: after SCC_MAX_FAILED_ATTEMPTS (default 3) FAILED outcomes the
registry entry gets retry_exhausted=true and is no longer retried (FORCE=1
overrides). When only such entries remain the process exits with
EXIT_RETRY_EXHAUSTED (3) so orchestrator.sh stops instead of looping;
EXIT_NOTHING_TO_RUN (4) tells the shell nothing is left within scope.

Without flags, when the planner has no NEW hash left, the scope's FAILED /
INTERRUPTED hashes are selected from run_registry.json and run exactly like
-r / -c (snapshot, retry budget, train_and_test); a failed run is recorded
and re-raised, so the pass exits non-zero. An in_progress hash is not
selected: it may be another orchestrator's live run, and only an explicit -c
resumes it. A hash whose training is DONE but whose test FAILED or was
INTERRUPTED is left to `make test`; the pass raises with that hint instead of
re-training it.

train-cmd -n <model_name|model_hash> runs main(registered_only=True): the same
path for exactly the named registered hashes, except that nothing is planned,
an unknown name or an exhausted hash raises, and a failed run is re-raised
after it is recorded, so the command exits non-zero.

Status updates are keyed by model_hash. On SIGINT/SIGTERM handle_signal()
marks INTERRUPTED only the hashes this process planned or selected
(SELECTED_HASHES) in its own env; entries of other orchestrator processes are
left alone.

orchestrator
  -> picks one hash
  -> train_and_test(hash=A)
       -> train in process, then subprocess test-cmd -n A
          (SCC_SINGLE_HASH=A: the child sweep yields A once, then RETURN)

Author: Shraddha Mishra
"""
import argparse, sys, signal, os
import shutil
import torch
import yaml
import copy
from pathlib import Path
from services.side_channel_core.utils.resolver_mode import _derive_attempt_window
from services.side_channel_core.utils.run_map import load_run_map, project_relative_path
from utils.data_reproducibility import read_params_snapshot
from services.side_channel_core.utils.model_naming import _load_mapping, _save_mapping
from utils.cli_helpers import train_and_test, _plan_scc_models_from_sweep_engine, resolve_registered_run_cfg
from utils.config_loader import get_project_config
from services.side_channel_core.src.sca.dataset_types import check_run_settings
from utils import heartbeat
from utils.path_utils import ensure_runtime_identifiers, resolve_path
from utils.logger import get_logger
from utils.test_registry import FINISHED_TEST_STATUSES, SKIPPED, load_test_registry, save_test_registry
from utils.sweep_engine import save_registry, load_registry, update_status, deep_template_resolve, _attempted_in_session, planned_hashes
from services.side_channel_core.utils.timestamps import now_iso
from utils.lifecycle_outcome import LIFECYCLE_REGISTRY_FIELDS
from utils.error_handler import catch_exceptions
from utils.batch import current_batch, held_by_another_batch, stamp_registry
from utils.test_registry import update_test_status
from utils.run_lifecycle_logger import fail_run_map_entry, complete_run_map_entry, interrupt_run_map_entry
from utils.run_utils import _entry_matches_scope, _select_registry_models, _status, _norm
from utils.device import failure_reason
from services.mlops.utils.zombie_reaper import prepare_trial_runtime

logger = get_logger("Orchestrator")

# Bounded retry (see module docstring). Counted per hash in run_registry.json
# as failed_attempts; retry_exhausted=true marks the entry terminal.
MAX_FAILED_ATTEMPTS = int(os.environ.get("SCC_MAX_FAILED_ATTEMPTS") or 3)
EXIT_RETRY_EXHAUSTED = 3
EXIT_NOTHING_TO_RUN = 4

# Set by main() and _plan_next(). handle_signal() marks only the hashes this
# process planned or selected, in its env: another orchestrator's running
# entries are not ours to interrupt.
PROJECT_ROOT: Path | None = None
RUN_ENV: str | None = None
SELECTED_HASHES: set[str] = set()


def _plan_next(**kwargs) -> list:
    """
    _plan_scc_models_from_sweep_engine(), claiming what it returns: the planner
    marks every hash it yields in_progress, so from then on the hash is this
    process's to interrupt (handle_signal), before train_and_test() starts.
    """
    models = _plan_scc_models_from_sweep_engine(**kwargs)
    SELECTED_HASHES.update(_norm(h) for h, _ in models if h)
    return models


def _retry_exhausted(info) -> bool:
    if not isinstance(info, dict):
        return False
    if info.get("retry_exhausted") is True:
        return True
    try:
        return int(info.get("failed_attempts") or 0) >= MAX_FAILED_ATTEMPTS
    except (TypeError, ValueError):
        return False


def _record_failed_attempt(project_root: Path, model_hash: str, *, reason=None, terminal=False) -> int:
    """
    Count one FAILED outcome for model_hash in run_registry.json.

    Sets retry_exhausted=true once MAX_FAILED_ATTEMPTS is reached, or right away
    for terminal=True (the run's config cannot be reproduced, so retrying is
    pointless). Returns the new failed_attempts count.
    """
    registry = load_registry(project_root)
    entry = registry.get(model_hash)
    if not isinstance(entry, dict):
        return 0
    attempts = int(entry.get("failed_attempts") or 0) + 1
    entry["failed_attempts"] = attempts
    if _status(entry) not in {"failed", "interrupted", "done"} or not str(entry.get("finished_at") or "").strip():
        entry["finished_at"] = now_iso()      # this attempt ended (sweep_engine.update_status)
    entry["status"] = "FAILED"
    entry["status_reason"] = str(reason or "").strip() or "unspecified"
    entry.pop("reason", None)                 # the legacy field; status_reason is the registry's
    if terminal or attempts >= MAX_FAILED_ATTEMPTS:
        entry["retry_exhausted"] = True
        logger.error(
            "[orchestrator] hash=%s name=%s is terminally FAILED after %d failed attempt(s) "
            "(limit SCC_MAX_FAILED_ATTEMPTS=%d). It will not be retried: fix the cause and "
            "delete the registry entry, or re-run with FORCE=1.",
            model_hash, entry.get("model_name"), attempts, MAX_FAILED_ATTEMPTS,
        )
    save_registry(registry, project_root)
    return attempts


def _trial_index(info) -> int:
    try:
        return int(info.get("trial_index") or 0)
    except (TypeError, ValueError):
        return 0


def _force_queue(
    project_root: Path, *, env: str, model_id: str, run_session_id: str, total_trials: int = 0,
    planned: set | None = None, batch=None,
) -> list:
    """
    FORCE=1 retrains the registered runs of the scope (ENV + MODEL_ID) that
    this pass plans (`planned`: sweep_engine.planned_hashes, the current sweep
    lists) before the planner generates a new hash. A run planned under other
    settings is not retrained, nor a DONE run another batch holds (`batch`:
    (BATCH, BATCH_ID); another figure's curve, batch.held_by_another_batch):
    on 2026-10-20 fig20's FORCE retrained five runs of an earlier fig20 sweep
    and fig24's haberdashery_flock. MODELS=<name> retrains any run by name.

    On the first FORCE pass of a session, every scoped run of run_registry.json
    not yet trained in this session is marked queued in run_registry.json and
    test_registry.json, numbered 1..N (trial_index) in its previous
    trial_index order (ties: earlier started_at first), given
    total_remaining_trials = total_trials - trial_index (total_trials: the
    sweep size, else the entry's own) and tagged with the session id; its
    prior status is kept as force_previous_status. Later passes
    take the queue in trial_index order. Returns [(model_hash, info)] still to
    train, first one first; [] once every scoped run is DONE/FAILED in this
    session.
    """
    registry = load_registry(project_root)
    run_map = load_run_map(project_root)
    scoped = _select_registry_models(
        registry, env=env, model_id=model_id, include_models=None, statuses=None,
    )
    scoped = [
        (h, info) for h, info in scoped
        if (planned is None or h in planned)
        # the status it had before FORCE queued it, so every pass of the session keeps the same queue
        and not (_norm(info.get("force_previous_status") or info.get("status")).lower() == "done"
                 and held_by_another_batch(info, batch))
    ]
    pending = [
        (h, info) for h, info in scoped
        if not _attempted_in_session((info, run_map.get(h) or {}), run_session_id)
    ]
    unmarked = sorted(
        (item for item in pending if _norm(item[1].get("run_session_id")) != run_session_id),
        key=lambda item: (_trial_index(item[1]) or float("inf"), _norm(item[1].get("started_at")), item[0]),
    )
    if unmarked:
        start = max(
            (_trial_index(info) for _, info in scoped if _norm(info.get("run_session_id")) == run_session_id),
            default=0,
        )
        tests = load_test_registry(project_root)
        for position, (h, info) in enumerate(unmarked, start=start + 1):
            for entry in (info, tests.get(h)):
                if not isinstance(entry, dict):
                    continue
                entry.setdefault("force_previous_status", entry.get("status"))
                entry["status"] = "queued"
                entry.pop("status_reason", None)
            info["run_session_id"] = run_session_id
            info["trial_index"] = position
            total = int(total_trials or info.get("total_trials") or info.get("trial_total") or 0)
            if total:
                info["total_trials"] = total
                info["total_remaining_trials"] = max(total - position, 0)
        save_registry(registry, project_root)
        save_test_registry(tests, project_root)
        for h, _ in unmarked:
            if h not in tests:
                update_test_status(project_root, model_id=h, new_status="queued")
        logger.info(
            "[orchestrator] FORCE: queued %d registered run(s) for session %s: %s",
            len(unmarked), run_session_id,
            ", ".join(f"{i['trial_index']}={i.get('model_name')}" for _, i in unmarked),
        )
    return sorted(pending, key=lambda item: (_trial_index(item[1]), item[0]))


def _restore_done_status(project_root: Path, model_hash: str, before: dict) -> None:
    """
    Put back the DONE status a run had when this pass started, with that
    attempt's times and key-recovery outcome. -f plans a DONE hash and claims
    it (in_progress, a new attempt: sweep_engine.update_status) before its
    config is resolved; left so, the next -c pass would record a failure on it.
    """
    registry = load_registry(project_root)
    entry = registry.get(model_hash)
    if isinstance(entry, dict) and _status(entry) != "done":
        entry["status"] = before["status"]
        entry.pop("force_previous_status", None)
        for key in ("started_at", "finished_at", *LIFECYCLE_REGISTRY_FIELDS):
            if key in before:
                entry[key] = before[key]
        save_registry(registry, project_root)
    # FORCE queued the test entry too; give back the status it had.
    tests = load_test_registry(project_root)
    test_entry = tests.get(model_hash)
    if isinstance(test_entry, dict) and test_entry.get("force_previous_status"):
        test_entry["status"] = test_entry.pop("force_previous_status")
        save_test_registry(tests, project_root)


def _unqueue_forced(project_root: Path, model_hash: str) -> bool:
    """
    A run FORCE queued that this session never started (stopped first): give
    back the status it had (force_previous_status) in both registries, with
    the trial number, sweep size and session of the training its folder holds
    (its params_out snapshot's runtime block): nothing of it was trained again.
    Marked INTERRUPTED, a DONE run whose folder is intact read as broken, and
    a resume would train it again (fig20's FORCE restart stopped on
    2026-10-20). True when the run was one.
    """
    registry = load_registry(project_root)
    entry = registry.get(model_hash)
    if _status(entry) != "queued" or not entry.get("force_previous_status"):
        return False
    entry["status"] = entry.pop("force_previous_status")
    entry.pop("status_reason", None)
    if _status(entry) != "done":
        entry["status_reason"] = "FORCE queued it again; the orchestrator stopped before it started"
    name = _norm(entry.get("model_name"))
    rel = entry.get("path") or f"artifacts/{entry.get('env') or 'dev'}/generated_models/{name}"
    try:
        snapshot = read_params_snapshot(
            Path(project_root) / project_relative_path(rel, project_root) / "config" / f"{name}_params.yaml"
        )
        runtime = snapshot.get("runtime") if isinstance(snapshot.get("runtime"), dict) else {}
    except (OSError, ValueError):
        runtime = {}                    # no readable snapshot: the status alone is given back
    for key in ("trial_index", "total_trials", "total_remaining_trials", "run_session_id"):
        if runtime.get(key) not in (None, ""):
            entry[key] = runtime[key]
    save_registry(registry, project_root)
    tests = load_test_registry(project_root)
    test_entry = tests.get(model_hash)
    if isinstance(test_entry, dict) and test_entry.get("force_previous_status"):
        test_entry["status"] = test_entry.pop("force_previous_status")
        test_entry.pop("status_reason", None)
        save_test_registry(tests, project_root)
    return True


def _reset_failed_attempts(project_root: Path, model_hash: str) -> None:
    """A completed run starts with a fresh retry budget."""
    registry = load_registry(project_root)
    entry = registry.get(model_hash)
    if isinstance(entry, dict) and ("failed_attempts" in entry or "retry_exhausted" in entry):
        entry.pop("failed_attempts", None)
        entry.pop("retry_exhausted", None)
        save_registry(registry, project_root)


def _apply_alias_to_new_run(
    *,
    cfg: dict,
    project_root: Path,
    model_hash: str,
    info: dict,
    run_cfg: dict,
    alias: str,
    registry_before: dict,
) -> None:
    """
    --alias names a model at its FIRST creation only.

    Identity is frozen once a hash exists in run_registry.json (model_name is
    part of every artifact path and of run_map/test_registry records), so an
    alias never renames an established run; that case is logged and ignored.
    For a hash planned fresh in this pass the alias replaces the generated
    model_name everywhere the planner already wrote it: model_name_map.json,
    run_registry.json, the run_cfg identity and paths, and the still-empty run
    directory. An alias already taken by another hash (e.g. by an earlier pass
    of the same shell loop) is ignored with a warning instead of failing the pass.
    """
    alias = _norm(alias)
    if not alias:
        return
    old_name = _norm(run_cfg.get("model_name") or info.get("model_name"))
    if old_name == alias:
        return

    prior = registry_before.get(model_hash) if isinstance(registry_before, dict) else None
    if isinstance(prior, dict) and _status(prior) not in {"", "new"}:
        logger.warning(
            "[orchestrator] --alias %r ignored: hash=%s is already registered as %r (identity frozen).",
            alias, model_hash, old_name,
        )
        return

    mapping = _load_mapping(project_root)
    registry = load_registry(project_root)
    taken = {h for h, n in mapping.items() if n == alias and h != model_hash}
    taken |= {
        h for h, e in registry.items()
        if isinstance(e, dict) and _norm(e.get("model_name")) == alias and h != model_hash
    }
    if taken:
        logger.warning(
            "[orchestrator] --alias %r ignored: already used by hash(es) %s.", alias, sorted(taken)
        )
        return

    env_val = _norm(run_cfg.get("env")) or _norm(os.environ.get("ENV")) or "dev"
    old_dir = Path(resolve_path(run_cfg["paths"]["artifacts_root"], run_cfg, inject=True))
    # Same template resolution the planner used, with the alias as model_name.
    run_cfg["paths"] = deep_template_resolve(
        copy.deepcopy(cfg["paths"]),
        {
            "env": env_val,
            "model_name": alias,
            "dataset_alias": (run_cfg.get("data") or {}).get("name", "dataset"),
        },
    )
    run_cfg.pop("paths_abs", None)
    run_cfg["model_name"] = alias
    run_cfg["frozen_identity"] = {"model_name": alias, "model_hash": str(model_hash)}
    run_cfg.setdefault("run", {})
    if isinstance(run_cfg["run"], dict):
        run_cfg["run"]["model_name"] = alias
    if isinstance(run_cfg.get("runtime"), dict):
        run_cfg["runtime"]["model_name"] = alias
    new_dir = Path(str(run_cfg["paths"]["artifacts_root"]))

    mapping[str(model_hash)] = alias
    _save_mapping(project_root, mapping)

    entry = registry.get(model_hash)
    if isinstance(entry, dict):
        entry["model_name"] = alias
        entry["path"] = str(new_dir)
        save_registry(registry, project_root)

    # The planner only created <old_dir>/config; drop it while nothing was written there.
    new_abs = Path(resolve_path(str(new_dir), run_cfg, inject=True))
    if old_dir.exists() and old_dir != new_abs and not any(p.is_file() for p in old_dir.rglob("*")):
        shutil.rmtree(old_dir, ignore_errors=True)

    info["model_name"] = alias
    info["run_dir"] = str(new_dir)
    logger.info(
        "[orchestrator] --alias applied at first creation: hash=%s %r -> %r", model_hash, old_name, alias
    )


def _resolve_selected_models(
    models: list,
    *,
    cfg: dict,
    project_root: Path,
    registry_before: dict,
    force: bool,
    run_session_id: str | None,
) -> tuple[list, list]:
    """
    Attach the exact run_cfg to every selected hash and enforce the retry budget.

    Returns (runnable, blocked). Blocked hashes are retry-exhausted (unless
    force) or unrecoverable (no snapshot and not re-plannable, legacy or
    damaged snapshot); the latter are marked FAILED + retry_exhausted so no
    later pass retries them, except a run that was DONE, which stays DONE. Hashes
    already DONE under the current session id are skipped silently (FORCE /
    --model-name re-run each hash once per session).
    """
    runnable: list = []
    blocked: list = []
    for model_hash, info in models:
        if not isinstance(info, dict):
            continue
        if not force and _retry_exhausted(info):
            logger.error(
                "[orchestrator] hash=%s name=%s reached %d failed attempt(s); not retried (FORCE=1 overrides).",
                model_hash, info.get("model_name"), int(info.get("failed_attempts") or MAX_FAILED_ATTEMPTS),
            )
            blocked.append(model_hash)
            continue
        if run_session_id and _status(info) == "done" and _norm(info.get("run_session_id")) == run_session_id:
            logger.info("[orchestrator] hash=%s name=%s already DONE in this session; skipping.",
                        model_hash, info.get("model_name"))
            continue
        try:
            run_cfg, source = resolve_registered_run_cfg(
                cfg=cfg,
                project_root=project_root,
                model_hash=model_hash,
                info=info,
                registry_before=registry_before,
                force=force,
            )
            if source != "snapshot" and isinstance(cfg.get("data"), dict) and "name" not in cfg["data"]:
                raise ValueError(
                    f"hash={model_hash}: no snapshot to train from, and no DATASET to re-plan it with; "
                    f"give DATASET=<one of {list(cfg['data'])}> to re-plan the run"
                )
        except Exception as exc:
            prior = registry_before.get(model_hash) or {}
            if _status(prior) == "done" or _norm(prior.get("force_previous_status")).lower() == "done":
                # A DONE run keeps its trained model and its test even when it
                # cannot be re-run (snapshot written before 2026-09-29, damaged
                # snapshot, edited master.yaml scripts). Recording a failure
                # would hide it from model_judge and graph_manager.
                logger.error(
                    "[orchestrator] hash=%s name=%s is DONE but cannot be re-run: %s "
                    "Its registry entries stay DONE.",
                    model_hash, info.get("model_name"), exc,
                )
                _restore_done_status(project_root, model_hash, dict(prior, status="DONE"))
                blocked.append(model_hash)
                continue
            logger.error("[orchestrator] hash=%s name=%s is unrecoverable: %s",
                         model_hash, info.get("model_name"), exc)
            _record_failed_attempt(
                project_root, model_hash, reason=f"run_cfg_unrecoverable: {exc}", terminal=True
            )
            blocked.append(model_hash)
            continue
        info["run_cfg"] = run_cfg
        info["run_cfg_source"] = source
        logger.info("[orchestrator] hash=%s name=%s cfg_source=%s", model_hash, info.get("model_name"), source)
        runnable.append((model_hash, info))
    return runnable, blocked

def _reconcile_completed_train_test_runs(
    project_root: Path,
    *,
    env: str,
    model_id: str,
) -> None:
    """
    Finalize runs where testing completed but the parent orchestrator exited
    before committing run_registry/run_map DONE.

    This preserves one-shot SCC streaming:

        A(train -> test) -> exit
        B(train -> test) -> exit
    """
    train_registry = load_registry(project_root)
    test_registry = load_test_registry(project_root)
    run_map = load_run_map(project_root)

    active_statuses = {
        "in_progress",
        "running",
        "in_process",
        "queued",
    }

    repaired_hashes: list[str] = []

    for model_hash, train_entry in train_registry.items():
        if not isinstance(train_entry, dict):
            continue

        if str(train_entry.get("env") or "").strip() != str(env).strip():
            continue

        entry_model_id = str(
            train_entry.get("model_id") or ""
        ).strip()

        if model_id and entry_model_id != str(model_id).strip():
            continue

        train_status = str(
            train_entry.get("status") or ""
        ).strip().lower()

        if train_status not in active_statuses:
            continue

        test_entry = test_registry.get(model_hash)

        if not isinstance(test_entry, dict):
            continue

        test_status = str(
            test_entry.get("status") or ""
        ).strip().upper()

        # DONE, or SKIPPED (perform_attack.test.attack_times: 0): no test is left to run.
        if test_status not in FINISHED_TEST_STATUSES:
            continue

        map_entry = (
            run_map.get(model_hash)
            if isinstance(run_map.get(model_hash), dict)
            else {}
        )

        run_cfg = train_entry.get("run_cfg")
        if not isinstance(run_cfg, dict):
            run_cfg = {}

        run_cfg = copy.deepcopy(run_cfg)

        run_cfg.setdefault("project", {})
        if not isinstance(run_cfg["project"], dict):
            run_cfg["project"] = {}

        run_cfg["project"]["root"] = str(project_root)
        run_cfg["env"] = train_entry.get("env") or env
        run_cfg["model_id"] = train_entry.get("model_id") or model_id
        run_cfg["model_hash"] = model_hash
        run_cfg["model_name"] = train_entry.get("model_name")

        runtime = run_cfg.setdefault("runtime", {})
        if not isinstance(runtime, dict):
            runtime = {}
            run_cfg["runtime"] = runtime

        runtime["trial_index"] = (
            train_entry.get("trial_index")
            or map_entry.get("trial_index")
            or runtime.get("trial_index")
        )
        runtime["total_remaining_trials"] = (
            train_entry.get("total_remaining_trials")
            or map_entry.get("total_remaining_trials")
            or runtime.get("total_remaining_trials")
        )
        runtime["trial_total"] = (
            train_entry.get("total_trials")
            or train_entry.get("trial_total")
            or map_entry.get("total_trials")
            or runtime.get("trial_total")
        )
        runtime["total_trials"] = runtime.get("trial_total")
        runtime["run_session_id"] = (
            train_entry.get("run_session_id")
            or map_entry.get("run_session_id")
            or runtime.get("run_session_id")
        )
        runtime["selection_source"] = (
            train_entry.get("selection_source")
            or map_entry.get("selection_source")
            or runtime.get("selection_source")
            or "scc_sweep"
        )

        # Authoritative train lifecycle.
        train_entry["status"] = "DONE"

        complete_run_map_entry(
            cfg=run_cfg,
            model_hash=model_hash,
            model_name=train_entry.get("model_name"),
            model_id=train_entry.get("model_id") or model_id,
            sampled=(
                map_entry.get("sampled")
                or train_entry.get("sampled")
                or run_cfg.get("sampled")
            ),
            mode=(
                train_entry.get("mode")
                or map_entry.get("mode")
                or "scc"
            ),
            run_dir=(
                train_entry.get("path")
                or map_entry.get("run_dir")
                or map_entry.get("path")
            ),
            trial_index=runtime.get("trial_index"),
            total_remaining_trials=runtime.get(
                "total_remaining_trials"
            ),
            run_session_id=runtime.get("run_session_id"),
            total_trials=runtime.get("trial_total"),
            selection_source=runtime.get(
                "selection_source",
                "scc_sweep",
            ),
            verify=True,
        )

        repaired_hashes.append(model_hash)

    if repaired_hashes:
        save_registry(train_registry, project_root)

        logger.warning(
            "[orchestrator] Reconciled completed train->test hashes: %s",
            repaired_hashes,
        )
def handle_signal(sig, frame):
    # The run's heartbeat first: this handler ends the process with sys.exit, so the run's own
    # `except KeyboardInterrupt` never closes it, and a beat left "running" outlives a terminated run.
    heartbeat.finish("interrupted")
    logger.info(
        "Orchestrator received signal %s. Marking this process's unfinished runs %s as INTERRUPTED...",
        sig,
        sorted(SELECTED_HASHES),
    )

    if not PROJECT_ROOT:
        logger.info("PROJECT_ROOT not yet initialized — skip marking.")
        sys.exit(130)

    train_reg = load_registry(PROJECT_ROOT)
    run_map = load_run_map(PROJECT_ROOT)
    try:
        signal_label = f"signal {int(sig)} ({signal.Signals(sig).name})"
    except (TypeError, ValueError):
        signal_label = f"signal {sig}"

    # Only the hashes this process selected, in its own env. Every other
    # active entry belongs to another orchestrator (another MODEL_ID, env or
    # session) that is still running it.
    for h in sorted(SELECTED_HASHES):
        i = train_reg.get(h)
        if not isinstance(i, dict):
            continue
        if RUN_ENV and _norm(i.get("env")) not in {"", RUN_ENV}:
            continue

        status = str(i.get("status") or "").strip().lower()
        if status not in {"in_progress", "queued", "running", "in_process"}:
            continue
        if status == "queued" and _unqueue_forced(PROJECT_ROOT, h):
            logger.info("FORCE had queued %s aka (%s); never started, so it keeps its status %s",
                        h, i.get("model_name"), i.get("force_previous_status"))
            continue

        map_entry = run_map.get(h) if isinstance(run_map.get(h), dict) else {}

        model_name = (
            i.get("model_name")
            or map_entry.get("model_name")
        )
        model_id = (
            i.get("model_id")
            or map_entry.get("model_id")
        )

        cfg = (
            map_entry.get("run_cfg")
            if isinstance(map_entry.get("run_cfg"), dict)
            else i.get("run_cfg")
            if isinstance(i.get("run_cfg"), dict)
            else {}
        )
        cfg = copy.deepcopy(cfg)

        rich_sampled = (
            map_entry.get("sampled")
            or i.get("sampled")
            or cfg.get("sampled")
        )

        rich_sampled_key = (
            map_entry.get("sampled_key")
            or i.get("sampled_key")
            or (cfg.get("runtime") or {}).get("sampled_key")
        )

        cfg.setdefault("project", {})
        if isinstance(cfg["project"], dict):
            cfg["project"]["root"] = str(PROJECT_ROOT)

        cfg["model_hash"] = h
        cfg["model_name"] = model_name
        cfg["model_id"] = model_id
        cfg["env"] = (
            i.get("env")
            or map_entry.get("env")
            or cfg.get("env")
            or "dev"
        )

        if rich_sampled is not None:
            cfg["sampled"] = copy.deepcopy(rich_sampled)

        cfg.setdefault("runtime", {})
        if not isinstance(cfg["runtime"], dict):
            cfg["runtime"] = {}

        runtime = cfg["runtime"]
        runtime["model_hash"] = h
        runtime["model_name"] = model_name
        runtime["model_id"] = model_id
        runtime["trial_index"] = (
            i.get("trial_index")
            or map_entry.get("trial_index")
            or runtime.get("trial_index")
        )
        runtime["total_remaining_trials"] = (
            i.get("total_remaining_trials")
            or map_entry.get("total_remaining_trials")
            or runtime.get("total_remaining_trials")
        )

        runtime["trial_total"] = (
            i.get("total_trials")
            or i.get("trial_total")
            or map_entry.get("total_trials")
            or map_entry.get("trial_total")
            or runtime.get("trial_total")
        )

        runtime["total_trials"] = runtime["trial_total"]
        runtime["run_session_id"] = (
                i.get("run_session_id")
                or map_entry.get("run_session_id")
                or runtime.get("run_session_id")
                or os.environ.get("ORCHESTRATOR_SESSION_ID")
                or os.environ.get("AUTO_SEARCH_SESSION_ID")
            )
        runtime["selection_source"] = (
            i.get("selection_source")
            or map_entry.get("selection_source")
            or runtime.get("selection_source")
            or "scc_sweep"
        )
        runtime["sampled_key"] = rich_sampled_key

        test_registry = load_test_registry(PROJECT_ROOT)
        test_entry = test_registry.get(h)
        test_status = (
            str(test_entry.get("status") or "").strip().upper()
            if isinstance(test_entry, dict)
            else ""
        )

        if test_status in FINISHED_TEST_STATUSES:
            # Train and test already completed (or the test is skipped by config). The parent was interrupted only
            # during post-test finalization, so finalize the whole lifecycle.
            update_status(PROJECT_ROOT, model_id=h, new_status="DONE")

            try:
                complete_run_map_entry(
                    cfg=cfg,
                    model_hash=h,
                    model_name=model_name,
                    model_id=model_id,
                    sampled=rich_sampled,
                    mode=i.get("mode") or map_entry.get("mode") or "scc",
                    run_dir=i.get("path") or map_entry.get("path"),
                    trial_index=runtime.get("trial_index"),
                    total_remaining_trials=runtime.get(
                        "total_remaining_trials"
                    ),
                    total_trials=runtime.get("trial_total"),
                    run_session_id=runtime.get("run_session_id"),
                    selection_source=runtime.get(
                        "selection_source",
                        "scc_sweep",
                    ),
                    verify=True,
                )
            except Exception as e:
                logger.warning(
                    "Failed to finalize completed run_map entry for %s: %s",
                    h,
                    e,
                )

            logger.info(
                "Test was already DONE; finalized %s aka (%s) as DONE.",
                h,
                model_name,
            )
            continue

        update_status(
            PROJECT_ROOT,
            model_id=h,
            new_status="INTERRUPTED",
            reason=f"{signal_label}: the orchestrator stopped while the run was {status}",
        )
        update_test_status(
            PROJECT_ROOT,
            model_id=h,
            new_status="FAILED",
            reason=f"{signal_label}: the run was interrupted before its test finished",
        )

        try:
            interrupt_run_map_entry(
                    cfg=cfg,
                    model_hash=h,
                    model_name=model_name,
                    model_id=model_id,
                    sampled=rich_sampled,
                    mode=i.get("mode") or map_entry.get("mode") or "scc",
                    run_dir=i.get("path") or map_entry.get("path"),
                    trial_index=runtime.get("trial_index"),
                    total_remaining_trials=runtime.get("total_remaining_trials"),
                    total_trials=runtime.get("trial_total"),
                    run_session_id=runtime.get("run_session_id"),
                    selection_source=runtime.get("selection_source", "scc_sweep"),
                    reason=f"signal_{sig}_interrupt",
                    verify=True,
                )
        except Exception as e:
            logger.warning(
                "Failed to mark run_map INTERRUPTED for %s: %s",
                h,
                e,
            )

        logger.info(
            "Marked %s aka (%s) as Train=INTERRUPTED, Test=FAILED, run_map=INTERRUPTED",
            h,
            model_name,
        )

    sys.exit(130)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

@catch_exceptions
def main(
    project=None,
    dataset=None,
    env=None,
    hypothesis=None,
    include_models=None,
    model_id=None,         # NEW: architecture selector
    retry_failed=False,
    resume_interrupted=False,
    force=False,
    alias=None,
    registered_only=False,
    **kwargs,
):
    """
    Sweep-capable orchestrator: Train → Test for one or multiple runs.

    Distinction:
    - model_name: runtime run identity / sweep registry identity
    - model_id:   architecture selector from params.yaml model: [...]

    registered_only=True (train-cmd -n): run exactly the registered hashes
    named by include_models. Nothing is planned; a name outside the scope, a
    retry-exhausted or unrecoverable hash raises; a failed run is recorded
    (FAILED, failed_attempts) and re-raised.
    """
    global PROJECT_ROOT, RUN_ENV

    logger = get_logger(__name__)
    root = Path(__file__).parents[2]
    master = yaml.safe_load((root / "config" / "master.yaml").read_text())
    torch.cuda.empty_cache()

    if project is None:
        p = argparse.ArgumentParser(
            description="Sweep-capable orchestrator: train → test for one or more models"
        )
        p.add_argument("--project", "-p", required=True)
        p.add_argument("--dataset", "-d", required=True)
        p.add_argument("--env", "-e", default="dev")
        p.add_argument("--model-name", "-n", nargs="+")
        p.add_argument("--model-id", type=str, default=None, help="Architecture model_id from params.yaml model: [...]")
        p.add_argument("--retry-failed", "-r", action="store_true")
        p.add_argument("--resume-interrupted", "-c", action="store_true")
        p.add_argument("--force", "-f", action="store_true")
        p.add_argument("--alias", "-a")
        args = p.parse_args()

        project = args.project
        dataset = args.dataset
        env = args.env
        include_models = args.model_name
        model_id = args.model_id
        retry_failed = args.retry_failed
        resume_interrupted = args.resume_interrupted
        force = args.force
        alias = args.alias
    else:
        env = env or master.get("env", "dev")

    # -----------------------------
    # Load config AFTER env is set
    # -----------------------------
    cfg = get_project_config(project, validate_data=True)
    # Resolve architecture model_id from CLI, ENV, or config.
    effective_model_id = (
        model_id
        or os.environ.get("MODEL_ID")
        or cfg.get("model_id")
    )

    if not effective_model_id:
        available = [
            m.get("model_id")
            for m in cfg.get("model", [])
            if isinstance(m, dict) and m.get("model_id")
        ]

        if len(available) == 1:
            effective_model_id = available[0]
        else:
            raise ValueError(
                f"model_id is required because multiple model entries exist: {available}. "
                f"Run with MODEL_ID=MLP1 or --model-id MLP1."
            )

    model_id = str(effective_model_id)
    cfg["model_id"] = model_id
    # Resolve env_path now that cfg exists
    env_cfgs = (cfg.get("project") or {}).get("env_configs") or {}
    # env configs are config files: read from the project's own root (config_root), not the outputs' root
    env_path = Path((cfg.get("project") or {}).get("config_root") or cfg["project"]["root"]) / env_cfgs[env]

    project_root = Path(cfg["project"]["root"])
    PROJECT_ROOT = project_root
    RUN_ENV = _norm(env)
    run_session_id = os.environ.get("ORCHESTRATOR_SESSION_ID") or os.environ.get("AUTO_SEARCH_SESSION_ID")

    entries = list((cfg.get("data") or {}).keys())
    if dataset:
        if dataset not in cfg["data"]:
            raise ValueError(f"Dataset '{dataset}' not found in config; the data entries of params.yaml: {entries}")
        cfg["data"] = cfg["data"][dataset]
        check_run_settings(cfg)   # attack byte, CV and sca.variant against the entry's database type
    elif not registered_only:
        # No default entry: a run never assumes one (the old default "ascad" failed every project
        # without an entry of that name; audit 2026-10-23).
        raise ValueError(f"DATASET is required; the data entries of params.yaml: {entries}")
    # registered_only without DATASET (make reproduce): each run is trained from its own snapshot,
    # whose data entry is the run's (resolve_registered_run_cfg); cfg["data"] stays unselected here.

    # NEW: inject model_id into cfg so downstream utilities can see it if needed
    if model_id is not None:
        cfg["model_id"] = str(model_id).strip()
    # Repair any run whose test completed but whose parent lifecycle did not
    # reach the final DONE write before process termination.
    _reconcile_completed_train_test_runs(
        project_root,
        env=env,
        model_id=model_id,
    )

    # Load registry
    train_reg = load_registry(project_root)
    test_reg = load_test_registry(project_root)
    next_trial_index, total_remaining_trials, total_trials = _derive_attempt_window(
                                                                    project_root,
                                                                    env=env,
                                                                    model_id=model_id,
                                                                    run_session_id=run_session_id,
                                                                )
    # Pick which runtime runs to operate on.
    #
    # CRITICAL:
    # All registry-backed selection must respect env + model_id.
    # If MODEL_ID=CNN1, never select old MLP1 entries.
    models: list[tuple[str | None, dict | None]] = []
    registry_fallback = False
    force_pending: list = []          # FORCE: queued registered runs after this pass's one
    force_queue_hashes: set = set()

    if include_models:
        models = _select_registry_models(
            train_reg,
            env=env,
            model_id=model_id,
            include_models=include_models,
            statuses=None,
        )
        if registered_only:
            matched = {_norm(h) for h, _ in models} | {_norm(i.get("model_name")) for _, i in models}
            missing = [m for m in include_models if _norm(m) not in matched]
            if missing:
                message = (
                    f"{missing} are not registered runs of ENV={env} MODEL_ID={model_id} "
                    f"in {project_root / 'artifacts' / 'run_registry.json'}. Pass a model_name or "
                    "model_hash of that scope (artifacts/model_name_map.json), with the ENV and "
                    "--model-id it was trained with; start new runs with make orchestrate."
                )
                logger.error("[orchestrator] %s", message)
                raise ValueError(message)

    elif force:
        # Registered runs first, in trial_index order; the planner generates
        # new hashes only once all of them were retrained in this session.
        force_pending = (
            _force_queue(
                project_root, env=env, model_id=model_id, run_session_id=run_session_id,
                total_trials=total_trials, planned=planned_hashes(cfg, project_root), batch=current_batch(),
            )
            if run_session_id
            else []
        )
        if force_pending:
            SELECTED_HASHES.update(_norm(h) for h, _ in force_pending)
            force_queue_hashes = {h for h, _ in force_pending}
            models = force_pending[:1]
            force_pending = force_pending[1:]
            train_reg = load_registry(project_root)
        else:
            models = _plan_next(
                cfg=cfg,
                project_root=project_root,
                retry_failed=False,
                resume_interrupted=False,
                force=force,
                limit=1,
            )

    elif retry_failed or resume_interrupted:
        # -r and -c together select both kinds, so a retry-exhausted FAILED
        # hash (blocked below) cannot keep an INTERRUPTED one from resuming.
        statuses: set[str] = set()
        if retry_failed:
            statuses |= {"failed"}
        if resume_interrupted:
            statuses |= {"in_progress", "interrupted", "running", "in_process", "queued"}
        models = _select_registry_models(
            train_reg,
            env=env,
            model_id=model_id,
            include_models=None,
            statuses=statuses,
        )

    else:

        # ---------------------------------------------------------
        # SCC streaming planner mode.
        #
        # IMPORTANT:
        # We DO NOT delegate planning independently to train_cmd
        # and test_cmd because that causes:
        #
        #   ALL TRAIN → ALL TEST
        #
        # Instead orchestrator itself streams:
        #
        #   TRAIN(hash1) → TEST(hash1)
        #   TRAIN(hash2) → TEST(hash2)
        #
        # one executable SCC identity at a time.
        # ---------------------------------------------------------
        models = _plan_next(
                    cfg=cfg,
                    project_root=project_root,
                    retry_failed=retry_failed,
                    resume_interrupted=resume_interrupted,
                    force=force,
                    limit=1,
                )
        if not models:
            # No NEW hash is left in scope. Finish the scope's FAILED /
            # INTERRUPTED hashes the way -r / -c do: selected by model_hash,
            # run from their params_out snapshot under the retry budget, and
            # trained then tested as a pair by train_and_test(). An in_progress
            # hash may be another orchestrator's live run; only -c resumes it.
            registry_fallback = True
            models = _select_registry_models(
                train_reg,
                env=env,
                model_id=model_id,
                include_models=None,
                statuses={"interrupted", "failed"},
            )
    for local_idx, (h, info) in enumerate(models, start=1):
        if not isinstance(info, dict):
            continue

        run_cfg_for_meta = info.get("run_cfg") if isinstance(info.get("run_cfg"), dict) else {}

        runtime = run_cfg_for_meta.setdefault("runtime", {})
        if not isinstance(runtime, dict):
            runtime = {}
            run_cfg_for_meta["runtime"] = runtime

        # Trial number: a FORCE-queued run keeps its queue number; a planned run is numbered by its
        # place in the sweep grid (runtime.sweep_position, utils/sweep_engine.py), the same in every
        # pass; the session counter is only the fallback (a snapshot without a sweep position).
        sweep_position = runtime.get("sweep_position")
        runtime["trial_index"] = (
            _trial_index(info)
            if h in force_queue_hashes and _trial_index(info)
            else int(sweep_position)
            if isinstance(sweep_position, int) and sweep_position > 0
            else int(next_trial_index + local_idx - 1)
        )
        # The size of the sweep planned now comes first. The registry value is
        # the size of an earlier sweep of the same model and is only a fallback.
        runtime["trial_total"] = int(
            runtime.get("trial_total")
            or info.get("total_trials")
            or info.get("trial_total")
            or total_trials
            or 0
        )
        runtime["total_trials"] = int(runtime["trial_total"])
        runtime["total_remaining_trials"] = max(
            int(runtime["trial_total"]) - int(runtime["trial_index"]), 0
        )
        if run_session_id:
            runtime["run_session_id"] = run_session_id
        info["trial_index"] = int(runtime["trial_index"])
        info["total_remaining_trials"] = int(runtime["total_remaining_trials"])
        info["total_trials"] = int(runtime["trial_total"])

        if isinstance(info.get("run_cfg"), dict):
            info["run_cfg"]["runtime"] = runtime
    # If the user requested a registry action, but no registry entries match the
    # current scope, fall back to fresh planner mode instead of accidentally picking
    # unrelated runs. train-cmd -n (registered_only) never plans: it raised above.
    if not models and not registered_only and (include_models or force or retry_failed or resume_interrupted):
        logger.info(
            "[orchestrator] No registry entries matched scope. "
            "Falling back to SCC planner mode instead of fake None model."
        )

        models = _plan_next(
            cfg=cfg,
            project_root=project_root,
            retry_failed=retry_failed,
            resume_interrupted=resume_interrupted,
            force=force,
            limit=1,
        )

        if not models:
            logger.info("[orchestrator] No SCC models planned. Nothing to run.")
            sys.exit(EXIT_NOTHING_TO_RUN)

    # Attach the exact config for every selected hash (snapshot for registered
    # hashes, planner cfg for new ones) and apply the retry budget. This must
    # happen before the entries are marked queued: a re-plan through
    # get_pending_runs() only accepts the entry's real lifecycle status.
    models, blocked = _resolve_selected_models(
        models,
        cfg=cfg,
        project_root=project_root,
        registry_before=train_reg,
        force=force,
        run_session_id=run_session_id,
    )
    if registered_only and blocked:
        message = (
            f"hash(es) {blocked} cannot run: retry budget exhausted "
            f"(failed_attempts >= SCC_MAX_FAILED_ATTEMPTS={MAX_FAILED_ATTEMPTS}; -f overrides) "
            "or their config cannot be reproduced (see the error above)."
        )
        logger.error("[orchestrator] %s", message)
        raise RuntimeError(message)
    if blocked and not models and not force:
        # Every retry candidate is terminal; keep the sweep moving with new
        # hashes only (retry/resume would re-select the exhausted ones).
        logger.warning(
            "[orchestrator] %d hash(es) cannot run (retry budget exhausted or config not "
            "reproducible, see above); planning fresh SCC candidates instead.",
            len(blocked),
        )
        models, _ = _resolve_selected_models(
            _plan_next(
                cfg=cfg,
                project_root=project_root,
                retry_failed=False,
                resume_interrupted=False,
                force=False,
                limit=1,
            ),
            cfg=cfg,
            project_root=project_root,
            registry_before=train_reg,
            force=False,
            run_session_id=run_session_id,
        )
    if blocked and not models:
        logger.error(
            "[orchestrator] Nothing runnable is left: %d hash(es) cannot run: %s "
            "(retry_exhausted=true in run_registry.json, or a config that cannot be "
            "reproduced, see above). Exiting with code %d.",
            len(blocked), blocked, EXIT_RETRY_EXHAUSTED,
        )
        sys.exit(EXIT_RETRY_EXHAUSTED)

    if registry_fallback and not models:
        # Trained but not tested: re-training them to repeat a test is not
        # this command's job, and a test-only pass here used to bypass the
        # snapshot and the retry budget. make test re-tests from the snapshot.
        untested = sorted(
            _norm(i.get("model_name")) or _norm(h)
            for h, i in test_reg.items()
            if _entry_matches_scope(h, i, env=env, model_id=model_id, include_models=None)
            and _status(i) in ("interrupted", "failed")
        )
        if untested:
            message = (
                f"Nothing to train within ENV={env} MODEL_ID={model_id}, but the test of "
                f"{untested} did not finish (test_registry.json). Re-test them with "
                f"make test ENV={env} PROJECT={cfg['project']['name']} MODEL={','.join(untested)}"
            )
            logger.error("[orchestrator] %s", message)
            raise RuntimeError(message)

    # handle_signal() may interrupt these hashes (and the planned ones,
    # _plan_next), and only these.
    SELECTED_HASHES.update(_norm(h) for h, _ in models if h)

    # BATCH / BATCH_ID (make experiment, or make orchestrate BATCH=): every run
    # this pass trains joins the batch; DONE runs in scope join in the planner.
    batch = current_batch()
    if batch is not None:
        stamp_registry(project_root, [h for h, _ in models if h], batch=batch)

    if resume_interrupted or include_models or force or retry_failed or registry_fallback:
        for h, i in models:
            if h and i:
                if not _entry_matches_scope(
                    h,
                    i,
                    env=env,
                    model_id=model_id,
                    include_models=include_models,
                ):
                    logger.warning(
                        "[orchestrator] refusing to mark out-of-scope entry in_progress: "
                        "hash=%s name=%s entry_model_id=%s requested_model_id=%s",
                        h,
                        i.get("model_name"),
                        i.get("model_id"),
                        model_id,
                    )
                    continue

                update_status(project_root, model_id=h, new_status="queued")
                update_test_status(project_root, model_id=h, new_status="queued")

    # logger.info(
    #     "[DEBUG] flags: force=%s retry_failed=%s resume_interrupted=%s include_models=%s model_id=%s",
    #     force, retry_failed, resume_interrupted, include_models, model_id
    # )
    #logger.info(
    #     "[DEBUG] train_reg=%d test_reg=%d selected_models=%d",
    #     len(train_reg), len(test_reg), len(models)
    # )
    total = len(models)
    for idx, (model_hash, info) in enumerate(models, 1):
        # _resolve_selected_models() attached the exact per-hash cfg: the
        # params_out snapshot for registered hashes (identity frozen), the
        # planner cfg for hashes created in this pass.
        run_cfg = copy.deepcopy(info["run_cfg"])
        cfg_source = info.get("run_cfg_source", "planner")

        if alias and idx == 1:
            _apply_alias_to_new_run(
                cfg=cfg,
                project_root=project_root,
                model_hash=model_hash,
                info=info,
                run_cfg=run_cfg,
                alias=alias,
                registry_before=train_reg,
            )

        if info:
            run_cfg["model_name"] = info["model_name"]
            run_cfg["model_hash"] = model_hash
            run_cfg.setdefault("runtime", {})
            run_cfg["runtime"]["model_hash"] = model_hash
            run_cfg["runtime"]["model_name"] = info["model_name"]
            run_cfg["runtime"]["model_id"] = info.get("model_id") or model_id
            run_cfg["runtime"]["trial_index"] = int(
                info.get("trial_index")
                or run_cfg["runtime"].get("trial_index")
                or idx
            )
            run_cfg["runtime"]["total_remaining_trials"] = int(
                info.get("total_remaining_trials")
                or run_cfg["runtime"].get("total_remaining_trials")
                or total_remaining_trials
                or 0
            )
            run_cfg["runtime"]["trial_total"] = int(
                info.get("total_trials")
                or info.get("trial_total")
                or run_cfg["runtime"].get("trial_total")
                or total_trials
                or 0
            )
            run_cfg["runtime"]["total_trials"] = int(run_cfg["runtime"]["trial_total"])
            if run_session_id:
                run_cfg["runtime"]["run_session_id"] = run_session_id
        # NEW: keep model_id on each concrete run cfg
        if model_id is not None:
            run_cfg["model_id"] = str(model_id).strip()

        ensure_runtime_identifiers(run_cfg, force=False)
        run_cfg["paths_abs"] = {
            k: str(resolve_path(v, run_cfg, inject=True)) if isinstance(v, str) else v
            for k, v in (run_cfg.get("paths") or {}).items()
        }

        current_name = run_cfg.get("model_name") or "(new)"

        logger.info(
            f"▶  [{idx}/{total}] Orchestrating: {current_name} in [{env}] mode. "
            f"model_id={run_cfg.get('model_id')} hash={model_hash} cfg_source={cfg_source}"
        )
        if env_path:
            logger.info(f"Applied env-specific config overrides from: {env_path}")
        else:
            logger.info("No env-specific override file registered for env=%s", env)

        # Leftovers of the previous experiment (orphaned loader workers, stale
        # GPU contexts, open HDF5 handles, a hot GPU) must not reach this one.
        prepare_trial_runtime(logger=logger)

        try:
            if model_hash:
                update_status(project_root, model_id=model_hash, new_status="in_progress")
                update_test_status(project_root, model_id=model_hash, new_status="queued")
                train_reg_now = load_registry(project_root)
                entry = train_reg_now.get(model_hash)

                if isinstance(entry, dict):
                    runtime = run_cfg.get("runtime") or {}

                    entry["trial_index"] = int(runtime.get("trial_index") or idx)
                    entry["total_remaining_trials"] = int(
                        runtime.get("total_remaining_trials")
                        or total_remaining_trials
                        or 0
                    )
                    entry["total_trials"] = int(
                        runtime.get("trial_total")
                        or runtime.get("total_trials")
                        or total_trials
                        or 0
                    )
                    entry["selection_source"] = runtime.get("selection_source", "scc_sweep")
                    if run_session_id:
                        entry["run_session_id"] = run_session_id
                    # Keep these stable too.
                    entry["model_hash"] = model_hash
                    entry["model_name"] = run_cfg.get("model_name") or entry.get("model_name")
                    entry["model_id"] = run_cfg.get("model_id") or entry.get("model_id")
                    entry["env"] = run_cfg.get("env") or env or entry.get("env")

                    save_registry(train_reg_now, project_root)

            # TraceFlow Monitor: the run beats until it is done, failed or interrupted.
            heartbeat.begin(
                run_cfg, project_root, env,
                batches=list(((load_registry(project_root).get(model_hash) or {}) if model_hash else {}).get("batches") or []),
                trial=(run_cfg.get("runtime") or {}).get("trial_index") or idx,
                trials=(run_cfg.get("runtime") or {}).get("trial_total") or total,
            )
            train_and_test(
                run_cfg,
                env,
                retry_failed=retry_failed,
                resume_interrupted=resume_interrupted,
                force=force,
                model_id=run_cfg.get("model_id"),   # NEW
            )
            # The full model lifecycle has now completed:
            #     training -> selected validation snapshot -> testing
            #
            # Reporting is deliberately non-fatal. Missing or malformed report artifacts
            # must never convert an otherwise successful experiment into a failed run.

            if info:
                update_status(
                    project_root,
                    model_id=model_hash,
                    new_status="DONE",
                )
                # A test skipped by config (perform_attack.test.attack_times: 0)
                # stays SKIPPED with its reason; it was never run.
                tested = load_test_registry(project_root).get(model_hash) or {}
                if str(tested.get("status") or "").strip().upper() != SKIPPED:
                    update_test_status(
                        project_root,
                        model_id=model_hash,
                        new_status="DONE",
                    )
                _reset_failed_attempts(project_root, model_hash)

                runtime = run_cfg.get("runtime") or {}

                complete_run_map_entry(
                    cfg=run_cfg,
                    model_hash=model_hash,
                    model_name=current_name,
                    model_id=run_cfg.get("model_id"),
                    sampled=info.get("sampled") or run_cfg.get("sampled"),
                    mode="scc",
                    run_dir=str(
                        resolve_path(
                            run_cfg["paths"]["artifacts_root"],
                            run_cfg,
                            inject=True,
                        )
                    ),
                    trial_index=runtime.get("trial_index"),
                    total_remaining_trials=runtime.get(
                        "total_remaining_trials"
                    ),
                    run_session_id=runtime.get("run_session_id"),
                    total_trials=runtime.get("trial_total"),
                    selection_source=runtime.get(
                        "selection_source",
                        "scc_sweep",
                    ),
                    verify=True,
                )

            heartbeat.finish("done")
            logger.info(
                "Finished [%s/%s] Train → Test: hash=%s name=%s. "
                "All lifecycle records are DONE.",
                idx,
                total,
                model_hash,
                current_name,
            )
        except KeyboardInterrupt:
            heartbeat.finish("interrupted")
            logger.exception("Orchestrator manually interrupted!")
            if info:
                update_status(project_root, model_id=model_hash, new_status="INTERRUPTED",
                              reason="keyboard_interrupt: stopped by the user during train/test")
                update_test_status(project_root, model_id=model_hash, new_status="FAILED",
                                   reason="keyboard_interrupt: stopped by the user before the test finished")
                try:
                    fail_run_map_entry(
                        cfg=run_cfg,
                        model_hash=model_hash,
                        model_name=current_name,
                        model_id=run_cfg.get("model_id"),
                        sampled=info.get("sampled") or run_cfg.get("sampled"),
                        mode="scc",
                        run_dir=str(resolve_path(run_cfg["paths"]["artifacts_root"], run_cfg, inject=True)),
                        trial_index=(run_cfg.get("runtime") or {}).get("trial_index"),
                        total_remaining_trials=(run_cfg.get("runtime") or {}).get("total_remaining_trials"),
                        run_session_id=(run_cfg.get("runtime") or {}).get("run_session_id"),
                        total_trials=(run_cfg.get("runtime") or {}).get("trial_total"),
                        selection_source=(run_cfg.get("runtime") or {}).get("selection_source", "scc_sweep"),
                        reason="keyboard_interrupt",
                    )
                except Exception as e:
                    logger.warning("Failed to mark run_map INTERRUPTED for %s: %s", model_hash, e)
                logger.info(f"Interrupted Experiment {idx}/{total}: {model_hash} aka ({current_name})")

            for remaining_hash, remaining_info in models[idx:] + force_pending:
                if remaining_info:
                    update_status(project_root, model_id=remaining_hash, new_status="INTERRUPTED",
                                  reason=f"keyboard_interrupt_unattempted: not started, the pass stopped at {model_hash}")
                    update_test_status(project_root, model_id=remaining_hash, new_status="FAILED",
                                       reason=f"keyboard_interrupt_unattempted: not trained, the pass stopped at {model_hash}")

                    try:
                        remaining_cfg = remaining_info.get("run_cfg") if isinstance(remaining_info, dict) else {}
                        if not isinstance(remaining_cfg, dict):
                            remaining_cfg = {}

                        fail_run_map_entry(
                            cfg=remaining_cfg or run_cfg,
                            model_hash=remaining_hash,
                            model_name=remaining_info.get("model_name"),
                            model_id=remaining_info.get("model_id") or model_id,
                            sampled=remaining_info.get("sampled"),
                            mode="scc",
                            run_dir=remaining_info.get("path"),
                            trial_index=remaining_info.get("trial_index"),
                            total_remaining_trials=remaining_info.get("total_remaining_trials"),
                            run_session_id=(
                                (remaining_cfg.get("runtime") or {}).get("run_session_id")
                                if isinstance(remaining_cfg, dict)
                                else run_session_id
                            ),
                            total_trials=remaining_info.get("total_trials"),
                            selection_source=remaining_info.get("selection_source", "scc_sweep"),
                            reason="keyboard_interrupt_unattempted",
                        )
                    except Exception as e:
                        logger.warning("Failed to mark queued run_map INTERRUPTED for %s: %s", remaining_hash, e)
                    logger.info(
                        f" Marking queued Experiment {remaining_hash} aka "
                        f"({remaining_info['model_name']}) as INTERRUPTED"
                    )
            sys.exit(1)

        except Exception as exc:
            heartbeat.finish("failed")
            logger.exception("Unhandled exception in orchestrator loop.")
            if info:
                failure = failure_reason(exc)   # utils/device.py: "GPU FAILED: ..." for a lost GPU
                update_status(project_root, model_id=model_hash, new_status="FAILED", reason=failure)
                update_test_status(project_root, model_id=model_hash, new_status="FAILED",
                                   reason=f"train/test failed: {failure}")
                _record_failed_attempt(project_root, model_hash, reason=failure)

            save_registry(load_registry(project_root), project_root)
            logger.info(f"Failed Experiment {idx}/{total}: {model_hash} aka ({current_name})")

            for remaining_hash, remaining_info in models[idx:]:
                if remaining_info:
                    # Do not mark unattempted queued experiments as FAILED.
                    # They were not actually attempted. Keep them resumable.
                    update_status(project_root, model_id=remaining_hash, new_status="INTERRUPTED",
                                  reason=f"unattempted: not started, experiment {model_hash} failed first")
                    update_test_status(project_root, model_id=remaining_hash, new_status="FAILED",
                                       reason=f"unattempted: not trained, experiment {model_hash} failed first")
                    logger.info(
                        f" Marking unattempted queued Experiment {remaining_hash} aka "
                        f"({remaining_info['model_name']}) as INTERRUPTED"
                    )
            if registered_only or registry_fallback:
                # train-cmd -n and the no-flag registry pass: the failure is
                # recorded above (FAILED, failed_attempts); the command fails.
                raise
            continue

    if not models:
        logger.info("[orchestrator] Nothing to run within scope env=%s model_id=%s models=%s.",
                    env, model_id, include_models or "<all>")
        sys.exit(EXIT_NOTHING_TO_RUN)

    sys.exit(0)
