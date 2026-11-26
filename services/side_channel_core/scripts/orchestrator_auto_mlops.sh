#!/usr/bin/env bash
# services/side_channel_core/scripts/orchestrator_auto_mlops.sh
#
# Unified orchestration switch:
#   1) auto_search  -> services.mlops.core.optuna_lite_entry
#   2) auto_mlops   -> services.mlops.core.auto_mlops_runner_entry
#   3) manual SCC   -> services/side_channel_core/scripts/orchestrator.sh
#
# Hard rules:
# - Do NOT parse YAML manually in bash.
# - Probe master.yaml through Python using services.mlops.utils.auto_config.MLOpsConfig.
# - Keep SCC manual sweep/orchestrate path separate from Optuna search control flow.
#
# Forwarded contract:
#   PROJECT DATASET  ALIAS MODELS ENV FORCE MODEL_ID
#
# Notes:
# - FORCE is SCC semantics only (rerun/retrain). It does NOT force auto_search/auto_mlops dispatch.
# - SEARCH=1 or AUTO_SEARCH_FORCE=1 can force auto_search path.
# - MLOPS=1 or AUTO_MLOPS_FORCE=1 can force auto_mlops path.
# - If both auto_search and auto_mlops are enabled in master.yaml for a project,
#   auto_search wins.


set -euo pipefail

log() { echo "[orchestrator_auto_mlops] $*" >&2; }
die() { log "FATAL: $*"; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCC_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${SCC_MASTER_YAML:=${SCC_ROOT}/config/master.yaml}"

: "${AUTO_SEARCH_FORCE:=${AUTO_SEARCH_FORCE:-}}"
: "${SEARCH:=${SEARCH:-0}}"

: "${AUTO_MLOPS_FORCE:=${AUTO_MLOPS_FORCE:-}}"
: "${MLOPS:=${MLOPS:-0}}"

: "${FORCE:=${FORCE:-0}}"

: "${PROJECT:=${PROJECT:-}}"
: "${DATASET:=${DATASET:-}}"
: "${ALIAS:=${ALIAS:-}}"
: "${MODELS:=${MODELS:-}}"
: "${MODEL_ID:=${MODEL_ID:-}}"
: "${ENV:=${ENV:-dev}}"
: "${HY:=${HY:-}}"
: "${PROJECTS_ROOT:=${PROJECTS_ROOT:-/app/projects}}"
: "${RERUN_EXPERIMENT:=${RERUN_EXPERIMENT:-}}"
: "${TEST_MODEL:=${TEST_MODEL:-}}"
: "${REPRODUCE_MODELS:=${REPRODUCE_MODELS:-}}"
: "${TEST_CONFIG:=${TEST_CONFIG:-}}"
: "${TEST_EXTERNAL:=${TEST_EXTERNAL:-}}"
: "${TEST_BUDGET:=${TEST_BUDGET:-}}"
: "${REPEAT:=${REPEAT:-}}"

export PROJECT DATASET ALIAS MODELS MODEL_ID ENV HY FORCE
export PROJECTS_ROOT SCC_MASTER_YAML RERUN_EXPERIMENT REPEAT

MANUAL_ORCH="${SCRIPT_DIR}/orchestrator.sh"
RERUN_ORCH="${SCRIPT_DIR}/orchestrator_rerun.sh"

: "${AUTO_SEARCH_RUNNER_MODULE:=services.mlops.core.optuna_lite_entry}"
: "${AUTO_MLOPS_RUNNER_MODULE:=services.mlops.core.auto_mlops_runner_entry}"

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="/app"
fi

debug_snapshot() {
  log "---- debug snapshot ----"
  log "SCRIPT_DIR=${SCRIPT_DIR}"
  log "SCC_ROOT=${SCC_ROOT}"
  log "SCC_MASTER_YAML=${SCC_MASTER_YAML}"
  log "AUTO_SEARCH_RUNNER_MODULE=${AUTO_SEARCH_RUNNER_MODULE}"
  log "AUTO_MLOPS_RUNNER_MODULE=${AUTO_MLOPS_RUNNER_MODULE}"
  log "PROJECT=${PROJECT:-<unset>} DATASET=${DATASET:-<unset>} ENV=${ENV:-<unset>}"
  log "ALIAS=${ALIAS:-<unset>} MODELS=${MODELS:-<unset>} MODEL_ID=${MODEL_ID:-<unset>}"
  log "SEARCH=${SEARCH:-<unset>} AUTO_SEARCH_FORCE=${AUTO_SEARCH_FORCE:-<unset>}"
  log "MLOPS=${MLOPS:-<unset>} AUTO_MLOPS_FORCE=${AUTO_MLOPS_FORCE:-<unset>}"
  log "FORCE=${FORCE:-<unset>} RERUN_EXPERIMENT=${RERUN_EXPERIMENT:-<unset>}"
  log "PROJECTS_ROOT=${PROJECTS_ROOT}"
  log "PYTHONPATH=${PYTHONPATH:-<unset>}"
  log "pwd=$(pwd)"
  log "python=$(command -v python || true)"
  log "---- end debug snapshot ----"
}
if [[ "${FORCE}" = "1" ]]; then
  export AUTO_SEARCH_FORCE_RUN_ALL=1
else
  export AUTO_SEARCH_FORCE_RUN_ALL=0
fi
AUTO_CHILD_PID=""

repair_auto_search_registry() {
  local reason="${1:-startup_repair}"

  if [[ -z "${PROJECT_ROOT:-}" ]]; then
    log "registry repair skipped: PROJECT_ROOT is unset"
    return 0
  fi

  python -m services.mlops.utils.registry_repair \
    --project-root "${PROJECT_ROOT}" \
    --env "${ENV:-dev}" \
    --reason "${reason}" || true
}

stop_auto_child() {
  if [[ -z "${AUTO_CHILD_PID:-}" ]]; then
    return 0
  fi

  if ! kill -0 "${AUTO_CHILD_PID}" 2>/dev/null; then
    return 0
  fi

  log "forwarding SIGINT to auto child PID=${AUTO_CHILD_PID}"
  kill -SIGINT "${AUTO_CHILD_PID}" 2>/dev/null || true

  # Give Python a very short chance to exit cleanly.
  sleep 0.3

  if kill -0 "${AUTO_CHILD_PID}" 2>/dev/null; then
    log "child still alive; forwarding SIGTERM to PID=${AUTO_CHILD_PID}"
    kill -SIGTERM "${AUTO_CHILD_PID}" 2>/dev/null || true
  fi

  sleep 0.5

  if kill -0 "${AUTO_CHILD_PID}" 2>/dev/null; then
    log "child still alive; forwarding SIGKILL to PID=${AUTO_CHILD_PID}"
    kill -SIGKILL "${AUTO_CHILD_PID}" 2>/dev/null || true
  fi

  wait "${AUTO_CHILD_PID}" 2>/dev/null || true
}

on_interrupt() {
  local sig="${1:-INT}"

  log "interrupt caught by orchestrator_auto_mlops.sh: ${sig}"

  # Mark registry immediately. This is the fastest visible state update.
  repair_auto_search_registry "interrupt_${sig}" || true

  # Then stop the Python child.
  stop_auto_child || true

  # Repair once more after child is stopped, in case it rewrote status.
  repair_auto_search_registry "interrupt_${sig}_post_child_stop" || true

  exit 130
}
trap 'on_interrupt SIGINT' SIGINT
trap 'on_interrupt SIGTERM' SIGTERM
on_exit() {
  rc=$?
  if [[ $rc -ne 0 ]]; then
    debug_snapshot
  fi
  exit $rc
}
trap on_exit EXIT

[[ -f "${MANUAL_ORCH}" ]] || die "manual orchestrator not found at ${MANUAL_ORCH}"
chmod +x "${MANUAL_ORCH}" 2>/dev/null || true

resolve_contract() {
  local out

  out="$(
    python - <<'PY'
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception as e:
    print(f"ERR|missing_yaml|{e}", file=sys.stderr)
    raise SystemExit(2)

master = Path(os.environ.get("SCC_MASTER_YAML", "/app/services/side_channel_core/config/master.yaml"))
projects_root = Path(os.environ.get("PROJECTS_ROOT", "/app/projects"))
req_project = (os.environ.get("PROJECT") or "").strip()
req_dataset = (os.environ.get("DATASET") or "").strip()

if not master.exists():
    print("ERR|master_missing|" + str(master), file=sys.stderr)
    raise SystemExit(2)

data = yaml.safe_load(master.read_text()) or {}
projects = data.get("project") or []
if isinstance(projects, dict):
    projects = [projects]
if not isinstance(projects, list):
    projects = []

def pick_default_project():
    dp = data.get("default_project")
    if isinstance(dp, str) and dp.strip():
        return dp.strip()

    for pr in projects:
        if isinstance(pr, dict):
            name = pr.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    return ""

def find_project_block(name):
    for pr in projects:
        if isinstance(pr, dict) and pr.get("name") == name:
            return pr
    return None

project = req_project or pick_default_project()
if not project:
    print("ERR|no_project|Set PROJECT or define default_project in master.yaml", file=sys.stderr)
    raise SystemExit(2)

block = find_project_block(project)
if not isinstance(block, dict):
    print(f"OK|{project}||", end="")
    raise SystemExit(0)

root_rel = str(block.get("root") or project).strip().rstrip("/")
params_rel = str(block.get("params") or "config/params.yaml").strip()

project_root = (projects_root / root_rel).resolve()
params_path = (project_root / params_rel).resolve()

if not params_path.exists():
    print(f"ERR|params_missing|{params_path}", file=sys.stderr)
    raise SystemExit(2)

params = yaml.safe_load(params_path.read_text()) or {}
data_block = params.get("data")
names = []

if isinstance(data_block, list):
    for ds in data_block:
        if isinstance(ds, dict):
            name = ds.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
elif isinstance(data_block, dict):
    names = [k.strip() for k in data_block.keys() if isinstance(k, str) and k.strip()]

deduped = []
for name in names:
    if name not in deduped:
        deduped.append(name)

dataset = req_dataset
if not dataset:
    if len(deduped) == 1:
        dataset = deduped[0]
    elif len(deduped) == 0:
        print("ERR|no_datasets|No datasets found in params.yaml data:", file=sys.stderr)
        raise SystemExit(2)
    else:
        print("ERR|dataset_required|" + ",".join(deduped), file=sys.stderr)
        raise SystemExit(2)

print(f"OK|{project}|{dataset}|{project_root}")
PY
  )" || return 2

  if echo "$out" | grep -q "^ERR|"; then
    case "$out" in
      ERR\|dataset_required\|*)
        local allowed="${out#ERR|dataset_required|}"
        die "DATASET is required for PROJECT=${PROJECT:-<unset>}. Allowed: ${allowed}"
        ;;
      ERR\|params_missing\|*)
        die "params.yaml not found: ${out#ERR|params_missing|}"
        ;;
      ERR\|master_missing\|*)
        die "master.yaml not found: ${out#ERR|master_missing|}"
        ;;
      *)
        die "contract resolution failed: ${out}"
        ;;
    esac
  fi

  local payload="${out#OK|}"
  local p="${payload%%|*}"
  payload="${payload#*|}"
  local d="${payload%%|*}"
  payload="${payload#*|}"
  local r="${payload}"

  export PROJECT="$p"
  export DATASET="$d"
  export PROJECT_ROOT="$r"

  log "contract → PROJECT=${PROJECT} DATASET=${DATASET} ENV=${ENV}"
  log "contract → ALIAS=${ALIAS:-<unset>} MODELS=${MODELS:-<unset>} MODEL_ID=${MODEL_ID:-<unset>}"
  log "contract → PROJECT_ROOT=${PROJECT_ROOT}"
}

plugin_import_ok() {
  local module="$1"

  python - <<PY
import importlib
import sys

module = "${module}"

try:
    importlib.import_module(module)
except Exception as e:
    print(
        f"[orchestrator_auto_mlops] plugin import FAIL: {module} "
        f"({e.__class__.__name__}: {e})",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(f"[orchestrator_auto_mlops] plugin import OK: {module}", file=sys.stderr)
PY
}

run_manual() {
  resolve_contract || die "contract resolution failed"

  log "EXECUTION: MANUAL → orchestrator=${MANUAL_ORCH}"
  exec "${MANUAL_ORCH}"
}

run_auto_search() {
  resolve_contract || die "contract resolution failed"

  log "mode=auto_search → checking plugin import"
  if plugin_import_ok "${AUTO_SEARCH_RUNNER_MODULE}"; then
    log "EXECUTION: AUTO_SEARCH → module=${AUTO_SEARCH_RUNNER_MODULE}"
    log "AUTO_SEARCH selector → MODEL_ID=${MODEL_ID:-<unset>}"

    # Repair stale entries from previous dirty shutdown before starting new work.
    repair_auto_search_registry "startup_auto_search"

    python -u -m "${AUTO_SEARCH_RUNNER_MODULE}" \
    --project "${PROJECT}" \
    --master-yaml "${SCC_MASTER_YAML}" \
    --mode orchestrate &

    AUTO_CHILD_PID=$!
    export AUTO_CHILD_PID

    set +e
    wait "${AUTO_CHILD_PID}"
    rc=$?
    set -e

    AUTO_CHILD_PID=""

    # If child exited due to signal or abnormal termination, repair stale entries.
    if [[ "${rc}" -eq 130 || "${rc}" -eq 143 || "${rc}" -ne 0 ]]; then
    repair_auto_search_registry "child_exit_${rc}"
    fi

    log "mode=auto_search → runner exit code=${rc}"
    exit "${rc}"
  fi

  die "auto_search requested/enabled but module could not be imported: ${AUTO_SEARCH_RUNNER_MODULE}"
}

run_auto_mlops() {
  resolve_contract || die "contract resolution failed"

  log "mode=auto_mlops → checking plugin import"
  if plugin_import_ok "${AUTO_MLOPS_RUNNER_MODULE}"; then
    log "EXECUTION: AUTO_MLOPS → module=${AUTO_MLOPS_RUNNER_MODULE}"

    repair_auto_search_registry "startup_auto_mlops"

    python -u -m "${AUTO_MLOPS_RUNNER_MODULE}" \
    --master-yaml "${SCC_MASTER_YAML}" &

    AUTO_CHILD_PID=$!
    export AUTO_CHILD_PID

    set +e
    wait "${AUTO_CHILD_PID}"
    rc=$?
    set -e

    AUTO_CHILD_PID=""

    if [[ "${rc}" -eq 130 || "${rc}" -eq 143 || "${rc}" -ne 0 ]]; then
    repair_auto_search_registry "auto_mlops_child_exit_${rc}"
    fi

    log "mode=auto_mlops → runner exit code=${rc}"
    exit "${rc}"
  fi

  log "mode=auto_mlops → plugin missing → fallback=manual"
  run_manual
}

probe_master_mode() {
  python - <<'PY'
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception as e:
    print(f"[AUTO_PROBE] yaml import failed: {e.__class__.__name__}: {e}", file=sys.stderr)
    print("manual||")
    raise SystemExit(0)

master = Path(os.environ.get("SCC_MASTER_YAML", "/app/services/side_channel_core/config/master.yaml"))
requested = (os.environ.get("PROJECT") or "").strip()

if not master.exists():
    print(f"[AUTO_PROBE] master.yaml missing: {master}", file=sys.stderr)
    print("manual||")
    raise SystemExit(0)

try:
    data = yaml.safe_load(master.read_text()) or {}
except Exception as e:
    print(f"[AUTO_PROBE] failed to parse master.yaml: {e.__class__.__name__}: {e}", file=sys.stderr)
    print("manual|" + (requested or "") + "|")
    raise SystemExit(0)

projects = data.get("project") or []
if isinstance(projects, dict):
    projects = [projects]
if not isinstance(projects, list):
    projects = []

default_project = data.get("default_project")
if not isinstance(default_project, str):
    default_project = ""

def normalize_flag(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(v, int):
        return v != 0
    return False

def find_project(name):
    for pr in projects:
        if isinstance(pr, dict) and pr.get("name") == name:
            return pr
    return None

def mode_for_project(pr):
    auto_search = pr.get("auto_search") or {}
    auto_mlops = pr.get("auto_mlops") or {}

    search_enabled = isinstance(auto_search, dict) and normalize_flag(auto_search.get("enabled", False))
    mlops_enabled = isinstance(auto_mlops, dict) and normalize_flag(auto_mlops.get("enabled", False))

    if search_enabled:
        return "search"
    if mlops_enabled:
        return "auto"
    return "manual"

if requested:
    pr = find_project(requested)
    if pr is None:
        print(f"[AUTO_PROBE] requested project not found: {requested}", file=sys.stderr)
        print("manual|" + requested + "|")
        raise SystemExit(0)

    print(mode_for_project(pr) + "|" + requested + "|")
    raise SystemExit(0)

if default_project:
    pr = find_project(default_project)
    if pr is not None:
        print(mode_for_project(pr) + "|" + default_project + "|")
        raise SystemExit(0)

for pr in projects:
    if not isinstance(pr, dict):
        continue

    mode = mode_for_project(pr)
    if mode in {"search", "auto"}:
        print(mode + "|" + str(pr.get("name") or "") + "|")
        raise SystemExit(0)

for pr in projects:
    if isinstance(pr, dict) and isinstance(pr.get("name"), str) and pr["name"].strip():
        print("manual|" + pr["name"].strip() + "|")
        raise SystemExit(0)

print("manual||")
PY
}

# ---------------------------------------------------------
# Single dispatch authority
#
# Priority:
#   0. TEST_MODEL          (make test: one-off test request)
#   0b. REPRODUCE_MODELS   (make reproduce: runs registered from shipped snapshots)
#   1. RERUN_EXPERIMENT
#   2. SEARCH / AUTO_SEARCH_FORCE
#   3. MLOPS / AUTO_MLOPS_FORCE
#   4. master.yaml project mode
#
# Important:
# - master.yaml owns normal mode selection.
# - MODEL_ID is a model selector only.
# - MODEL_ID must NOT force manual mode.
# - MODELS is a manual/rerun filter, not a mode selector.
# ---------------------------------------------------------

if [[ -n "${TEST_MODEL}" ]]; then
  # One-off test request: core/cli.py test-cmd --test-request re-tests one
  # model from its training snapshot + the project's test_config.yaml
  # (src/sca/test_request.py). Every test is appended beside the recorded test;
  # TEST_EXTERNAL=1 attacks with test_config.yaml's settings, TEST_BUDGET=1 with the data entry's budget.
  [[ -n "${PROJECT}" ]] || die "TEST_MODEL needs PROJECT"
  export PYTHONPATH="${WORKSPACE_ROOT:-/app}:${PYTHONPATH:-}"
  PYTHON_BIN="python"
  [[ -x /venv/bin/python ]] && PYTHON_BIN="/venv/bin/python"
  test_args=(test-cmd --project "${PROJECT}" --env "${ENV}" --test-request -n "${TEST_MODEL}")
  [[ -n "${TEST_CONFIG}" ]] && test_args+=(--test-config "${TEST_CONFIG}")
  # make test EXTERNAL=1: test_config.yaml's attack settings, beside the recorded test
  [[ "${TEST_EXTERNAL}" == "1" ]] && test_args+=(--external)
  [[ "${TEST_BUDGET}" == "1" ]] && test_args+=(--budget)
  log "DECISION: TEST REQUEST → model=${TEST_MODEL} project=${PROJECT} config=${TEST_CONFIG:-<extra_configs.test>} external=${TEST_EXTERNAL:-0} budget=${TEST_BUDGET:-0}"
  cd "${SCC_ROOT}/src"
  exec "${PYTHON_BIN}" -m core.cli "${test_args[@]}"
fi

if [[ -n "${REPRODUCE_MODELS}" ]]; then
  # make reproduce: the runs utils/reproduce.py registered from their params_out snapshots
  # (shipped alone). train-cmd -n trains exactly these registered runs from their snapshots,
  # then tests them; it never plans from params.yaml. Name, hash and settings are the snapshot's.
  [[ -n "${PROJECT}" ]] || die "REPRODUCE_MODELS needs PROJECT"
  export PYTHONPATH="${WORKSPACE_ROOT:-/app}:${PYTHONPATH:-}"
  PYTHON_BIN="python"
  [[ -x /venv/bin/python ]] && PYTHON_BIN="/venv/bin/python"
  if [[ -n "${REPRODUCE_ROOT:-}" ]]; then
    # the chosen folder (mounted at /app/reproduce_root) is the root of every output of these runs;
    # config files and scripts still come from the project (utils/config_loader.py TRACEFLOW_OUTPUT_ROOT)
    export TRACEFLOW_OUTPUT_ROOT=/app/reproduce_root TRACEFLOW_OUTPUT_PROJECT="${PROJECT}"
  fi
  log "DECISION: REPRODUCE → models=${REPRODUCE_MODELS} project=${PROJECT} env=${ENV} batch=${BATCH:-<unset>} root=${REPRODUCE_ROOT:-<project>}"
  cd "${SCC_ROOT}/src"
  reproduce_args=(train-cmd --project "${PROJECT}" --env "${ENV}" -n "${REPRODUCE_MODELS}")
  [[ "${FORCE:-0}" == "1" ]] && reproduce_args+=(-f)     # FORCE=1: past an exhausted retry budget too
  exec "${PYTHON_BIN}" -m core.cli "${reproduce_args[@]}"
fi

if [[ -n "${RERUN_EXPERIMENT}" ]]; then
  [[ -f "${RERUN_ORCH}" ]] || die "rerun orchestrator not found at ${RERUN_ORCH}"
  chmod +x "${RERUN_ORCH}" 2>/dev/null || true

  export MODELS="${RERUN_EXPERIMENT}"

  log "DECISION: RERUN → ${RERUN_EXPERIMENT}"
  log "FORCING MODELS=${MODELS} for downstream rerun orchestrator"
  exec "${RERUN_ORCH}"
fi

if [[ "${SEARCH}" == "1" || "${AUTO_SEARCH_FORCE}" == "1" ]]; then
  log "DECISION: AUTO_SEARCH forced"
  run_auto_search
fi

if [[ "${MLOPS}" == "1" || "${AUTO_MLOPS_FORCE}" == "1" ]]; then
  log "DECISION: AUTO_MLOPS forced"
  run_auto_mlops
fi

AUTO_PROBE="$(probe_master_mode)" || AUTO_PROBE="manual||"

PROBE_MODE="${AUTO_PROBE%%|*}"
PROBE_REST="${AUTO_PROBE#*|}"
PROBE_PROJECT="${PROBE_REST%%|*}"

if [[ -n "${PROBE_PROJECT}" ]]; then
  export PROJECT="${PROJECT:-$PROBE_PROJECT}"
fi

case "${PROBE_MODE}" in
  search)
    log "DECISION: AUTO_SEARCH selected by master.yaml | PROJECT=${PROJECT:-<unset>} MODEL_ID=${MODEL_ID:-<unset>}"
    run_auto_search
    ;;
  auto)
    log "DECISION: AUTO_MLOPS selected by master.yaml | PROJECT=${PROJECT:-<unset>} MODEL_ID=${MODEL_ID:-<unset>}"
    run_auto_mlops
    ;;
  manual|*)
    log "DECISION: MANUAL selected by master.yaml | PROJECT=${PROJECT:-<unset>} MODELS=${MODELS:-<unset>} MODEL_ID=${MODEL_ID:-<unset>}"
    run_manual
    ;;
esac
