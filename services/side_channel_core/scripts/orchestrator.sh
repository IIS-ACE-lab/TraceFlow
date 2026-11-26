#!/usr/bin/env bash
# services/side_channel_core/scripts/orchestrator.sh
# Robust orchestrator wrapper — trap-safe, clear parent-child
# Registry counts support BOTH shapes:
#   - object: {id:{...}}
#   - array:  [{...}, ...]
# normal run:
#   run_map FAILED       -> --retry-failed
#   run_map INTERRUPTED  -> --resume-interrupted   (both flags when both exist)
#   registry fallback    -> retry/resume
#   then new SCC samples
#   stop only when done >= total_trials
# 
# FORCE=1:
#   every pass runs --force, ahead of any run_map recovery: the planner walks
#   the whole SCC grid and retrains every scoped hash (NEW, DONE, FAILED,
#   INTERRUPTED) once in this session (ORCHESTRATOR_SESSION_ID)
#   stop only when done >= total_trials
set -euo pipefail

# -----------------------------------------------------------------------------
# Basic setup
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SCRIPTS_DIR:-$SCRIPT_DIR}"

: "${SCC_TMP_DIR:=/app/tmp}"
TMP_DIR="$SCC_TMP_DIR"
install -d -m 1777 "$TMP_DIR" || { echo "[ORCHESTRATOR] Cannot create $TMP_DIR"; exit 1; }

export PATH="/venv/bin:$PATH"
umask 0022

: "${PROJECT:=${PROJECT:-}}"
: "${DATASET:=${DATASET:-ascad}}"
: "${ALIAS:=${ALIAS:-}}"
: "${MODELS:=${MODELS:-}}"
: "${ENV:=${ENV:-dev}}"
: "${FORCE:=${FORCE:-0}}"
: "${MODEL_ID:=${MODEL_ID:-}}"
export PROJECT DATASET ALIAS MODELS ENV FORCE MODEL_ID
: "${ORCHESTRATOR_SESSION_ID:=$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export ORCHESTRATOR_SESSION_ID
echo "[ORCHESTRATOR] SESSION=${ORCHESTRATOR_SESSION_ID}"
: "${SCC_MASTER_YAML:=/app/services/side_channel_core/config/master.yaml}"
: "${PROJECTS_ROOT:=/app/projects}"
export SCC_MASTER_YAML PROJECTS_ROOT

export RUN_CONTEXT=container SCC_TMP_DIR TMP_DIR SCRIPTS_DIR

# -----------------------------------------------------------------------------
# Orchestrator pid
# -----------------------------------------------------------------------------
echo "[ORCHESTRATOR] Will write orchestrator.pid to $SCC_TMP_DIR"
echo $$ > "$SCC_TMP_DIR/orchestrator.pid"
echo "[ORCHESTRATOR] Registered orchestrator PID: $$"

# Ensure host-side tmp exists when bind-mounted
if [ -d "/app/services/side_channel_core" ]; then
  mkdir -p /app/services/side_channel_core/deployment/docker/tmp || true
  chmod 1777 /app/services/side_channel_core/deployment/docker/tmp || true
fi

# -----------------------------------------------------------------------------
# Resolve PROJECT + PROJECT_ROOT via Python (no bash YAML parsing)
# Outputs: "<project_name>|<project_root_rel>"
# -----------------------------------------------------------------------------
_resolve_project_and_root() {
  /venv/bin/python - <<'PY'
import os, sys
from pathlib import Path

master = Path(os.environ.get("SCC_MASTER_YAML", "/app/services/side_channel_core/config/master.yaml"))
requested = (os.environ.get("PROJECT") or "").strip()

if not master.exists():
    print("||"); sys.exit(0)

try:
    import yaml
    d = yaml.safe_load(master.read_text()) or {}
except Exception:
    print("||"); sys.exit(0)

projects = d.get("project") or []
if isinstance(projects, dict): projects = [projects]
if not isinstance(projects, list): projects = []

def pick_default():
    dp = d.get("default_project")
    if isinstance(dp, str) and dp.strip(): return dp.strip()
    for pr in projects:
        if isinstance(pr, dict) and isinstance(pr.get("name"), str) and pr["name"].strip():
            return pr["name"].strip()
    return ""

def find_block(name: str):
    for pr in projects:
        if isinstance(pr, dict) and pr.get("name") == name:
            return pr
    return None

chosen = requested or pick_default()
block = find_block(chosen) if chosen else None

root_rel = ""
if isinstance(block, dict) and isinstance(block.get("root"), str):
    root_rel = block["root"].strip().rstrip("/")

print(f"{chosen}|{root_rel}")
PY
}

RESOLVED="$(_resolve_project_and_root || true)"
RESOLVED_PROJECT="${RESOLVED%%|*}"
RESOLVED_ROOT_REL="${RESOLVED#*|}"

if [[ -z "${RESOLVED_PROJECT}" ]]; then
  echo "[ORCHESTRATOR] Could not resolve PROJECT. Set PROJECT=... or ensure master.yaml has default_project."
  exit 2
fi

PROJECT="${RESOLVED_PROJECT}"
export PROJECT

[[ -n "${RESOLVED_ROOT_REL}" ]] || RESOLVED_ROOT_REL="${PROJECT}"
PROJECT_ROOT="${PROJECTS_ROOT%/}/${RESOLVED_ROOT_REL}"
export PROJECT_ROOT

: "${ARTIFACTS_DIR:=${PROJECT_ROOT}/artifacts}"
export ARTIFACTS_DIR

RUN_REGISTRY="${ARTIFACTS_DIR}/run_registry.json"
TEST_REGISTRY="${ARTIFACTS_DIR}/test_registry.json"
RUN_MAP="${ARTIFACTS_DIR}/run_map.jsonl"
export RUN_REGISTRY TEST_REGISTRY RUN_MAP

echo "[ORCHESTRATOR] Resolved: PROJECT=${PROJECT} PROJECT_ROOT=${PROJECT_ROOT}"
echo "[ORCHESTRATOR] Contract: DATASET=${DATASET} ALIAS=${ALIAS:-<unset>} MODELS=${MODELS:-<unset>} ENV=${ENV}"
echo "[ORCHESTRATOR] Batch: BATCH=${BATCH:-<unset>} BATCH_ID=${BATCH_ID:-<from session>}"

# -----------------------------------------------------------------------------
# Run summary: total wall time of this orchestration and the GPU it ran on
# -----------------------------------------------------------------------------
ORCH_START_EPOCH="$(date +%s)"
ORCH_START_ISO="$(date '+%Y-%m-%d %H:%M:%S %Z')"
RUN_SUMMARY_FILE="${TMP_DIR}/run_summary.txt"
rm -f "${RUN_SUMMARY_FILE}" 2>/dev/null || true
ORCH_END_EPOCH=""
ORCH_END_ISO=""
mark_run_end() {
  ORCH_END_EPOCH="$(date +%s)"
  ORCH_END_ISO="$(date '+%Y-%m-%d %H:%M:%S %Z')"
}
print_run_summary() {
  local outcome="${1:-finished}"
  [[ -n "${ORCH_END_EPOCH}" ]] || mark_run_end
  local elapsed=$(( ORCH_END_EPOCH - ORCH_START_EPOCH ))
  local gpus
  gpus="$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null | sed 's/^/id=/; s/, /  /g' | paste -sd ';' - || true)"
  {
    echo "[ORCHESTRATOR] ------------------------------------------------------------"
    printf '[ORCHESTRATOR] Total time : %02dh %02dm %02ds (%d s)  [%s]\n' \
      $(( elapsed / 3600 )) $(( (elapsed % 3600) / 60 )) $(( elapsed % 60 )) "${elapsed}" "${outcome}"
    echo "[ORCHESTRATOR] Started    : ${ORCH_START_ISO}"
    echo "[ORCHESTRATOR] Finished   : ${ORCH_END_ISO}"
    echo "[ORCHESTRATOR] GPU        : ${gpus:-not reported by nvidia-smi}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
    echo "[ORCHESTRATOR] Scope      : ENV=${ENV} PROJECT=${PROJECT:-<unset>} DATASET=${DATASET:-<unset>} MODEL_ID=${MODEL_ID:-<unset>}"
    echo "[ORCHESTRATOR] ------------------------------------------------------------"
  } | tee "${RUN_SUMMARY_FILE}" || true
}

# -----------------------------------------------------------------------------
# Trap: forward to child, cleanup
# -----------------------------------------------------------------------------
child_pid=""
cleanup() {
  local sig_rc=130

  echo "[ORCHESTRATOR] Trap caught. Forwarding SIGINT to child PID=${child_pid:-<none>}..."

  if [[ -n "${child_pid:-}" ]]; then
    kill -SIGINT "$child_pid" 2>/dev/null || true

    # give python orchestrator a short chance to update registries itself
    sleep 2

    wait "$child_pid" 2>/dev/null || true
  fi

  echo "[ORCHESTRATOR] Host-side interrupt reconciliation..."
  _mark_interrupted_on_host_exit || true

  print_run_summary "interrupted"

  bash "${SCRIPT_DIR}/cleanup.sh" --no-exit-on-error || true

  rm -f "$TMP_DIR/orchestrator.pid" || true

  exit "$sig_rc"
}

trap cleanup SIGINT SIGTERM

# -----------------------------------------------------------------------------
# Build optional CLI flags from user contract
# -----------------------------------------------------------------------------
build_user_flags() {
  local out=""
  [[ -n "${ALIAS:-}" ]] && out+=" --alias ${ALIAS}"
  [[ -n "${MODEL_ID:-}" ]] && out+=" --model-id ${MODEL_ID}"

  if [[ -n "${MODELS:-}" ]]; then
    IFS=',' read -r -a arr <<< "${MODELS}"
    for m in "${arr[@]}"; do
      m="$(echo "${m}" | xargs)"
      [[ -n "${m}" ]] && out+=" --model-name ${m}"
    done
  fi

  echo "$out"
}
USER_FLAGS="$(build_user_flags)"
_mark_interrupted_on_host_exit() {
  /venv/bin/python - <<'PY'
import json, os
from pathlib import Path
from datetime import datetime, timezone

project_root = Path(os.environ["PROJECT_ROOT"])
art = project_root / "artifacts"

paths = {
    "run": art / "run_registry.json",
    "test": art / "test_registry.json",
    "map": art / "run_map.jsonl",
}

now = datetime.now(timezone.utc).isoformat()

# utils/seal.py: verify before rewriting, so a changed registry is refused and
# not re-signed; the rewrite carries a fresh seal.
from utils.seal import check as check_seal, seal_text, strip as strip_seal

ACTIVE = {"in_progress", "running", "in_process", "queued", "IN_PROGRESS", "RUNNING", "IN_PROCESS", "QUEUED"}

# Only this orchestration's runs: ENV, MODEL_ID (when set) and this shell's
# ORCHESTRATOR_SESSION_ID, recorded as run_session_id in run_registry.json or
# run_map.jsonl. Active entries of other ENVs, MODEL_IDs or sessions belong to
# another orchestrator that is still running them. The child's handle_signal()
# has already marked its own hashes, including a planner claim that does not
# carry the session id yet; a later -c pass resumes one it missed.
env = (os.environ.get("ENV") or "").strip()
model_id = (os.environ.get("MODEL_ID") or "").strip()
session = (os.environ.get("ORCHESTRATOR_SESSION_ID") or "").strip()

def norm(v):
    return ("" if v is None else str(v)).strip()

def ours(rec):
    return (
        isinstance(rec, dict)
        and bool(session)
        and norm(rec.get("run_session_id")) == session
        and norm(rec.get("env")) == env
        and (not model_id or norm(rec.get("model_id")) == model_id)
    )

def load_json(path):
    # run_registry.json / test_registry.json: one (indented) JSON object.
    if not path.exists():
        return {}
    text = path.read_text().strip()
    if not text:
        return {}
    check_seal(path, cache=False)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"[ORCHESTRATOR] {path} is not a JSON object keyed by model_hash; not rewritten.")
    return strip_seal(data)

def load_jsonl(path):
    # run_map.jsonl: one record per line, the latest record per model_hash wins.
    out = {}
    if not path.exists():
        return out
    check_seal(path, cache=False)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = strip_seal(json.loads(line))
        h = rec.get("model_hash") if isinstance(rec, dict) else None
        if h:
            out[str(h)] = rec
    return out

def save_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    lines = [
        json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        for v in data.values()
        if isinstance(v, dict)
    ]
    tmp.write_text(seal_text(path, "\n".join(lines) + ("\n" if lines else "")))
    tmp.replace(path)

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(seal_text(path, json.dumps(data, indent=2)))
    tmp.replace(path)

run_reg = load_json(paths["run"])
test_reg = load_json(paths["test"])
run_map = load_jsonl(paths["map"])

active_hashes = set()

for h, rec in run_reg.items():
    if ours(rec) and str(rec.get("status", "")).strip() in ACTIVE:
        active_hashes.add(str(h))

for h, rec in run_map.items():
    if ours(rec) and str(rec.get("status", "")).strip() in ACTIVE:
        active_hashes.add(str(h))

for h in active_hashes:
    if isinstance(run_reg.get(h), dict):
        run_reg[h]["status"] = "INTERRUPTED"
        run_reg[h]["finished_at"] = now
        run_reg[h]["status_reason"] = "host_shell_interrupt"
        run_reg[h].pop("reason", None)

    if isinstance(test_reg.get(h), dict):
        test_reg[h]["status"] = "FAILED"
        test_reg[h]["finished_at"] = now
        test_reg[h]["status_reason"] = "host_shell_interrupt"
        test_reg[h].pop("reason", None)

    if isinstance(run_map.get(h), dict):
        run_map[h]["status"] = "INTERRUPTED"
        run_map[h]["updated_at"] = now
        run_map[h]["reason"] = "host_shell_interrupt"

if active_hashes:
    save_json(paths["run"], run_reg)
    save_json(paths["test"], test_reg)
    save_jsonl(paths["map"], run_map)
    print(f"[ORCHESTRATOR] marked interrupted hashes: {sorted(active_hashes)}", flush=True)
PY
}
# -----------------------------------------------------------------------------
# Python-based counters (no jq)
# -----------------------------------------------------------------------------
_py_counts() {
  # args: <json_file> <env_or_empty>
  local jf="$1"
  local envf="${2:-}"
  /venv/bin/python - <<'PY'
import json, os, sys
jf = os.environ.get("JF","")
env = os.environ.get("ENVF","")
def norm(s):
    return ("" if s is None else str(s)).strip()

try:
    with open(jf,"r",encoding="utf-8") as f:
        d = json.load(f)
except Exception:
    print("0 0")
    sys.exit(0)

if isinstance(d, dict):
    vals = list(d.values())
elif isinstance(d, list):
    vals = d
else:
    vals = []
vals = [x for x in vals if isinstance(x, dict)]  # not the registry's "_tfseal" (utils/seal.py)

if env:
    vals = [x for x in vals if norm(x.get("env","")) == norm(env)]

total = len(vals)
# SKIPPED: a test not run because perform_attack.test.attack_times is 0; finished.
done = sum(1 for x in vals if str(x.get("status","")) in ("DONE", "SKIPPED"))
print(f"{total} {done}")
PY
}

_py_status_count() {
  # args: <json_file> <status_lc>
  local jf="$1"
  local st="${2:-}"
  /venv/bin/python - <<'PY'
import json, os, sys
jf = os.environ.get("JF","")
st = (os.environ.get("ST","") or "").strip().lower()
try:
    with open(jf,"r",encoding="utf-8") as f:
        d = json.load(f)
except Exception:
    print("0")
    sys.exit(0)

if isinstance(d, dict):
    vals = list(d.values())
elif isinstance(d, list):
    vals = d
else:
    vals = []
vals = [x for x in vals if isinstance(x, dict)]  # not the registry's "_tfseal" (utils/seal.py)

def norm(s): return ("" if s is None else str(s)).strip().lower()
cnt = sum(1 for x in vals if norm(x.get("status","")) == st)
print(str(cnt))
PY
}
_py_run_map_flags() {
  /venv/bin/python - <<'PY'
import json, os
from pathlib import Path

path = Path(os.environ.get("RUN_MAP", ""))
env = (os.environ.get("ENV") or "").strip()
model_id = (os.environ.get("MODEL_ID") or "").strip()

def norm(v):
    return ("" if v is None else str(v)).strip()

def load_records(path):
    # run_map.jsonl is JSONL (utils/run_map.py): one record per line, the latest
    # record per model_hash wins. Falls back to a legacy single JSON document.
    try:
        text = path.read_text().strip()
    except Exception:
        return []
    if not text:
        return []
    latest = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            latest = None
            break
        if isinstance(rec, dict) and norm(rec.get("model_hash")):
            latest[norm(rec.get("model_hash"))] = rec
    if latest:
        return list(latest.values())
    try:
        data = json.loads(text)
    except Exception:
        return []
    return list(data.values()) if isinstance(data, dict) else data if isinstance(data, list) else []

records = load_records(path)

failed = 0
interrupted = 0

for r in records:
    if not isinstance(r, dict):
        continue
    if env and norm(r.get("env")) != env:
        continue
    if model_id and norm(r.get("model_id")) != model_id:
        continue

    st = norm(r.get("status")).upper()
    if st == "FAILED":
        failed += 1
    elif st in {"INTERRUPTED", "IN_PROGRESS", "IN_PROCESS", "RUNNING"}:
        interrupted += 1

# Both when both are present: a FAILED record (possibly retry-exhausted) must
# not hide an INTERRUPTED one, which would then never be resumed.
flags = []
if failed:
    flags.append("--retry-failed")
if interrupted:
    flags.append("--resume-interrupted")
print(" ".join(flags))
PY
}

_py_scope_state() {
  /venv/bin/python - <<'PY'
import json, os
from pathlib import Path

env = (os.environ.get("ENV") or "").strip()
model_id = (os.environ.get("MODEL_ID") or "").strip()
# Completion is judged on the runs of THIS orchestration only. ENV and MODEL_ID
# alone also match finished runs of the same model on another dataset or with
# other trainer values, which would end a sweep before its last combination.
session = (os.environ.get("ORCHESTRATOR_SESSION_ID") or "").strip()
paths = [
    Path(os.environ.get("RUN_MAP", "")),
    Path(os.environ.get("RUN_REGISTRY", "")),
]

def norm(v):
    return ("" if v is None else str(v)).strip()

def load(path):
    # run_map.jsonl is JSONL (one record per line, keyed here by model_hash);
    # run_registry.json is an indented dict, whose first line is not valid
    # JSON on its own, so it takes the whole-document fallback.
    try:
        text = path.read_text().strip()
    except Exception:
        return {}
    if not text:
        return {}
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            out = None
            break
        if isinstance(rec, dict) and norm(rec.get("model_hash")):
            out[norm(rec.get("model_hash"))] = rec
    if out:
        return out
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}

done_hashes = set()
failed_hashes = set()
interrupted_hashes = set()
known_hashes = set()
total_trials = 0

for path in paths:
    data = load(path)
    for h, r in data.items():
        if not isinstance(r, dict):
            continue
        if env and norm(r.get("env")) != env:
            continue
        if model_id and norm(r.get("model_id")) != model_id:
            continue

        h = str(h)
        known_hashes.add(h)

        st = norm(r.get("status")).upper()

        if session and norm(r.get("run_session_id")) != session:
            continue
        if st == "DONE":
            done_hashes.add(h)
        elif st == "FAILED":
            failed_hashes.add(h)
        elif st in {"INTERRUPTED", "IN_PROGRESS", "IN_PROCESS", "RUNNING", "QUEUED"}:
            interrupted_hashes.add(h)

        for k in ("total_trials", "trial_total"):
            try:
                total_trials = max(total_trials, int(r.get(k) or 0))
            except Exception:
                pass

print(json.dumps({
    "known": len(known_hashes),
    "done": len(done_hashes),
    "failed": len(failed_hashes),
    "interrupted": len(interrupted_hashes),
    "total_trials": total_trials,
    "remaining": max(total_trials - len(done_hashes), 0) if total_trials else -1,
}))
PY
}

_print_counts() {
  local total=0 done=0 test_total=0 test_done=0
  local env_total=0 env_done=0 env_test_total=0 env_test_done=0

  read -r total done < <(JF="$RUN_REGISTRY" ENVF="" _py_counts "$RUN_REGISTRY" "") || { total=0; done=0; }
  read -r test_total test_done < <(JF="$TEST_REGISTRY" ENVF="" _py_counts "$TEST_REGISTRY" "") || { test_total=0; test_done=0; }

  echo "Total Runs in Database: ${done}/${total}, Tests: ${test_done}/${test_total}"

  if [[ -n "${ENV:-}" ]]; then
    read -r env_total env_done < <(JF="$RUN_REGISTRY" ENVF="$ENV" _py_counts "$RUN_REGISTRY" "$ENV") || { env_total=0; env_done=0; }
    read -r env_test_total env_test_done < <(JF="$TEST_REGISTRY" ENVF="$ENV" _py_counts "$TEST_REGISTRY" "$ENV") || { env_test_total=0; env_test_done=0; }
    echo "${ENV}-scoped Runs: ${env_done}/${env_total}, Tests: ${env_test_done}/${env_test_total}"
  fi
}

_is_done_env_scoped() {
  local env_total=0 env_done=0 env_test_total=0 env_test_done=0
  read -r env_total env_done < <(JF="$RUN_REGISTRY" ENVF="$ENV" _py_counts "$RUN_REGISTRY" "$ENV") || { env_total=0; env_done=0; }
  read -r env_test_total env_test_done < <(JF="$TEST_REGISTRY" ENVF="$ENV" _py_counts "$TEST_REGISTRY" "$ENV") || { env_test_total=0; env_test_done=0; }

  if [[ "${env_total}" -eq 0 && "${env_test_total}" -eq 0 ]]; then
    echo "No registry entries for ENV=${ENV}. Nothing to do. Exiting."
    return 0
  fi

  [[ "${env_done}" -eq "${env_total}" && "${env_test_done}" -eq "${env_test_total}" ]]
}

_is_done_global() {
  local total=0 done=0 test_total=0 test_done=0
  read -r total done < <(JF="$RUN_REGISTRY" ENVF="" _py_counts "$RUN_REGISTRY" "") || { total=0; done=0; }
  read -r test_total test_done < <(JF="$TEST_REGISTRY" ENVF="" _py_counts "$TEST_REGISTRY" "") || { test_total=0; test_done=0; }

  if [[ "${total}" -eq 0 && "${test_total}" -eq 0 ]]; then
    echo "No registry entries (global). Nothing to do. Exiting."
    return 0
  fi

  [[ "${done}" -eq "${total}" && "${test_done}" -eq "${test_total}" ]]
}

# -----------------------------------------------------------------------------
# Main orchestrator loop
# -----------------------------------------------------------------------------

FINAL_RC=0
while true; do
  CLI_FLAGS=""

  run_reg="${RUN_REGISTRY}"
  test_reg="${TEST_REGISTRY}"

  state="$(_py_scope_state)"
  run_map_flag="$(_py_run_map_flags || true)"

  # FORCE first: an INTERRUPTED/FAILED run_map record must not turn a FORCE
  # pass into a resume/retry pass that skips the DONE hashes.
  if [[ "${FORCE}" = "1" ]]; then
    CLI_FLAGS="--force"
    echo "[ORCHESTRATOR] FORCE selected: retraining every scoped hash once in session ${ORCHESTRATOR_SESSION_ID}."

  elif [[ -n "${run_map_flag}" ]]; then
    CLI_FLAGS="${run_map_flag}"
    echo "[ORCHESTRATOR] run_map recovery selected: ${CLI_FLAGS}"

  else
    has_failed="$(JF="$run_reg" ST="failed" _py_status_count "$run_reg" "failed")"
    has_in_progress="$(JF="$run_reg" ST="in_progress" _py_status_count "$run_reg" "in_progress")"
    has_interrupted="$(JF="$run_reg" ST="interrupted" _py_status_count "$run_reg" "interrupted")"

    has_test_failed="$(JF="$test_reg" ST="failed" _py_status_count "$test_reg" "failed")"
    has_test_in_progress="$(JF="$test_reg" ST="in_progress" _py_status_count "$test_reg" "in_progress")"
    has_test_interrupted="$(JF="$test_reg" ST="interrupted" _py_status_count "$test_reg" "interrupted")"

    [[ "${has_failed}" -gt 0 || "${has_test_failed}" -gt 0 ]] && CLI_FLAGS+=" --retry-failed"

    if [[ "${has_in_progress}" -gt 0 || "${has_interrupted}" -gt 0 || "${has_test_in_progress}" -gt 0 || "${has_test_interrupted}" -gt 0 ]]; then
      CLI_FLAGS+=" --resume-interrupted"
    fi
  fi

    # Optional strict single-model override
  if [[ -n "${MODEL_NAME:-}" && "${MODEL_NAME:-}" != "\"\"" ]]; then
    CLI_FLAGS+=" --model-name ${MODEL_NAME}"
  fi

  CLI_FLAGS+=" ${USER_FLAGS}"

  echo "ENV=<${ENV}>"
  echo "CLI_FLAGS for this pass: ${CLI_FLAGS}"

  python -m core.cli orchestrate-cmd \
    --project "${PROJECT}" \
    --dataset "${DATASET}" \
    ${CLI_FLAGS} \
    --env "${ENV}" &
  child_pid=$!

  # `|| ...`: under set -e a bare `wait` of a non-zero child ends this script before its exit code is
  # read, so none of the handling below ran (fig24, 2026-10-19: exit 3 left no message and no code).
  compose_exit_code=0
  wait "$child_pid" || compose_exit_code=$?
  
  echo "compose-child exit code=${compose_exit_code}"
  
  if [[ "${compose_exit_code}" = "130" || "${compose_exit_code}" = "143" ]]; then
    echo "[ORCHESTRATOR] Child interrupted. Reconciling registries..."
    _mark_interrupted_on_host_exit || true
    print_run_summary "interrupted"
    exit "${compose_exit_code}"
  fi

  # orchestrator.py exit contract (see its module docstring):
  #   3 = EXIT_RETRY_EXHAUSTED: only terminally FAILED hashes remain
  #       (retry_exhausted=true in run_registry.json). Retrying would loop forever.
  #   4 = EXIT_NOTHING_TO_RUN: nothing left within ENV/MODEL_ID/MODELS scope;
  #       the state below decides whether that is success.
  if [[ "${compose_exit_code}" = "3" ]]; then
    echo "[ORCHESTRATOR] ❌ Retry budget exhausted: some hashes FAILED ${SCC_MAX_FAILED_ATTEMPTS:-3} times."
    echo "[ORCHESTRATOR]    See run_registry.json entries with retry_exhausted=true (and their 'reason')."
    echo "[ORCHESTRATOR]    Fix the cause and delete those entries, or re-run with FORCE=1. Stopping."
    _print_counts
    FINAL_RC=3
    break
  fi

  # Any other failure of the pass (a config error, a crash before a hash was attempted) ends the
  # orchestration with its code: another pass would fail the same way, with no retry budget to stop it.
  if [[ "${compose_exit_code}" != "0" && "${compose_exit_code}" != "4" ]]; then
    echo "[ORCHESTRATOR] ❌ The orchestrator pass failed (exit code ${compose_exit_code}); see the error above. Stopping."
    _print_counts
    FINAL_RC="${compose_exit_code}"
    break
  fi

  echo "Checking pass result..."
  _print_counts

  state="$(_py_scope_state)"
  total_trials="$(python - "$state" <<'PY'
import json, sys
print(int(json.loads(sys.argv[1]).get("total_trials") or 0))
PY
)"
  done_trials="$(python - "$state" <<'PY'
import json, sys
print(int(json.loads(sys.argv[1]).get("done") or 0))
PY
)"
  failed_trials="$(python - "$state" <<'PY'
import json, sys
print(int(json.loads(sys.argv[1]).get("failed") or 0))
PY
)"
  interrupted_trials="$(python - "$state" <<'PY'
import json, sys
print(int(json.loads(sys.argv[1]).get("interrupted") or 0))
PY
)"

  if [[ "${total_trials}" -gt 0 && "${done_trials}" -ge "${total_trials}" && "${failed_trials}" -eq 0 && "${interrupted_trials}" -eq 0 ]]; then
    echo "✅ ✅ ✅ ALL SCC trials complete for ENV=${ENV}: ${done_trials}/${total_trials}."
    break
  fi

  if [[ "${compose_exit_code}" = "4" && "${done_trials}" -gt 0 && "${failed_trials}" -eq 0 && "${interrupted_trials}" -eq 0 ]]; then
    # The planner has nothing left and this orchestration finished runs cleanly:
    # the remaining combinations of the sweep were already DONE before it started.
    echo "✅ ✅ ✅ ALL SCC trials complete for ENV=${ENV}: ${done_trials} run(s) in this orchestration, the rest of the sweep was already DONE."
    break
  fi

  if [[ "${compose_exit_code}" = "4" && "${failed_trials}" -eq 0 && "${interrupted_trials}" -eq 0 ]]; then
    # A resumed queue line whose runs were all DONE before: nothing to run is success.
    echo "✅ ✅ ✅ Nothing left to run for ENV=${ENV} MODEL_ID=${MODEL_ID:-<unset>}: every run in scope is DONE."
    break
  fi

  if [[ "${compose_exit_code}" = "4" ]]; then
    echo "[ORCHESTRATOR] Nothing left to run within scope ENV=${ENV} MODEL_ID=${MODEL_ID:-<unset>} MODELS=${MODELS:-<unset>}:"
    echo "[ORCHESTRATOR]    done=${done_trials}/${total_trials} failed=${failed_trials} interrupted=${interrupted_trials}."
    echo "[ORCHESTRATOR]    Another pass would change nothing. Stopping."
    FINAL_RC=4
    break
  fi

  echo "Not done yet. Continuing with appropriate retry/resume flags..."
  sleep 1
done

mark_run_end

# -----------------------------------------------------------------------------
# Post-orchestration cleanup
# -----------------------------------------------------------------------------
echo "[ORCHESTRATOR] 🧹 Running Post-Orchestration cleanup..."
CLEAN_LOCK_FILE="${TMP_DIR:-/app/tmp}/cleanup.lock"

/bin/bash "${SCRIPT_DIR}/cleanup.sh" \
  --no-exit-on-error \
  --lock-file "$CLEAN_LOCK_FILE" \
  || true

rm -f "${TMP_DIR:-/app/tmp}/orchestrator.pid" 2>/dev/null || true
echo "[ORCHESTRATOR] 🚮 Orchestrator PID deleted."

if [[ "${FINAL_RC}" -eq 0 ]]; then
  print_run_summary "all runs complete"
else
  print_run_summary "stopped with exit code ${FINAL_RC}"
fi
exit "${FINAL_RC}"
