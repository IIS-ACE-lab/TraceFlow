#!/usr/bin/env bash
#
# In‑container cleanup (safe, fast, idempotent).
# Defaults can be tuned via env:
#   SCC_TMP_DIR=/app/tmp   CLEAN_PY=1   CLEAN_TORCH=1   CLEAN_NV=auto|0|1
# Flags:
#   --no-exit-on-error     : never exit non‑zero
#   --lock-file <path>     : optional flock to avoid concurrent runs

set -Euo pipefail
trap 'echo "[CLEANUP] failed at line $LINENO: $BASH_COMMAND" >&2' ERR

# --------- args ---------
NO_EXIT=0
LOCK_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-exit-on-error) NO_EXIT=1; shift ;;
    --lock-file) LOCK_FILE="${2:-}"; shift 2 ;;
    *) echo "[CLEANUP] Unknown arg: $1"; shift ;;
  esac
done

# --------- context guards ---------
: "${SCC_TMP_DIR:=/app/tmp}"
TMP_DIR="$SCC_TMP_DIR"
mkdir -p "$TMP_DIR" 2>/dev/null || true
echo "[CLEANUP] tmp dir: $TMP_DIR"

# Only perform deep cleanup when inside a container
if [[ "${RUN_CONTEXT:-}" != "container" ]]; then
  if [[ ! -f /.dockerenv && ! -f /run/.containerenv ]] \
     && ! grep -qaE '(docker|containerd|kubepods|podman)' /proc/1/cgroup 2>/dev/null; then
    echo "[CLEANUP] Not in container → skipping."
    exit 0
  fi
fi

# Optional single‑runner lock
if [[ -n "$LOCK_FILE" ]]; then
  exec {FD}>"$LOCK_FILE" || true
  if ! flock -n "$FD"; then
    echo "[CLEANUP] Another cleanup in progress; exiting."
    exit 0
  fi
fi

# --------- process cleanup ---------
if [[ "${CLEAN_PY:-1}" == "1" ]]; then
  echo "[CLEANUP] Terminating Python processes (except self)..."
  SELF=$$
  # First SIGTERM, then SIGKILL for stragglers
  mapfile -t PIDS < <(ps -eo pid,comm,args --no-headers 2>/dev/null \
    | awk -v self="$SELF" '($2 ~ /^python(3)?$/ || $0 ~ /python(3)?/) && $1 != self {print $1}' || true)
  if ((${#PIDS[@]})); then
    kill -TERM "${PIDS[@]}" 2>/dev/null || true
    sleep 0.3
    # re-scan and kill -9 the survivors
    mapfile -t PLEFT < <(ps -o pid= -p "${PIDS[@]}" 2>/dev/null || true)
    ((${#PLEFT[@]})) && kill -KILL "${PLEFT[@]}" 2>/dev/null || true
  else
    echo "[CLEANUP] No Python processes."
  fi
fi

# --------- caches ---------
if [[ "${CLEAN_TORCH:-1}" == "1" ]]; then
  echo "[CLEANUP] Releasing Torch CUDA cache..."
  /venv/bin/python - <<'PY' 2>/dev/null || true
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[CLEANUP] Torch CUDA cache cleared.")
    else:
        print("[CLEANUP] CUDA not available.")
except Exception as e:
    print(f"[CLEANUP] Torch cleanup exception: {e}")
PY
fi

echo "[CLEANUP] Removing user caches (safe subsets)..."
# Torch extensions & NV JIT caches (safe to delete)
rm -rf "${HOME:-/root}/.cache/torch_extensions" 2>/dev/null || true
rm -rf "${HOME:-/root}/.nv/ComputeCache"        2>/dev/null || true
rm -rf "${HOME:-/root}/.nv/CompileCache"        2>/dev/null || true
# PyTorch cpp extensions (common path)
rm -rf "${HOME:-/root}/.cache/torch_extensions" 2>/dev/null || true

# tmp hygiene (leave logs & sockets alone)
find "$TMP_DIR" -maxdepth 1 -type f -name '*.lock' -delete 2>/dev/null || true
find /tmp -maxdepth 1 -type f -mmin +120 -size -64M -delete 2>/dev/null || true

# --------- zombie check ---------
echo "[CLEANUP] Checking for zombie Python processes..."
ZPIDS="$(ps -eo pid,stat,comm --no-headers 2>/dev/null | awk '$2 ~ /Z/ && $3 ~ /python/ {print $1}' || true)"
if [[ -n "${ZPIDS:-}" ]]; then
  echo "[CLEANUP] Zombies: $ZPIDS"
  # Attempt to reap by killing parents
  ps -eo ppid=,pid=,stat= 2>/dev/null | awk '$3 ~ /Z/ {print $1}' | sort -u \
    | while read -r ZPPID; do [[ "$ZPPID" != "1" ]] && kill -9 "$ZPPID" 2>/dev/null || true; done
else
  echo "[CLEANUP] No zombies."
fi

# --------- optional GPU soft reset ---------
# CLEAN_NV=auto → try reset only if no compute processes are active
# CLEAN_NV=1    → force try
# CLEAN_NV=0    → skip
: "${CLEAN_NV:=auto}"
if command -v nvidia-smi >/dev/null 2>&1 && [[ "$CLEAN_NV" != "0" ]]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$' || true)"
  if [[ -z "$GPUS" ]]; then
    echo "[GPU-RESET] No GPUs visible."
  else
    SAFE_TO_RESET=1
    if [[ "$CLEAN_NV" == "auto" ]]; then
      # If any compute processes are running, skip
      if nvidia-smi | awk '/Processes:/,/^$/' | grep -qE '[0-9]'; then
        SAFE_TO_RESET=0
      fi
    fi
    if [[ "$SAFE_TO_RESET" == "1" ]]; then
      for GPU in $GPUS; do
        nvidia-smi --gpu-reset -i "$GPU" >/dev/null 2>&1 \
          || echo "[GPU-RESET] Soft reset failed for GPU $GPU."
      done
    else
      echo "[GPU-RESET] Skipping reset (active processes)."
    fi
  fi
fi

# --------- post snapshot ---------
echo "[CLEANUP] GPU snapshot (post):"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi 2>/dev/null || true

echo "[CLEANUP] Deep cleanup finished."

# never propagate failure if asked not to
if [[ "$NO_EXIT" == "1" ]]; then
  exit 0
fi
