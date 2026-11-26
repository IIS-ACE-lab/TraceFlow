#!/usr/bin/env bash
# services/side_channel_core/scripts/gpu_reset.sh

# ──────────────────────────────
# gpu_reset.sh — Robust GPU recoverer:
# persistenced + stale GPU-context kill + cache clean + safe reboot fallback
# Side Channel Core 
# ──────────────────────────────

set -euo pipefail

echo "[GPU-RESET]🚦Checking visible GPUs..."
GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)"

# 1. If no GPUs found → try persistenced restart
if [ -z "$GPUS" ]; then
    echo "[GPU-RESET] 🐌 No GPUs found — driver or nvidia-smi may be down."
    echo "[GPU-RESET] 🔄️ Attempting persistenced restart..."

    if ! command -v nvidia-persistenced &>/dev/null; then
        echo "[GPU-RESET] ❌ nvidia-persistenced not found. 🪛 Install NVIDIA driver tools. 🔄️ A full reboot may be required."
    else
        sudo pkill -f nvidia-persistenced || true
        sudo rm -rf /var/run/nvidia-persistenced
        sudo mkdir -p /var/run/nvidia-persistenced
        sudo chown "$USER:$(id -gn)" /var/run/nvidia-persistenced
        sudo nvidia-persistenced --verbose || true
        echo "[GPU-RESET]  ✅ Persistenced restarted. Run 'nvidia-smi' to verify."
    fi
else
    echo "[GPU-RESET] ✅ GPUs found. GPU ID/s: $GPUS"
fi

# 2. Kill ONLY processes actually holding GPU contexts (NOT all Python)
echo "[GPU-RESET] 💀 Killing GPU-bound processes (safe; not all python)..."

# Primary: kill compute processes shown by nvidia-smi
GPU_PIDS=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk '{print $1}' | sort -u || true)"
fi

if [ -n "${GPU_PIDS}" ]; then
    echo "[GPU-RESET] 🗑️ GPU compute PIDs: ${GPU_PIDS}"
    for PID in ${GPU_PIDS}; do
        # Skip if PID doesn't exist anymore
        if ! kill -0 "${PID}" 2>/dev/null; then
            continue
        fi
        CMD="$(ps -p "${PID}" -o cmd= 2>/dev/null || echo "unknown")"
        echo "[GPU-RESET] 💀 Killing GPU PID ${PID} :: ${CMD}"
        sudo kill -TERM "${PID}" 2>/dev/null || true
    done

    # Give a short grace, then SIGKILL remaining
    sleep 2
    for PID in ${GPU_PIDS}; do
        if kill -0 "${PID}" 2>/dev/null; then
            CMD="$(ps -p "${PID}" -o cmd= 2>/dev/null || echo "unknown")"
            echo "[GPU-RESET] 🗡️ Forcing kill GPU PID ${PID} :: ${CMD}"
            sudo kill -KILL "${PID}" 2>/dev/null || true
        fi
    done
else
    echo "[GPU-RESET] ✅ No GPU compute PIDs reported by nvidia-smi."

    # Fallback: kill processes with open /dev/nvidia* handles (covers some edge cases)
    if command -v fuser >/dev/null 2>&1; then
        DEV_PIDS="$(sudo fuser -v /dev/nvidia* 2>/dev/null | awk 'NR>1 {for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$$/) print $i}' | sort -u || true)"
        if [ -n "${DEV_PIDS}" ]; then
            echo "[GPU-RESET] 🗑️ PIDs holding /dev/nvidia*: ${DEV_PIDS}"
            for PID in ${DEV_PIDS}; do
                if ! kill -0 "${PID}" 2>/dev/null; then
                    continue
                fi
                CMD="$(ps -p "${PID}" -o cmd= 2>/dev/null || echo "unknown")"
                echo "[GPU-RESET] 🗡️ Killing /dev/nvidia* holder PID ${PID} :: ${CMD}"
                sudo kill -TERM "${PID}" 2>/dev/null || true
            done
            sleep 2
            for PID in ${DEV_PIDS}; do
                if kill -0 "${PID}" 2>/dev/null; then
                    CMD="$(ps -p "${PID}" -o cmd= 2>/dev/null || echo "unknown")"
                    echo "[GPU-RESET] 🗡️ Forcing kill /dev/nvidia* holder PID ${PID} :: ${CMD}"
                    sudo kill -KILL "${PID}" 2>/dev/null || true
                fi
            done
        else
            echo "[GPU-RESET] ✅ No /dev/nvidia* holders found."
        fi
    else
        echo "[GPU-RESET] ❌ fuser not available; skipping /dev/nvidia* holder scan."
    fi
fi

# 3. Empty CUDA cache (Python) — do NOT crash script if torch isn't available
echo "[GPU-RESET] 🧹 Clearing CUDA cache..."
python - <<'PY' || echo "[GPU-RESET] ❌ Python cleanup failed."
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[GPU-RESET] 🚮 CUDA cache cleared.")
    else:
        print("[GPU-RESET] ❌ No CUDA device detected.")
except Exception as e:
    print(f"[GPU-RESET] ❌ torch cleanup skipped: {e}")
PY

# 4. Final check — if still broken, offer safe reboot twice
echo "[GPU-RESET] 🔍 Checking GPU status post-cleanup..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "[GPU-RESET] nvidia-smi still failing — you may need to reboot."

    read -rp "[GPU-RESET] Do you want to reboot now? [y/N]: " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        read -rp "[GPU-RESET] Confirm again — really reboot now? [y/N]: " REPLY2
        if [[ "$REPLY2" =~ ^[Yy]$ ]]; then
            echo "[GPU-RESET] 🔄️ Rebooting system now..."
            sudo reboot
        else
            echo "[GPU-RESET] ❌ Reboot cancelled."
        fi
    else
        echo "[GPU-RESET] ❌Reboot skipped by user."
    fi
else
    echo "[GPU-RESET] ✅ GPUs healthy — no reboot needed!"
fi

echo "[GPU-RESET] 🫧✨ Done."
