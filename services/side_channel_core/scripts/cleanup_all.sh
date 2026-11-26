#!/usr/bin/env bash
# robust cleanup_all.sh — Side Channel Core 
# Kills everything: no orchestrator PID spared

echo "[CLEANUP-ALL] Starting aggressive full cleanup..."

# 1. Show GPU usage
echo "[CLEANUP-ALL] Current GPU processes:"
nvidia-smi || echo "[CLEANUP-ALL] nvidia-smi failed."

# 2. Kill ALL python processes forcefully
echo "[CLEANUP-ALL] Killing ALL Python processes..."
ps -eo pid,cmd | grep '[p]ython' | awk '{print $1}' | while read PID; do
  kill -9 "$PID" 2>/dev/null
done

# 3. Torch cleanup if possible
echo "[CLEANUP-ALL] Releasing torch CUDA cache..."
python -c "
import torch
if torch.cuda.is_available():
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()
else:
  print('[CLEANUP-ALL] No CUDA available, skipping torch cleanup.')
" || echo "[CLEANUP-ALL] Torch cleanup script failed."

# 4. Remove torch extensions cache
echo "[CLEANUP-ALL] Removing torch extensions cache..."
rm -rf ~/.cache/torch_extensions

# 5. Re-check GPU
echo "[CLEANUP-ALL] Verifying GPU health post-cleanup:"
nvidia-smi || echo "[CLEANUP-ALL] nvidia-smi failed."

# 6. Reap zombies by killing parents, no exceptions
echo "[CLEANUP-ALL] Killing parents of any zombies..."
ps -eo ppid=,pid=,stat= | awk '$3 ~ /Z/ {print $1}' | sort -u | while read ZPPID; do
  if [ "$ZPPID" != "1" ] && [ "$ZPPID" != "$$" ]; then
    echo "[CLEANUP-ALL] Killing parent PID $ZPPID..."
    kill -9 "$ZPPID" 2>/dev/null
  fi
done

# 7. Attempt GPU reset
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$')
if [ -z "$GPUS" ]; then
  echo "[GPU-RESET] No valid GPUs found for reset."
else
  for GPU in $GPUS; do
    sudo nvidia-smi --gpu-reset -i "$GPU" > /dev/null 2>&1 || \
    echo "[GPU-RESET] Soft reset failed for GPU $GPU (common for GeForce)."
  done
fi

echo "[CLEANUP-ALL] Full aggressive cleanup done. No process spared!"
