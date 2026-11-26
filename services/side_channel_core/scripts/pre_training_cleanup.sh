#!/usr/bin/env bash
# ──────────────
# pre_training_cleanup.sh
# Safe cleanup BEFORE training starts.
# Kills leftover user Python, clears CUDA contexts, only logs on failure.
# ──────────────

# Kill leftover Python processes (your own only)
ps -u $USER -o pid,comm | grep python | awk '{print $1}' | xargs -r kill -9

# Clear torch extension build cache
rm -rf ~/.cache/torch_extensions

# Quick CUDA context release using a fresh Python
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()" 2>/dev/null || echo "[PRE-CLEANUP] torch cleanup failed — check environment."

# Optional: sanity check driver
nvidia-smi > /dev/null 2>&1 || echo "[PRE-CLEANUP] NVIDIA-SMI failed — driver may be crashed."

# Silent success if no issues.
