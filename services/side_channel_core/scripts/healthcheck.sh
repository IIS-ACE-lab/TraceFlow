#!/usr/bin/env bash
# healthcheck.sh — only warn on failure

nvidia-smi > /dev/null 2>&1 || echo "[HEALTHCHECK] ❌ NVIDIA-SMI failed!"

nvcc -V | grep "release" || echo "[HEALTHCHECK] ❌ nvcc not found!"

python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || echo "[HEALTHCHECK] torch.cuda.is_available() is False!"
echo "[HEALTH] Checking NVML and CUDA..."
nvidia-smi || { echo "[HEALTH] ❌ nvidia-smi failed. Please reboot."; exit 1; }
python -c "import torch; print(torch.cuda.is_available())" | grep True || { echo "[HEALTH] ❌ torch sees no CUDA. Please reboot."; exit 1; }
echo "[HEALTH] ✅ GPU healthy, all green!"
