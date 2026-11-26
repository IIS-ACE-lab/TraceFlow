#!/usr/bin/env bash
# ──────────────────────────────
# gpu_watchdog.sh — Continuous GPU health monitor & auto-healer 
# Runs in the background, checks GPU health every X seconds.
# Detects driver or GPU hang mid-training → auto runs cleanup + reset.

#Author: Shraddha Mishra
# ──────────────────────────────

# Safe fallback: use env or default to 60 seconds.
CHECK_INTERVAL=${CHECK_INTERVAL:-60}

LOG_DIR=/app/services/side_channel_core/deployment/docker/tmp
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/watchdog.log"

echo "[WATCHDOG] GPU watchdog started — checking every $CHECK_INTERVAL seconds." #>> "$LOG"

while true; do
  if ! nvidia-smi > /dev/null 2>&1; then
    echo "[WATCHDOG] nvidia-smi failed at $(date)! Running cleanup + reset attempt." >> "$LOG"
    bash "$(dirname "$0")/cleanup.sh" >> "$LOG"
    bash "$(dirname "$0")/gpu_reset.sh" >> "$LOG"
  else
    echo "[WATCHDOG] GPU healthy at $(date)" >> "$LOG"
  fi
  sleep "$CHECK_INTERVAL"
done
