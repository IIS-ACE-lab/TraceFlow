#!/bin/bash
#services/side_channel_core/ci_cd/build.sh
set -e

echo "[BUILD] Running build stage..."
mkdir -p ./artifacts
echo "Build artifacts generated on $(date)" > ./artifacts/build.log
echo "[BUILD] Success."
echo "[BUILD] Build artifacts:"
ls -l ./artifacts
sleep 2
echo "[BUILD] Build stage completed."
echo "[BUILD] Build artifacts:"
