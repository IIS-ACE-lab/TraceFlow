#!/bin/bash
#services/side_channel_core/ci_cd/test.sh
set -e

echo "[TEST] ENV is: $ENV"

PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
TEST_DIR="$PROJECT_ROOT/tests"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

echo "[TEST] Running tests from: $TEST_DIR"

pytest "$TEST_DIR" | tee "$LOG_DIR/test_results.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[TEST] Failed. See $LOG_DIR/test_results.log"
    exit 1
fi

echo "[TEST] Success"
