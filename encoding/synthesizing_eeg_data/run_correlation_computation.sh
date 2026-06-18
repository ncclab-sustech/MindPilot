#!/bin/bash
set -euo pipefail

# Script to compute correlation results from synthetic EEG data
# This script reads pre-computed synthetic EEG data and computes correlation
# and explained variance with biological EEG test data

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$("$PYTHON_BIN" "$SCRIPT_DIR/encoding_config.py" shell)"
cd "$SCRIPT_DIR"

echo "Starting correlation computation from synthetic EEG data..."

"$PYTHON_BIN" compute_correlation_from_synthetic.py \
    --project_dir "$PROJECT_DIR" \
    --n_iter 10

echo "Correlation computation completed!"
