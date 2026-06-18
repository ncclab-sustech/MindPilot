#!/bin/bash
set -euo pipefail

# Run heuristic generation benchmark
# Three methods: EEG Feature Guidance, Target Image CLIP Guidance, Random Generation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Heuristic Generation Benchmark..."
echo "This will run 3 methods on multiple target images"
echo ""
# Set library path
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

python exp-benchmark_heuristic_generation.py

echo ""
echo "Benchmark completed! Check outputs/benchmark_heuristic_generation/ for results"
