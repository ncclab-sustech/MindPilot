#!/bin/bash

# Comprehensive evaluation framework run script - 7 methods comparison
# Usage: bash run_benchmark_total.sh

# ========== Configuration ==========
GPU_ID=6
CONFIG_FILE="benchmark_config_total.json"
EXP_TYPE="exp1"  # Options: all, exp1, exp2

# ========== Environment Check ==========
echo "=========================================="
echo "Comprehensive Evaluation Framework - 7 Optimization Methods Comparison"
echo "=========================================="
echo ""

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE does not exist!"
    echo "Please create a config file first, refer to benchmark_config_total.json template"
    exit 1
fi

echo "Config file: $CONFIG_FILE"
echo "GPU device: $GPU_ID"
echo "Experiment type: $EXP_TYPE"
echo ""
echo "Supported 7 methods:"
echo "  1. PseudoModel (Offline)         - Offline sampling + GP optimization"
echo "  2. HeuristicClosedLoop          - Closed-loop iteration (fusion + greedy sampling)"
echo "  3. DDPO                          - Reinforcement learning (PPO)"
echo "  4. DPOK                          - Reinforcement learning (KL regularization)"
echo "  5. D3PO                          - Reinforcement learning (DPO)"
echo "  6. BayesianOpt                   - Bayesian optimization"
echo "  7. CMA-ES                        - Evolution strategy"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; print('  PyTorch:', torch.__version__)" || exit 1
python3 -c "import diffusers; print('  Diffusers:', diffusers.__version__)" || exit 1
python3 -c "import scipy; print('  SciPy:', scipy.__version__)" || exit 1

# CMA-ES is optional
if python3 -c "import cma" 2>/dev/null; then
    python3 -c "import cma; print('  CMA-ES:', 'installed')"
else
    echo "  WARNING: CMA-ES library not installed (CMA-ES method will use fallback mode)"
    echo "  Install with: pip install cma"
fi

echo ""

# ========== Run Experiments ==========
echo "=========================================="
echo "Starting experiments..."
echo "=========================================="
echo ""

# Set GPU and run
CUDA_VISIBLE_DEVICES=$GPU_ID python benchmark_framework_total.py \
    --config $CONFIG_FILE \
    --exp $EXP_TYPE

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Experiment completed!"
    echo "=========================================="
    echo ""
    echo "Results saved in the output_dir specified in the config file"
    echo ""
    echo "View results:"
    echo "  - CSV results: output_dir/exp1_results.csv"
    echo "  - Summary statistics: output_dir/exp1_summary.csv"
    echo "  - Comparison charts: output_dir/exp1_comparison.png"
    echo "  - Generated images: output_dir/images/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Experiment failed!"
    echo "=========================================="
    echo ""
    echo "Please check:"
    echo "  1. Are the paths in the config file correct?"
    echo "  2. Is there sufficient GPU memory?"
    echo "  3. Are all dependencies installed?"
    echo ""
    exit 1
fi

# ========== Optional: Show Summary Statistics ==========
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output_dir'])" 2>/dev/null)

if [ -n "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/exp1_summary.csv" ]; then
    echo "=========================================="
    echo "Summary statistics preview:"
    echo "=========================================="
    head -n 10 "$OUTPUT_DIR/exp1_summary.csv"
    echo ""
fi
