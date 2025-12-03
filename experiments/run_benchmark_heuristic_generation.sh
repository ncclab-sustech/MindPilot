#!/bin/bash

# 运行启发式生成评测
# 三种方法：EEG Feature Guidance, Target Image CLIP Guidance, Random Generation

cd /home/ldy/Workspace/Closed_loop_optimizing/experiments

echo "Starting Heuristic Generation Benchmark..."
echo "This will run 3 methods on multiple target images"
echo ""
# 设置库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=5  python exp-benchmark_heuristic_generation.py

echo ""
echo "Benchmark completed! Check outputs/benchmark_heuristic_generation/ for results"

