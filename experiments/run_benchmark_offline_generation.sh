#!/bin/bash
# 设置库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# export https_proxy="10.16.11.87:7890"
# export http_proxy="10.16.11.87:7890"

# 运行benchmark
CUDA_VISIBLE_DEVICES=5 python /home/ldy/Workspace/Closed_loop_optimizing/experiments/benchmark_offline_generation.py
