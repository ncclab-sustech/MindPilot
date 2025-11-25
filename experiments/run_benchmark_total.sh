#!/bin/bash

# 全面评估框架运行脚本 - 7种方法对比
# 使用方法：bash run_benchmark_total.sh

# ========== 配置 ==========
GPU_ID=6
CONFIG_FILE="benchmark_config_total.json"
EXP_TYPE="exp1"  # 可选: all, exp1, exp2

# ========== 环境检查 ==========
echo "=========================================="
echo "全面评估框架 - 7种优化方法对比"
echo "=========================================="
echo ""

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件 $CONFIG_FILE 不存在！"
    echo "请先创建配置文件，参考 benchmark_config_total.json 模板"
    exit 1
fi

echo "✅ 配置文件: $CONFIG_FILE"
echo "✅ GPU设备: $GPU_ID"
echo "✅ 实验类型: $EXP_TYPE"
echo ""
echo "支持的7种方法："
echo "  1. PseudoModel (Offline)         - 离线采样 + GP优化"
echo "  2. HeuristicClosedLoop          - 闭环迭代（融合+贪婪采样）⭐ 新增"
echo "  3. DDPO                          - 强化学习（PPO）"
echo "  4. DPOK                          - 强化学习（KL正则化）"
echo "  5. D3PO                          - 强化学习（DPO）"
echo "  6. BayesianOpt                   - 贝叶斯优化"
echo "  7. CMA-ES                        - 进化策略"
echo ""

# 检查依赖库
echo "检查依赖库..."
python3 -c "import torch; print('  ✅ PyTorch:', torch.__version__)" || exit 1
python3 -c "import diffusers; print('  ✅ Diffusers:', diffusers.__version__)" || exit 1
python3 -c "import scipy; print('  ✅ SciPy:', scipy.__version__)" || exit 1

# CMA-ES是可选的
if python3 -c "import cma" 2>/dev/null; then
    python3 -c "import cma; print('  ✅ CMA-ES:', 'installed')"
else
    echo "  ⚠️  CMA-ES库未安装（CMA-ES方法将使用fallback模式）"
    echo "     安装方法: pip install cma"
fi

echo ""

# ========== 运行实验 ==========
echo "=========================================="
echo "开始运行实验..."
echo "=========================================="
echo ""

# 设置GPU并运行
CUDA_VISIBLE_DEVICES=$GPU_ID python benchmark_framework_total.py \
    --config $CONFIG_FILE \
    --exp $EXP_TYPE

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 实验完成！"
    echo "=========================================="
    echo ""
    echo "结果保存在配置文件中指定的output_dir目录"
    echo ""
    echo "查看结果："
    echo "  - CSV结果: output_dir/exp1_results.csv"
    echo "  - 统计摘要: output_dir/exp1_summary.csv"
    echo "  - 对比图表: output_dir/exp1_comparison.png"
    echo "  - 生成图像: output_dir/images/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 实验运行失败！"
    echo "=========================================="
    echo ""
    echo "请检查："
    echo "  1. 配置文件中的路径是否正确"
    echo "  2. GPU显存是否充足"
    echo "  3. 所有依赖库是否已安装"
    echo ""
    exit 1
fi

# ========== 可选：显示统计摘要 ==========
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output_dir'])" 2>/dev/null)

if [ -n "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/exp1_summary.csv" ]; then
    echo "=========================================="
    echo "统计摘要预览："
    echo "=========================================="
    head -n 10 "$OUTPUT_DIR/exp1_summary.csv"
    echo ""
fi


