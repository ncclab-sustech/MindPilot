#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi
BOOTSTRAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$("$PYTHON_BIN" "$BOOTSTRAP_DIR/encoding_config.py" shell)"

# Space-separated overrides:
#   ANALYSIS=timeseries SUBJECTS="1 2" DNN_MODELS="alexnet resnet50" bash train_data_amount.sh
analysis="${ANALYSIS:-scalar}"

read -r -a dnn_models <<< "${DNN_MODELS:-alexnet cornet_s dino_vit_b_16 dino2_vit_b_14 moco openclip_vit_b_32 resnet50 synclr_vit_b_16 vit_b_32}"
read -r -a subjects <<< "${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"

cd "$SCRIPT_DIR" || exit 1

echo "Starting training-data amount analysis..."
echo "Analysis mode: $analysis"
echo "Total jobs: ${#subjects[@]} subjects x ${#dnn_models[@]} models = $((${#subjects[@]} * ${#dnn_models[@]}))"

start_time=$(date)
echo "Start time: $start_time"
echo ""

total_tasks=$((${#subjects[@]} * ${#dnn_models[@]}))
current_task=0

for sub in "${subjects[@]}"; do
    for dnn in "${dnn_models[@]}"; do
        current_task=$((current_task + 1))
        echo "========================================"
        echo "Progress: [$current_task/$total_tasks] Subject: $sub, DNN Model: $dnn"
        echo "========================================"

        "$PYTHON_BIN" training_data_amount.py \
            --analysis "$analysis" \
            --sub "$sub" \
            --dnn "$dnn" \
            --project_dir "$PROJECT_DIR"

        echo "Done: subject $sub, DNN $dnn"
        remaining=$((total_tasks - current_task))
        if [ "$remaining" -gt 0 ]; then
            echo "Remaining jobs: $remaining"
        fi
        echo ""
    done
done

end_time=$(date)
echo "========================================"
echo "All training-data amount jobs completed."
echo "Start time: $start_time"
echo "End time: $end_time"
echo "Completed jobs: $total_tasks"
echo "========================================"
