#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi
BOOTSTRAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$("$PYTHON_BIN" "$BOOTSTRAP_DIR/encoding_config.py" shell)"

# Read list-like env vars (space- or comma-separated). CLI flags override these later.
_read_list_env() {
    local _raw="${1:-}"
    _raw="${_raw//,/ }"
    # shellcheck disable=SC2206
    echo ${_raw}
}

# Defaults: env var (if set) -> built-in default. Example:
#   BRAIN_REGIONS=occipital DNN_MODELS=moco bash train.sh --eval_start 0 --eval_end 1000
_subjects_raw="${SUBJECTS:-1}"
# shellcheck disable=SC2206
SUBJECTS=($(_read_list_env "$_subjects_raw"))
_dnn_raw="${DNN_MODELS:-alexnet}"
# shellcheck disable=SC2206
DNN_MODELS=($(_read_list_env "$_dnn_raw"))

SUBJECTS_MODE="${SUBJECTS_MODE:-within}"
PRETRAINED="${PRETRAINED:-True}"
LAYERS="${LAYERS:-all}"
AVG_REPETITIONS="${AVG_REPETITIONS:-true}"
BRAIN_REGIONS="${BRAIN_REGIONS:-occipital_parietal}"
USE_PCA="${USE_PCA:-true}"
N_COMPONENTS="${N_COMPONENTS:-1000}"
TIME_MODE="${TIME_MODE:-all}"
TIME_POINT_MS="${TIME_POINT_MS:-120}"
# Evaluation uses the saved EEG time axis; default analysis window 60--500 ms (paper)
EVAL_TIME_START_MS="${EVAL_TIME_START_MS:-60}"
EVAL_TIME_END_MS="${EVAL_TIME_END_MS:-500}"
RUN_EVAL="${RUN_EVAL:-true}"
FORCE_PCA="${FORCE_PCA:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_PCA="${SKIP_PCA:-false}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Pipeline: optional PCA -> linearizing encoding training -> evaluation

Options:
  --sub SUB [SUB ...]       Subject ID(s), default: 1
  --dnn DNN [DNN ...]       DNN model(s), default: alexnet
  --subjects MODE           within|between, default: within
  --pretrained BOOL         true|false, default: true
  --layers MODE             all|single|appended, default: all
  --avg_repetitions BOOL    Average EEG repetitions per image, default: true
  --brain_regions REGIONS   Comma-separated regions, default: occipital_parietal
                            Available: occipital, parietal, occipital_parietal,
                            central, frontal, temporal, centro_parietal
  --use_pca BOOL            Use PCA feature maps, default: true
  --n_components N          PCA dimensions, default: 1000
  --time_mode MODE          all|single, default: all
  --time_point_ms MS        Time point for single mode, default: 120
  --eval_start MS           Eval window start on saved EEG time axis, default: 60
  --eval_end MS             Eval window end on saved EEG time axis, default: 500
  --no_eval                 Skip evaluation after training
  --force_pca               Re-run PCA even if outputs exist
  --skip_train              Skip training (evaluation only)
  --skip_pca                Skip PCA step (use existing PCA features)
  -h, --help                Show this help

Environment variables (used when flag not passed; space- or comma-separated lists):
  SUBJECTS, DNN_MODELS, SUBJECTS_MODE, PRETRAINED, LAYERS, AVG_REPETITIONS,
  BRAIN_REGIONS, USE_PCA, N_COMPONENTS, TIME_MODE, TIME_POINT_MS,
  EVAL_TIME_START_MS, EVAL_TIME_END_MS, RUN_EVAL, FORCE_PCA, SKIP_TRAIN, SKIP_PCA,
  PROJECT_DIR, EEG_DATA_DIR, IMAGE_SET_DIR, PCA_SCRIPT_DIR, PRETRAIN_WEIGHTS_DIR,
  DATASET_ROOT, PYTHON_BIN

Examples:
  BRAIN_REGIONS=occipital bash train.sh --eval_start 0 --eval_end 1000
  DNN_MODELS="moco alexnet" SUBJECTS="1 2" bash train.sh --skip_train
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sub)
            shift
            SUBJECTS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SUBJECTS+=("$1")
                shift
            done
            ;;
        --dnn)
            shift
            DNN_MODELS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DNN_MODELS+=("$1")
                shift
            done
            ;;
        --subjects) SUBJECTS_MODE="$2"; shift 2 ;;
        --pretrained) PRETRAINED="$2"; shift 2 ;;
        --layers) LAYERS="$2"; shift 2 ;;
        --avg_repetitions) AVG_REPETITIONS="$2"; shift 2 ;;
        --brain_regions) BRAIN_REGIONS="$2"; shift 2 ;;
        --use_pca) USE_PCA="$2"; shift 2 ;;
        --n_components) N_COMPONENTS="$2"; shift 2 ;;
        --time_mode) TIME_MODE="$2"; shift 2 ;;
        --time_point_ms) TIME_POINT_MS="$2"; shift 2 ;;
        --eval_start) EVAL_TIME_START_MS="$2"; shift 2 ;;
        --eval_end) EVAL_TIME_END_MS="$2"; shift 2 ;;
        --no_eval) RUN_EVAL="false"; shift ;;
        --force_pca) FORCE_PCA="true"; shift ;;
        --skip_train) SKIP_TRAIN="true"; shift ;;
        --skip_pca) SKIP_PCA="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

cd "$SCRIPT_DIR" || exit 1

COMMON_ARGS=(
    --subjects "$SUBJECTS_MODE"
    --pretrained "$PRETRAINED"
    --layers "$LAYERS"
    --avg_repetitions "$AVG_REPETITIONS"
    --brain_regions "$BRAIN_REGIONS"
    --use_pca "$USE_PCA"
    --n_components "$N_COMPONENTS"
    --time_mode "$TIME_MODE"
    --time_point_ms "$TIME_POINT_MS"
    --project_dir "$PROJECT_DIR"
)

print_settings() {
    echo ""
    echo "============================================================"
    echo "Experiment settings"
    echo "============================================================"
    echo "  subjects           : ${SUBJECTS[*]}"
    echo "  dnn models         : ${DNN_MODELS[*]}"
    echo "  subjects mode      : $SUBJECTS_MODE"
    echo "  pretrained         : $PRETRAINED"
    echo "  layers             : $LAYERS"
    echo "  avg repetitions    : $AVG_REPETITIONS"
    echo "  brain regions      : $BRAIN_REGIONS"
    echo "  use pca            : $USE_PCA"
    echo "  pca n_components   : $N_COMPONENTS"
    echo "  time mode          : $TIME_MODE"
    echo "  time point (ms)    : $TIME_POINT_MS"
    echo "  eval window (ms)   : $EVAL_TIME_START_MS - $EVAL_TIME_END_MS (on saved EEG time axis)"
    echo "  run evaluation     : $RUN_EVAL"
    echo "  force pca          : $FORCE_PCA"
    echo "  skip pca           : $SKIP_PCA"
    echo "  eeg data dir       : $EEG_DATA_DIR"
    echo "  image set dir      : $IMAGE_SET_DIR"
    echo "  pca script dir     : $PCA_SCRIPT_DIR"
    echo "  project_dir        : $PROJECT_DIR"
    echo "============================================================"
    echo ""
}

config_path() {
    "$PYTHON_BIN" "$SCRIPT_DIR/encoding_config.py" "$@"
}

maybe_run_pca() {
    local dnn="$1"
    if [[ "$USE_PCA" != "true" || "$SKIP_PCA" == "true" ]]; then
        echo "[PCA] Skipped"
        return 0
    fi

    local pca_dir
    pca_dir="$(config_path pca-dir \
        --dnn "$dnn" \
        --pretrained "$PRETRAINED" \
        --layers "$LAYERS" \
        --project_dir "$PROJECT_DIR")"
    local train_pca="$pca_dir/pca_feature_maps_training.npy"
    local test_pca="$pca_dir/pca_feature_maps_test.npy"

    if [[ "$FORCE_PCA" == "true" || ! -f "$train_pca" || ! -f "$test_pca" ]]; then
        echo "[PCA] Running feature_maps_pca.py for $dnn (n_components=$N_COMPONENTS)..."
        "$PYTHON_BIN" "$PCA_SCRIPT_DIR/feature_maps_pca.py" \
            --dnn "$dnn" \
            --pretrained "$PRETRAINED" \
            --layers "$LAYERS" \
            --n_components "$N_COMPONENTS" \
            --project_dir "$PROJECT_DIR"
        echo "[PCA] Done -> $pca_dir"
    else
        echo "[PCA] Using existing features: $pca_dir"
    fi
}

run_training() {
    local sub="$1"
    local dnn="$2"
    echo "[Train] Subject $sub, DNN $dnn"
    "$PYTHON_BIN" linearizing_encoding.py \
        --sub "$sub" \
        --dnn "$dnn" \
        "${COMMON_ARGS[@]}"
}

run_evaluation() {
    local sub="$1"
    local dnn="$2"
    echo "[Eval] Subject $sub, DNN $dnn (saved EEG times, window ${EVAL_TIME_START_MS}-${EVAL_TIME_END_MS} ms)"
    "$PYTHON_BIN" evaluate_linearizing_encoding.py \
        --sub "$sub" \
        --dnn "$dnn" \
        "${COMMON_ARGS[@]}" \
        --eval_time_start_ms "$EVAL_TIME_START_MS" \
        --eval_time_end_ms "$EVAL_TIME_END_MS"
}

print_settings

total_tasks=$((${#SUBJECTS[@]} * ${#DNN_MODELS[@]}))
current_task=0
summary_files=()

echo "Starting pipeline: $total_tasks subject×model job(s)"
start_time=$(date)

for dnn in "${DNN_MODELS[@]}"; do
    maybe_run_pca "$dnn"
done

for sub in "${SUBJECTS[@]}"; do
    for dnn in "${DNN_MODELS[@]}"; do
        current_task=$((current_task + 1))
        echo ""
        echo "========================================"
        echo "Job [$current_task/$total_tasks] sub=$sub dnn=$dnn"
        echo "========================================"

        if [[ "$SKIP_TRAIN" != "true" ]]; then
            run_training "$sub" "$dnn"
        fi

        if [[ "$RUN_EVAL" == "true" ]]; then
            run_evaluation "$sub" "$dnn"
            eval_csv="$(config_path eval-summary \
                --sub "$sub" \
                --dnn "$dnn" \
                "${COMMON_ARGS[@]}")"
            if [[ -f "$eval_csv" ]]; then
                summary_files+=("$eval_csv")
            fi
        fi
    done
done

end_time=$(date)
echo ""
echo "============================================================"
echo "Pipeline complete"
echo "  started : $start_time"
echo "  finished: $end_time"
echo "============================================================"

if [[ ${#summary_files[@]} -gt 0 ]]; then
    echo ""
    echo "Evaluation summaries:"
    for f in "${summary_files[@]}"; do
        echo "--- $f ---"
        cat "$f"
        echo ""
    done
fi
