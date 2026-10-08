#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITERATIONS=30000

# train.py 在最后一次 iteration 保存的 point cloud 目录。
# 7000 iterations -> iteration_6997
SAVED_ITERATION=$((ITERATIONS - 3))
densify_grad_threshold=0.0001
densification_interval=100
densify_until_iter=25000
usage() {
    echo "Usage:"
    echo "  $0 <dataset_path> [model_path]"
    echo "  $0 --recursive <dataset_root> [model_root]"
    echo
    echo "Examples:"
    echo "  $0 /path/to/stump"
    echo "  $0 /path/to/stump /path/to/stump/trained"
    echo "  $0 --recursive /path/to/mipnerf360"
    echo "  $0 --recursive /path/to/mipnerf360 /path/to/mipnerf360_trained"
}

train_one() {
    local source_path="$1"
    local model_path="$2"

    echo
    echo "============================================================"
    echo "Training dataset:"
    echo "  Source: ${source_path}"
    echo "  Model:  ${model_path}"
    echo "============================================================"
    echo

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=2 \
        train.py \
        --bsz 1 \
        -s "${source_path}" \
        -m "${model_path}" \
        --iterations "${ITERATIONS}" \
        --save_iterations 10000 20000 30000 \
        --eval \
        --densify_grad_threshold "${densify_grad_threshold}" \
        --densification_interval "${densification_interval}" \
        --resolution -1 \
        --min_opacity 0.001 \
        --prune_scale3d 0.2 \
        --lambda_dssim 0.2 \
        --percent_dense 0.01 \
        --densify_until_iter ${densify_until_iter} \
        --image_loading_mode lazy

    local status=$?

    echo
    echo "[DEBUG] train.py exit code: ${status}"
    echo

    return "${status}"
}

is_dataset() {
    local path="$1"

    [[ -d "${path}/sparse" || -f "${path}/transforms_train.json" ]]
}

is_trained() {
    local model_path="$1"

    local checkpoint_path="${model_path}/point_cloud/iteration_${SAVED_ITERATION}"

    [[ -d "${checkpoint_path}" ]]
}

# ----------------------------------------------------------------------
# Recursive mode
# ----------------------------------------------------------------------
if [[ "${1:-}" == "--recursive" ]]; then
    DATASET_ROOT="${2:-${SOURCE_PATH:-}}"
    MODEL_ROOT="${3:-${MODEL_PATH:-}}"

    if [[ -z "${DATASET_ROOT}" ]]; then
        echo "Error: dataset root is required." >&2
        usage
        exit 1
    fi

    if [[ ! -d "${DATASET_ROOT}" ]]; then
        echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
        exit 1
    fi

    # Default: each dataset is trained into its own "trained" directory.
    if [[ -z "${MODEL_ROOT}" ]]; then
        MODEL_ROOT="${DATASET_ROOT}"
    fi

    mapfile -t DATASETS < <(
        find "${DATASET_ROOT}" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -print | sort
    )

    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        echo "No dataset directories found under: ${DATASET_ROOT}" >&2
        exit 1
    fi

    TRAINED_COUNT=0
    SKIPPED_COUNT=0
    FAILED_DATASETS=()

    echo "Discovered dataset directories:"
    printf '  %s\n' "${DATASETS[@]}"
    echo

    for DATASET_PATH in "${DATASETS[@]}"; do
        DATASET_NAME="$(basename "${DATASET_PATH}")"

        if ! is_dataset "${DATASET_PATH}"; then
            echo "Skipping non-dataset directory: ${DATASET_NAME}"
            continue
        fi

        if [[ "${MODEL_ROOT}" == "${DATASET_ROOT}" ]]; then
            CURRENT_MODEL_PATH="${DATASET_PATH}/trained"
        else
            CURRENT_MODEL_PATH="${MODEL_ROOT}/${DATASET_NAME}"
        fi

        # --------------------------------------------------------------
        # Skip already trained dataset
        # --------------------------------------------------------------
        if is_trained "${CURRENT_MODEL_PATH}"; then
            echo
            echo "############################################################"
            echo "# Dataset: ${DATASET_NAME}"
            echo "# Status:  ALREADY TRAINED - SKIPPING"
            echo "# Model:   ${CURRENT_MODEL_PATH}"
            echo "# Checkpoint:"
            echo "#   ${CURRENT_MODEL_PATH}/point_cloud/iteration_${SAVED_ITERATION}"
            echo "############################################################"
            echo

            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
        fi

        echo
        echo "############################################################"
        echo "# Dataset: ${DATASET_NAME}"
        echo "# Source:  ${DATASET_PATH}"
        echo "# Model:   ${CURRENT_MODEL_PATH}"
        echo "# Status:  TRAINING"
        echo "############################################################"
        echo

        set +e
        train_one "${DATASET_PATH}" "${CURRENT_MODEL_PATH}"
        STATUS=$?
        set -e

        echo
        echo "[DEBUG] train_one returned: ${STATUS}"
        echo "[DEBUG] dataset: ${DATASET_NAME}"
        echo

        if [[ ${STATUS} -eq 0 ]]; then
            TRAINED_COUNT=$((TRAINED_COUNT + 1))
            echo "[SUCCESS] ${DATASET_NAME}"
        else
            FAILED_DATASETS+=("${DATASET_NAME}")
            echo "[FAILED] ${DATASET_NAME}, exit code=${STATUS}"
        fi

        echo "[DEBUG] Moving to next dataset..."
    done

    echo
    echo "============================================================"
    echo "Recursive training finished."
    echo "  Root:     ${DATASET_ROOT}"
    echo "  Trained:  ${TRAINED_COUNT}"
    echo "  Skipped:  ${SKIPPED_COUNT}"
    echo "  Failed:   ${#FAILED_DATASETS[@]}"
    echo "============================================================"

    if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
        echo
        echo "Failed datasets:"
        printf '  - %s\n' "${FAILED_DATASETS[@]}"
        exit 1
    fi

    exit 0
fi

# ----------------------------------------------------------------------
# Normal single-dataset mode
# ----------------------------------------------------------------------
SOURCE_PATH="${1:-${SOURCE_PATH:-"${SCRIPT_DIR}/../res/training/tanksandtemples/train"}}"
MODEL_PATH="${2:-${MODEL_PATH:-"${SOURCE_PATH}/trained"}}"

if [[ ! -d "${SOURCE_PATH}" ]]; then
    echo "Dataset path does not exist: ${SOURCE_PATH}" >&2
    usage
    exit 1
fi

if ! is_dataset "${SOURCE_PATH}"; then
    echo "Unrecognized dataset layout: ${SOURCE_PATH}" >&2
    echo "Expected COLMAP data with sparse/ or Blender data with transforms_train.json." >&2
    exit 1
fi

train_one "${SOURCE_PATH}" "${MODEL_PATH}"