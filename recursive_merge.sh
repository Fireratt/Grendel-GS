#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_distributed_ply.py"
ITERATIONS=30000
SAVED_ITERATION=$((ITERATIONS))

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

is_dataset() {
    local path="$1"
    [[ -d "${path}/sparse" || -f "${path}/transforms_train.json" ]]
}

checkpoint_path_for() {
    local model_path="$1"
    echo "${model_path}/point_cloud/iteration_${SAVED_ITERATION}"
}

has_distributed_shards() {
    local checkpoint_path="$1"
    compgen -G "${checkpoint_path}/point_cloud_rk*_ws*.ply" > /dev/null
}

merge_one() {
    local source_path="$1"
    local model_path="$2"
    local checkpoint_path
    checkpoint_path="$(checkpoint_path_for "${model_path}")"

    echo
    echo "============================================================"
    echo "Merging dataset:"
    echo "  Source:     ${source_path}"
    echo "  Model:      ${model_path}"
    echo "  Checkpoint: ${checkpoint_path}"
    echo "============================================================"
    echo

    if [[ ! -d "${checkpoint_path}" ]]; then
        echo "Error: checkpoint directory does not exist: ${checkpoint_path}" >&2
        return 1
    fi

    if [[ -f "${checkpoint_path}/point_cloud.ply" ]]; then
        echo "[SKIP] Merged point cloud already exists:"
        echo "       ${checkpoint_path}/point_cloud.ply"
        return 0
    fi

    if ! has_distributed_shards "${checkpoint_path}"; then
        echo "Error: no distributed Gaussian PLY shards found in: ${checkpoint_path}" >&2
        echo "Expected files matching point_cloud_rk*_ws*.ply." >&2
        return 1
    fi

    python "${MERGE_SCRIPT}" "${checkpoint_path}"
}

# Recursive mode
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
    if [[ -z "${MODEL_ROOT}" ]]; then
        MODEL_ROOT="${DATASET_ROOT}"
    fi

    mapfile -t DATASETS < <(find "${DATASET_ROOT}" -mindepth 1 -maxdepth 1 -type d -print | sort)
    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        echo "No dataset directories found under: ${DATASET_ROOT}" >&2
        exit 1
    fi

    MERGED_COUNT=0
    SKIPPED_COUNT=0
    FAILED_DATASETS=()

    echo "Discovered dataset directories:"
    for dataset_path in "${DATASETS[@]}"; do
        echo "  ${dataset_path}"
    done
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

        checkpoint_path="$(checkpoint_path_for "${CURRENT_MODEL_PATH}")"
        if [[ -f "${checkpoint_path}/point_cloud.ply" ]]; then
            echo "[SKIP] ${DATASET_NAME}: already merged"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
        fi

        set +e
        merge_one "${DATASET_PATH}" "${CURRENT_MODEL_PATH}"
        STATUS=$?
        set -e

        if [[ ${STATUS} -eq 0 ]]; then
            MERGED_COUNT=$((MERGED_COUNT + 1))
            echo "[SUCCESS] ${DATASET_NAME}"
        else
            FAILED_DATASETS+=("${DATASET_NAME}")
            echo "[FAILED] ${DATASET_NAME}, exit code=${STATUS}"
        fi
    done

    echo
    echo "============================================================"
    echo "Recursive merging finished."
    echo "  Root:     ${DATASET_ROOT}"
    echo "  Merged:   ${MERGED_COUNT}"
    echo "  Skipped:  ${SKIPPED_COUNT}"
    echo "  Failed:   ${#FAILED_DATASETS[@]}"
    echo "============================================================"

    if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
        echo
        echo "Failed datasets:"
        for dataset_name in "${FAILED_DATASETS[@]}"; do
            echo "  - ${dataset_name}"
        done
        exit 1
    fi
    exit 0
fi

# Normal single-dataset mode
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

merge_one "${SOURCE_PATH}" "${MODEL_PATH}"
