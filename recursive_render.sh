#!/bin/bash
set -e

# ================= 配置区 =================
RES_HOME="/home/ps/hjf/3dgs/res"

# 允许配置多个分辨率，脚本会自动用逗号拼接
RESOLUTION_LIST=("3840x2160")

# 默认 GPU 进程数
NPROC_DEFAULT=1
NPROC_MULTI=2

# 将分辨率数组转换为逗号分隔的字符串
RES_STR=$(IFS=,; echo "${RESOLUTION_LIST[*]}")
# ==========================================

# 核心渲染函数
# 参数:
#   $1: 源目录 (source_dir)
#   $2: 输出目录 (output_dir)
#   $3: iteration
#   $4: nproc_per_node
#   $5: 日志文件名 (可选)
run_render() {
    local src_dir="$1"
    local out_dir="$2"
    local iteration="$3"
    local nproc="${4:-$NPROC_DEFAULT}"
    local log_file="$5"

    local cmd=(
        torchrun --standalone --nnodes=1 "--nproc-per-node=${nproc}" recursive_render.py
        "${src_dir}" "${out_dir}"
        --iteration "${iteration}"
        --resolutions "${RES_STR}"
        --fps-test
    )

    echo "=========================================="
    echo "Running: ${src_dir} -> ${out_dir} (iter: ${iteration}, nproc: ${nproc})"
    echo "=========================================="

    if [[ -n "${log_file}" ]]; then
        "${cmd[@]}" 2>&1 | tee "${log_file}"
    else
        "${cmd[@]}"
    fi
}

# ================= 执行区 =================

# 1. training/building
run_render \
    "${RES_HOME}/training/building" \
    "${RES_HOME}/rubble-colmap/output/grendel_all" \
    70000 \
    "${NPROC_MULTI}" \
    "building_fps.log"

# 2. rubble-colmap
run_render \
    "${RES_HOME}/rubble-colmap" \
    "${RES_HOME}/rubble-colmap/output/grendel_all" \
    70000 \
    "${NPROC_MULTI}" \
    "building_fps.log"

# 3. tanksandtemples/truck
run_render \
    "${RES_HOME}/training/tanksandtemples/truck" \
    "${RES_HOME}/training/tanksandtemples/truck/trained" \
    29997 \
    "${NPROC_DEFAULT}"

# 4. tanksandtemples/train
run_render \
    "${RES_HOME}/training/tanksandtemples/train" \
    "${RES_HOME}/training/tanksandtemples/train/trained" \
    29997 \
    "${NPROC_DEFAULT}"

# 5. deep-blending/drjohnson
run_render \
    "${RES_HOME}/training/deep-blending/drjohnson" \
    "${RES_HOME}/training/deep-blending/drjohnson/trained" \
    30000 \
    "${NPROC_DEFAULT}"

# 6. deep-blending/playroom
run_render \
    "${RES_HOME}/training/deep-blending/playroom" \
    "${RES_HOME}/training/deep-blending/playroom/trained" \
    30000 \
    "${NPROC_DEFAULT}"
