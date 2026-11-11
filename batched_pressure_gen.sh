#!/bin/bash

# 定义基础路径和常量
RES_HOME=~/hjf/3dgs/res
PT_SUBPATH=output/grendel_all/point_cloud
PT_NAME=point_cloud.ply

# 定义场景路径常量
TandT="training/tanksandtemples"
Rubble="rubble-colmap"
Building="training/building"
DeepBlending="training/deep-blending"

FRAME_COUNT=500
# 定义核心执行函数
# 参数1: 场景相对路径 (SCENE)
# 参数2: 迭代次数 (ITERATION)
generate_views() {
    local scene=$1
    local iteration=$2
    
    # 拼接完整的 PLY 文件路径
    local ply_path="$RES_HOME/$scene/$PT_SUBPATH/iteration_$iteration/$PT_NAME"
    
    echo "Processing: $ply_path"
    
    # 检查文件是否存在，避免报错
    if [ ! -f "$ply_path" ]; then
        echo "Warning: File not found - $ply_path"
        return 1
    fi

  local output_dir="${ply_path%/*}"
 
  python gen_pressure_view.py "$ply_path" \
      --trajectory scan --count $FRAME_COUNT \
      --output "$output_dir/pressure_scan.json"
 
  python gen_pressure_view.py "$ply_path" \
      --trajectory orbit --count $FRAME_COUNT \
      --output "$output_dir/pressure_orbit.json"
}

# --- 主执行流程 ---

# 1. 处理 Rubble 场景 (Iteration 70000)
generate_views "$Rubble" 70000

# 2. 处理 Building 场景
generate_views "$Building" 70000

# 3. 处理 Iteration 40000 的批量场景
ITER_40K=29997
PT_SUBPATH=trained/point_cloud
generate_views "$TandT/train" $ITER_40K
generate_views "$TandT/truck" $ITER_40K
ITER_40K=30000
generate_views "$DeepBlending/drjohnson" $ITER_40K
generate_views "$DeepBlending/playroom" $ITER_40K

echo "All tasks completed."