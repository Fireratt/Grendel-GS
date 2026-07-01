#!/bin/bash
set -e

echo "=============================="
echo "Grendel CUDA Build Script"
echo "=============================="

# 1. env check
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 使用gcc11编译。
export CC=gcc-11
export CXX=g++-11
export CUDAHOSTCXX=g++-11

echo "[1] CUDA_HOME = $CUDA_HOME"
nvcc --version

# 2. verify torch
echo "[2] Torch info"
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# 3. build submodules
echo "[3] Build simple-knn"
pip install -e submodules/simple-knn

echo "[4] Build diff-gaussian-rasterization"
pip install -e submodules/diff-gaussian-rasterization

echo "[5] Build gsplat"
pip install -e submodules/gsplat

echo "DONE ✔"