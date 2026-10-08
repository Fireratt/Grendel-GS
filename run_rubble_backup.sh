# 该文件用于封存rubble的训练参数，用于实验环境说明。
RUBBLE_HOME=/home/ps/hjf/3dgs/res/rubble-colmap
TOTAL_ITERATION=60000
torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py \
    -s ${RUBBLE_HOME} \
    -m ${RUBBLE_HOME}/output/grendel_all \
    --eval \
    --bsz 4 \
    --backend gsplat \
    --iterations ${TOTAL_ITERATION} \
    --test_iterations 7000 15000 30000 40000 50000\
    --save_iterations 10000 20000 30000 40000 50000\
    --checkpoint_iterations 10000 20000 30000 40000\
    --densify_until_iter 50000 \
    --densify_grad_threshold 0.0001 \
    --densification_interval 100 \
    --opacity_reset_interval 5000 \
    --lambda_dssim 0.2 \
    --lr_scale_mode sqrt \
    --image_loading_mode lazy
# python render.py -s ${RUBBLE_HOME} --model_path ${RUBBLE_HOME}/output/grendel_all 
# python metrics.py --model_path ${RUBBLE_HOME}/output/grendel_all 