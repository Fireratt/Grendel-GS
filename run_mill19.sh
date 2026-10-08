RUBBLE_HOME=/home/ps/hjf/3dgs/res/rubble-colmap
BUILDING_HOME=/home/ps/hjf/3dgs/res/training/building
MODEL_HOME=${RUBBLE_HOME}
TOTAL_ITERATION=70000
torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py \
    -s ${MODEL_HOME} \
    -m ${MODEL_HOME}/output/grendel_all \
    --eval \
    --bsz 1 \
    --backend gsplat \
    --iterations ${TOTAL_ITERATION} \
    --test_iterations 10000 20000 30000 40000 60000 80000 100000 \
    --save_iterations 20000 40000 50000 60000 \
    --checkpoint_iterations 15000 \
    --densify_until_iter 25000 \
    --densify_grad_threshold 0.0001\
    --densification_interval 100 \
    --opacity_reset_interval 5000 \
    --min_opacity 0.001 \
    --prune_scale3d 0.2 \
    --lambda_dssim 0.2 \
    --percent_dense 0.01 \
    --resolution -1 \
    --lr_scale_mode sqrt \
    --image_loading_mode lazy

python render.py -s ${MODEL_HOME} --model_path ${MODEL_HOME}/output/grendel_all --iterations ${TOTAL_ITERATION}
python metrics.py --model_path ${MODEL_HOME}/output/grendel_all 
