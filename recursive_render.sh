torchrun --standalone --nnodes=1 --nproc-per-node=2 recursive_render.py \
/home/ps/hjf/3dgs/res/training/building \
/home/ps/hjf/3dgs/res/rubble-colmap/output/grendel_all \
--iteration 70000 \
--resolutions 3840x2160 \
--no-save-renders \
2>&1 | tee building_fps.log

torchrun --standalone --nnodes=1 --nproc-per-node=2 recursive_render.py \
/home/ps/hjf/3dgs/res/rubble-colmap \
/home/ps/hjf/3dgs/res/rubble-colmap/output/grendel_all \
--iteration 70000 \
--resolutions 3840x2160 \
2>&1 | tee building_fps.log

TARGET_HOME=/home/ps/hjf/3dgs/res/training/tanksandtemples/truck

torchrun --standalone --nnodes=1 --nproc-per-node=1 recursive_render.py \
${TARGET_HOME} ${TARGET_HOME}/trained \
--iteration 29997 \
--resolutions 3840x2160 \
--no-save-renders

TARGET_HOME=/home/ps/hjf/3dgs/res/training/tanksandtemples/train

torchrun --standalone --nnodes=1 --nproc-per-node=1 recursive_render.py \
${TARGET_HOME} ${TARGET_HOME}/trained \
--iteration 29997 \
--resolutions 3840x2160 \
--no-save-renders

TARGET_HOME=/home/ps/hjf/3dgs/res/training/deep-blending/drjohnson

torchrun --standalone --nnodes=1 --nproc-per-node=1 recursive_render.py \
${TARGET_HOME} ${TARGET_HOME}/trained \
--iteration 30000 \
--resolutions 3840x2160 \
--no-save-renders

TARGET_HOME=/home/ps/hjf/3dgs/res/training/deep-blending/playroom

torchrun --standalone --nnodes=1 --nproc-per-node=1 recursive_render.py \
${TARGET_HOME} ${TARGET_HOME}/trained \
--iteration 30000 \
--resolutions 3840x2160 \
--no-save-renders


# torchrun --standalone --nnodes=1 --nproc-per-node=2 \
# recursive_render.py \
# ${TARGET_HOME} ${TARGET_HOME}/trained \
# --iteration 29997 \
# --resolutions 1.0 0.5 3840x2160 \
# --no-save-renders
