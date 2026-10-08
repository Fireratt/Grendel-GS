#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import get_args, get_log_file, load_image_tensor_from_path
import utils.general_utils as utils
from tqdm import tqdm
from utils.graphics_utils import fov2focal
import time
import multiprocessing
from multiprocessing import shared_memory
import torch
from PIL import Image


WARNED = False


def resolve_resolution(args, orig_w, orig_h, resolution_scale=1.0):
    """Resolve the image size used for camera tensors and training images."""
    global WARNED

    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K "
                        "pixels width), rescaling to 1.6K.\n"
                        "If this is not desired, please explicitly specify "
                        "'--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return max(1, resolution[0]), max(1, resolution[1])


def loadCam(
    args,
    id,
    cam_info,
    decompressed_image=None,
    return_image=False,
    resolution_scale=1.0,
):
    orig_w, orig_h = cam_info.width, cam_info.height

    args = get_args()
    log_file = get_log_file()
    resolution = resolve_resolution(args, orig_w, orig_h, resolution_scale)

    # may use cam_info.uid
    if (
        (
            args.local_sampling
            and args.distributed_dataset_storage
            and utils.GLOBAL_RANK == id % utils.WORLD_SIZE
        )
        or (
            not args.local_sampling
            and args.distributed_dataset_storage
            and utils.LOCAL_RANK == 0
        )
        or (not args.distributed_dataset_storage)
    ):
        gt_image = None
        if (not args.lazy_image_loading) or return_image:
            if args.time_image_loading:
                start_time = time.time()
            resized_image_rgb = load_image_tensor_from_path(
                cam_info.image_path,
                resolution,
                args,
                log_file,
                white_background=args.white_background,
                decompressed_image=decompressed_image,
            )
            if args.time_image_loading:
                log_file.write(
                    f"PILtoTorch image in {time.time() - start_time} seconds\n"
                )

            gt_image = resized_image_rgb[:3, ...].contiguous()
        loaded_mask = None
    else:
        gt_image = None
        loaded_mask = None

    if return_image:
        return gt_image

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        image_path=cam_info.image_path,
        image_width=resolution[0],
        image_height=resolution[1],
    )


def load_decompressed_image(params):
    args, id, cam_info, resolution_scale = params
    return loadCam(
        args,
        id,
        cam_info,
        decompressed_image=None,
        return_image=True,
        resolution_scale=resolution_scale,
    )


# Modify this code to support shared_memory.SharedMemory to make inter-process communication faster
def decompressed_images_from_camInfos_multiprocess(
    cam_infos, args, resolution_scale=1.0
):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Create a pool of processes
    with multiprocessing.Pool(processes=2) as pool:
        # Prepare data for processing
        tasks = [
            (args, id, cam_info, resolution_scale)
            for id, cam_info in enumerate(cam_infos)
        ]

        # Map load_camera_data to the tasks
        # results = pool.map(load_decompressed_image, tasks)
        results = list(
            tqdm(
                pool.imap(load_decompressed_image, tasks),
                total=total_cameras,
                disable=(utils.LOCAL_RANK != 0),
            )
        )

        for id, result in enumerate(results):
            decompressed_images.append(result)

    return decompressed_images


def decompress_and_scale_image(cam_info, resolution_scale=1.0):
    args = get_args()
    log_file = get_log_file()
    resolution = resolve_resolution(
        args, cam_info.width, cam_info.height, resolution_scale
    )
    return (
        load_image_tensor_from_path(
            cam_info.image_path,
            resolution,
            args,
            log_file,
            white_background=args.white_background,
        )
        .cpu()
        .numpy()
    )


def load_decompressed_image_shared(params):
    shared_mem_name, args, id, cam_info, resolution_scale = params
    # Retrieve the shared memory block
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)

    # Assume each image will be stored as a flat array in shared memory
    # Example: using numpy for manipulation; adjust size and dtype as needed
    resolution_width, resolution_height = resolve_resolution(
        args, cam_info.width, cam_info.height, resolution_scale
    )
    image_shape = (3, resolution_height, resolution_width)  # Set appropriate values
    dtype = np.uint8  # Adjust as per your image data type

    # Calculate the offset for this particular image
    offset = id * np.prod(image_shape)
    np_image_array = np.ndarray(
        image_shape, dtype=dtype, buffer=existing_shm.buf, offset=offset
    )

    # Decompress image into the numpy array directly
    decompressed_image = decompress_and_scale_image(
        cam_info, resolution_scale
    )  # Implement this
    np_image_array[:] = decompressed_image

    # Clean up
    existing_shm.close()


def decompressed_images_from_camInfos_multiprocess_sharedmem(
    cam_infos, resolution_scale, args
):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Assume each image shape and dtype
    resolution_width, resolution_height = resolve_resolution(
        args,
        cam_infos[0].width,
        cam_infos[0].height,
        resolution_scale,
    )
    image_shape = (
        3,
        resolution_height,
        resolution_width,
    )  # Define these as per your data
    dtype = np.uint8
    image_size = np.prod(image_shape) * np.dtype(dtype).itemsize

    # Create shared memory
    total_size = image_size * total_cameras
    shm = shared_memory.SharedMemory(create=True, size=total_size)

    # Create a pool of processes
    with multiprocessing.Pool(16) as pool:
        # Prepare data for processing
        tasks = [
            (shm.name, args, id, cam_info, resolution_scale)
            for id, cam_info in enumerate(cam_infos)
        ]

        # print("Start Parallel loading...")
        # Map load_camera_data to the tasks
        list(
            tqdm(pool.imap(load_decompressed_image_shared, tasks), total=total_cameras)
        )

    # Read images from shared memory
    decompressed_images = []
    for id in range(total_cameras):
        offset = id * np.prod(image_shape)
        np_image_array = np.ndarray(
            image_shape, dtype=dtype, buffer=shm.buf, offset=offset
        )
        decompressed_images.append(
            torch.from_numpy(np_image_array)
        )  # Make a copy if necessary

    # Clean up shared memory
    shm.close()
    shm.unlink()

    return decompressed_images


def cameraList_from_camInfos(cam_infos, args, resolution_scale=1.0):
    args = get_args()

    if args.multiprocesses_image_loading and not args.lazy_image_loading:
        decompressed_images = decompressed_images_from_camInfos_multiprocess(
            cam_infos, args, resolution_scale
        )
        # decompressed_images = decompressed_images_from_camInfos_multiprocess_sharedmem(cam_infos, resolution_scale, args)
    else:
        decompressed_images = [None for _ in cam_infos]

    camera_list = []
    for id, c in tqdm(
        enumerate(cam_infos), total=len(cam_infos), disable=(utils.LOCAL_RANK != 0)
    ):
        camera_list.append(
            loadCam(
                args,
                id,
                c,
                decompressed_image=decompressed_images[id],
                return_image=False,
                resolution_scale=resolution_scale,
            )
        )

    if utils.DEFAULT_GROUP.size() > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
