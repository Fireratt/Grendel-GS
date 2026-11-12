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

import os
import sys
import glob
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import utils.general_utils as utils
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    args = utils.get_args()
    cam_infos = []
    utils.print_rank_0("Loading cameras from disk...")
    for idx, key in tqdm(
        enumerate(cam_extrinsics),
        total=len(cam_extrinsics),
        disable=(utils.LOCAL_RANK != 0),
    ):

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            # we're ignoring the 4 distortion
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        if not getattr(args, "fps_test", False):
            with Image.open(image_path) as image:
                width, height = image.size

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=None,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )

        cam_infos.append(cam_info)
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    try:
        colors = (
            np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
        )
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=10):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    if getattr(utils.get_args(), "fps_test", False):
        return SceneInfo(None, train_cam_infos, test_cam_infos, nerf_normalization, ply_path)
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        if utils.GLOBAL_RANK == 0:
            print(
                "Converting point3d.bin to .ply, will happen only the first time you open the scene."
            )
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
            if utils.DEFAULT_GROUP.size() > 1:
                torch.distributed.barrier()
        else:
            if utils.DEFAULT_GROUP.size() > 1:
                torch.distributed.barrier()
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransformsCity(
    path,
    transformsfile,
    random_background,
    white_background,
    extension=".png",
    undistorted=False,
    is_debug=False,
):
    cam_infos = []
    if undistorted:
        print("Undistortion the images!!!")
        # TODO: Support undistortion here. Please refer to octree-gs implementation.
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split(".")[-1] in ["jpg", "jpeg", "JPG", "png"]:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])

        Ts = c2ws[:, :3, 3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"] + extension)
            cam_name = frame["file_path"]
            if not os.path.exists(cam_name):
                print(f"File {cam_name} not found, skipping...")
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            if idx % 10 == 0:
                progress_bar.set_postfix({"num": f"{ct}/{len(frames)}"})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()

            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = cam_name[-17:]  # Path(cam_name).stem
            image = Image.open(image_path)

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=None,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

            # release memory
            image.close()
            image = None

            if is_debug and idx > 50:
                break
    return cam_infos


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name
            image_name = Path(cam_name).stem
            if getattr(utils.get_args(), "fps_test", False):
                width = frame.get("w") or contents.get("w")
                height = frame.get("h") or contents.get("h")
                if not width or not height:
                    raise ValueError(
                        f"FPS-only Blender camera metadata requires positive w/h in {transformsfile}; "
                        "image dimensions cannot be inferred without the source images."
                    )
                width, height = int(width), int(height)
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid camera dimensions in {transformsfile}: {width}x{height}")
            else:
                with Image.open(image_path) as image:
                    width, height = image.size

            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=None,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if getattr(utils.get_args(), "fps_test", False):
        return SceneInfo(None, train_cam_infos, test_cam_infos, nerf_normalization, ply_path)
    
    # ========== 关键修复：删除旧的错误点云缓存 ==========
    # 如果之前已经用错误的 [-1.3, 1.3] 范围生成过 points3d.ply，
    # 必须手动删除它，否则下面的新逻辑不会执行。
    # 首次运行此修复后，后续训练将直接加载正确的新点云。
    if os.path.exists(ply_path):
        # 简单检查：读取前几个点判断是否是旧的错误点云
        try:
            old_pcd = fetchPly(ply_path)
            old_xyz = np.asarray(old_pcd.points)
            if old_xyz.max() <= 1.35 and old_xyz.min() >= -1.35:
                print(f"[WARNING] Detected legacy random point cloud in {ply_path}")
                print(f"          Range: [{old_xyz.min():.3f}, {old_xyz.max():.3f}]")
                print(f"          Deleting to regenerate with scene-aware bounds...")
                os.remove(ply_path)
        except Exception:
            pass
    # =====================================================

    if not os.path.exists(ply_path):
        num_pts = 300_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # ========== 核心修复：在训练相机视锥内采样初始化点云 ==========
        cam_centers = []
        cam_rotations = []
        view_dirs = []
        tan_half_fovx = []
        tan_half_fovy = []

        for cam_info in train_cam_infos:
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = cam_info.R
            w2c[:3, 3] = cam_info.T
            c2w = np.linalg.inv(w2c)

            rotation = c2w[:3, :3].astype(np.float32)
            center = c2w[:3, 3].astype(np.float32)
            view_dir = rotation[:, 2]
            view_dir /= max(np.linalg.norm(view_dir), 1e-8)

            cam_centers.append(center)
            cam_rotations.append(rotation)
            view_dirs.append(view_dir)
            tan_half_fovx.append(np.tan(cam_info.FovX * 0.5))
            tan_half_fovy.append(np.tan(cam_info.FovY * 0.5))

        cam_centers = np.stack(cam_centers, axis=0)
        cam_rotations = np.stack(cam_rotations, axis=0)
        view_dirs = np.stack(view_dirs, axis=0)
        tan_half_fovx = np.asarray(tan_half_fovx, dtype=np.float32)
        tan_half_fovy = np.asarray(tan_half_fovy, dtype=np.float32)

        cam_center_mean = cam_centers.mean(axis=0)
        scene_radius = np.linalg.norm(cam_centers - cam_center_mean, axis=1).max()
        scene_radius = max(float(scene_radius), 1.0)

        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        eye = np.eye(3, dtype=np.float64)
        for center, view_dir in zip(cam_centers, view_dirs):
            projector = eye - np.outer(view_dir, view_dir)
            A += projector
            b += projector @ center

        if np.linalg.matrix_rank(A) == 3:
            focus_point = np.linalg.solve(A, b).astype(np.float32)
        else:
            mean_dir = view_dirs.mean(axis=0)
            mean_dir_norm = np.linalg.norm(mean_dir)
            if mean_dir_norm < 1e-6:
                mean_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                mean_dir = (mean_dir / mean_dir_norm).astype(np.float32)
            focus_point = cam_center_mean + mean_dir * scene_radius

        focus_h = np.concatenate([focus_point, np.array([1.0], dtype=np.float32)])
        depth_min = []
        depth_max = []
        z_signs = []
        for cam_info in train_cam_infos:
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = cam_info.R.transpose()
            w2c[:3, 3] = cam_info.T
            focus_cam = w2c @ focus_h
            focus_depth = abs(float(focus_cam[2]))
            depth_lo = max(focus_depth * 0.5, scene_radius * 0.15)
            depth_hi = max(focus_depth * 1.5, depth_lo + scene_radius * 0.1)
            depth_min.append(depth_lo)
            depth_max.append(depth_hi)
            z_signs.append(1.0 if focus_cam[2] >= 0.0 else -1.0)

        depth_min = np.asarray(depth_min, dtype=np.float32)
        depth_max = np.asarray(depth_max, dtype=np.float32)
        z_signs = np.asarray(z_signs, dtype=np.float32)

        cam_indices = np.random.randint(0, len(train_cam_infos), size=num_pts)
        depths = np.random.uniform(depth_min[cam_indices], depth_max[cam_indices]).astype(np.float32)
        xs = np.random.uniform(-0.8, 0.8, size=num_pts).astype(np.float32) * depths * tan_half_fovx[cam_indices]
        ys = np.random.uniform(-0.8, 0.8, size=num_pts).astype(np.float32) * depths * tan_half_fovy[cam_indices]
        zs = depths * z_signs[cam_indices]
        local_points = np.stack([xs, ys, zs], axis=1)
        xyz = cam_centers[cam_indices] + np.einsum("nij,nj->ni", cam_rotations[cam_indices], local_points)

        print(f"[INIT] Camera centers range: x=[{cam_centers[:,0].min():.2f}, {cam_centers[:,0].max():.2f}] "
              f"y=[{cam_centers[:,1].min():.2f}, {cam_centers[:,1].max():.2f}] "
              f"z=[{cam_centers[:,2].min():.2f}, {cam_centers[:,2].max():.2f}]")
        print(f"[INIT] Focus point: ({focus_point[0]:.2f}, {focus_point[1]:.2f}, {focus_point[2]:.2f})")
        print(f"[INIT] Depth range: [{depth_min.min():.2f}, {depth_max.max():.2f}]")
        print(f"[INIT] Point cloud sample range: x=[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}] "
              f"y=[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}] "
              f"z=[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")
        # ================================================================
        
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCityInfo(
    path,
    random_background,
    white_background,
    extension=".tif",
    llffhold=8,
    undistorted=False,
):

    train_json_path = os.path.join(path, f"transforms_train.json")
    test_json_path = os.path.join(path, f"transforms_test.json")
    print(
        "Reading Training Transforms from {} {}".format(train_json_path, test_json_path)
    )

    train_cam_infos = readCamerasFromTransformsCity(
        path,
        train_json_path,
        random_background,
        white_background,
        extension,
        undistorted,
    )
    test_cam_infos = readCamerasFromTransformsCity(
        path,
        test_json_path,
        random_background,
        white_background,
        extension,
        undistorted,
    )
    print("Load Cameras(train, test): ", len(train_cam_infos), len(test_cam_infos))

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = glob.glob(os.path.join(path, "*.ply"))[0]
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except:
            raise ValueError("must have tiepoints!")
    else:
        assert False, "No ply file found!"

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "City": readCityInfo,
}
