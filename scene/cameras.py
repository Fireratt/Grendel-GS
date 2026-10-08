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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import get_args, get_log_file
import utils.general_utils as utils
import time


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        image_path=None,
        image_width=None,
        image_height=None,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path

        args = get_args()
        log_file = get_log_file()

        if args.time_image_loading:
            start_time = time.time()

        self.stores_image_locally = (
            (
                args.local_sampling
                and args.distributed_dataset_storage
                and utils.GLOBAL_RANK == uid % utils.WORLD_SIZE
            )
            or (
                not args.local_sampling
                and args.distributed_dataset_storage
                and utils.LOCAL_RANK == 0
            )
            or (not args.distributed_dataset_storage)
        )

        if self.stores_image_locally and image is not None:
            self.original_image_backup = image.contiguous()
            if args.preload_dataset_to_gpu:
                self.original_image_backup = self.original_image_backup.to("cuda")
        else:
            self.original_image_backup = None

        if image_width is None or image_height is None:
            fallback_height, fallback_width = utils.get_img_size()
            image_height = fallback_height
            image_width = fallback_width

        self.image_width = image_width
        self.image_height = image_height

        if self.original_image_backup is not None:
            self.image_width = self.original_image_backup.shape[2]
            self.image_height = self.original_image_backup.shape[1]

        if args.time_image_loading:
            log_file.write(f"Image processing in {time.time() - start_time} seconds\n")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.world_view_transform_backup = self.world_view_transform.clone().detach()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def has_local_image_source(self):
        return self.stores_image_locally

    def get_image_backup(self):
        if not self.stores_image_locally:
            return None
        if self.original_image_backup is not None:
            return self.original_image_backup

        args = get_args()
        log_file = get_log_file()
        return utils.load_image_tensor_from_path(
            self.image_path,
            (self.image_width, self.image_height),
            args,
            log_file,
            white_background=args.white_background,
        )

    def get_camera2world(self):
        return self.world_view_transform_backup.t().inverse()

    def update(self, dx, dy, dz):
        # Update the position of this camera pose. TODO: support updating rotation of camera pose.
        with torch.no_grad():
            c2w = self.get_camera2world()
            c2w[0, 3] += dx
            c2w[1, 3] += dy
            c2w[2, 3] += dz

            t_prime = c2w[:3, 3]
            self.T = (-c2w[:3, :3].t() @ t_prime).cpu().numpy()
            # import pdb; pdb.set_trace()

            self.world_view_transform = (
                torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale))
                .transpose(0, 1)
                .cuda()
            )
            self.projection_matrix = (
                getProjectionMatrix(
                    znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
                )
                .transpose(0, 1)
                .cuda()
            )
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
