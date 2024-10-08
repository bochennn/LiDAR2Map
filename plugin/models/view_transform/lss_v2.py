# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import BaseModule
from ...ops.bev_pool_v2.bev_pool import bev_pool_v2
from torch.utils.checkpoint import checkpoint

from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class LSSViewTransformerV2(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample: int = 16,
        in_channels: int = 512,
        out_channels: int = 64,
        accelerate: bool = False,
        sid: bool = False,
        collapse_z: bool = True,
        with_cp: bool = False,
        with_depth_from_lidar: bool = False,
    ):
        super(LSSViewTransformerV2, self).__init__()
        self.with_cp = with_cp
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z
        self.with_depth_from_lidar = with_depth_from_lidar
        if self.with_depth_from_lidar:
            self.lidar_input_net = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=int(2 * self.downsample / 8),
                          padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True))
            out_channels = self.D + self.out_channels
            self.depth_net = nn.Sequential(
                nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, 1))

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        return torch.stack((x, y, d), -1)

    def get_lidar_coor(
        self,
        cam2lidar: torch.Tensor,
        cam2imgs: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        bda: torch.Tensor = None
    ):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = cam2lidar.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(cam2lidar) - img_aug_matrix[..., :3, 3].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(img_aug_matrix[..., :3, :3]).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = cam2lidar[..., :3, :3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += cam2lidar[..., :3, 3].view(B, N, 1, 1, 1, 3)
        # points = bda[:, :3, :3].view(B, 1, 1, 1, 1, 3, 3).matmul(
        #     points.unsqueeze(-1)).squeeze(-1)
        # points += bda[:, :3, 3].view(B, 1, 1, 1, 1, 3)
        return points

    def convert_metas(self, template: torch.Tensor, img_metas: List):
        """ """
        cam2lidar = template.new_tensor(np.stack([m['lidar2cam'] for m in img_metas]))
        cam2imgs = template.new_tensor(np.stack([m['cam2img'] for m in img_metas]))
        img_aug_matrix = template.new_tensor(np.stack([m['img_aug_matrix'] for m in img_metas]))
        return cam2lidar, cam2imgs, img_aug_matrix

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.arange(
            0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(
            0, num_points // D, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.arange(0, B).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, template, img_metas):
        if self.initial_flag:
            coor = self.get_lidar_coor(*self.convert_metas(template, img_metas))
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, depth, tran_feat, img_metas):
        _, C, H, W = tran_feat.shape
        B = len(img_metas)

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, -1, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, -1, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]), C)  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_lidar_coor(*self.convert_metas(tran_feat, img_metas))
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, -1, self.D, H, W),
                tran_feat.view(B, -1, self.out_channels, H, W))
        return bev_feat, depth

    def view_transform(
        self,
        depth: torch.Tensor,
        tran_feat: torch.Tensor,
        img_metas: List
    ):
        for shape_id in range(3):
            assert depth.shape[shape_id+1] == self.frustum.shape[shape_id]
        if self.accelerate:
            self.pre_compute(tran_feat, img_metas)
        return self.view_transform_core(depth, tran_feat, img_metas)

    def forward(
        self,
        lvl_img_feats: List[torch.Tensor],
        img_metas: List,
        depth_from_lidar=None
    ):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        img_feats = lvl_img_feats[0]

        if self.with_depth_from_lidar:
            assert depth_from_lidar is not None
            if isinstance(depth_from_lidar, list):
                assert len(depth_from_lidar) == 1
                depth_from_lidar = depth_from_lidar[0]
            h_img, w_img = depth_from_lidar.shape[2:]
            depth_from_lidar = depth_from_lidar.view(img_feats.shape[0], 1, h_img, w_img)
            depth_from_lidar = self.lidar_input_net(depth_from_lidar)
            img_feats = torch.cat([img_feats, depth_from_lidar], dim=1)
        if self.with_cp:
            img_feats = checkpoint(self.depth_net, img_feats)
        else:
            img_feats = self.depth_net(img_feats)

        depth_digit = img_feats[:, :self.D, ...]
        tran_feat = img_feats[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(depth, tran_feat, img_metas)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None

