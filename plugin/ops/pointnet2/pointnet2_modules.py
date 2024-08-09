# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet3d.ops.pointnet_modules.builder import SA_MODULES

from ...utils import sample_points_with_roi
from . import pointnet2_utils


@SA_MODULES.register_module()
class StackSAModuleMSG(BaseModule):
    """Stack point set abstraction module.

    Args:
        in_channels (int): Input channels.
        radius (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify mlp channels of the
            pointnet before the global pooling for each scale to encode
            point features.
        use_xyz (bool): Whether to use xyz. Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        norm_cfg (dict): Type of normalization method. Defaults to
            dict(type='BN2d', eps=1e-5, momentum=0.01).
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 radius: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],
                 use_xyz: bool = True,
                 pool_method='max',
                 norm_cfg: Dict = dict(type='BN2d', eps=1e-5, momentum=0.01),
                 init_cfg: Dict = None,
                 **kwargs) -> None:
        super(StackSAModuleMSG, self).__init__(init_cfg=init_cfg)
        assert len(radius) == len(sample_nums) == len(mlp_channels)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radius)):
            cin = in_channels
            if use_xyz:
                cin += 3
            cur_radius = radius[i]
            nsample = sample_nums[i]
            mlp_spec = mlp_channels[i]

            self.groupers.append(
                pointnet2_utils.QueryAndGroup(cur_radius, nsample, use_xyz=use_xyz))

            mlp = nn.Sequential()
            for i in range(len(mlp_spec)):
                cout = mlp_spec[i]
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        cin,
                        cout,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=False))
                cin = cout
            self.mlps.append(mlp)
        self.pool_method = pool_method

    def forward(
        self,
        xyz: torch.Tensor,
        xyz_batch_cnt: torch.Tensor,
        new_xyz: torch.Tensor,
        new_xyz_batch_cnt: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            xyz (Tensor): Tensor of the xyz coordinates
                of the features shape with (N1 + N2 ..., 3).
            xyz_batch_cnt: (Tensor): Stacked input xyz coordinates nums in
                each batch, just like (N1, N2, ...).
            new_xyz (Tensor): New coords of the outputs shape with
                (M1 + M2 ..., 3).
            new_xyz_batch_cnt: (Tensor): Stacked new xyz coordinates nums
                in each batch, just like (M1, M2, ...).
            features (Tensor, optional): Features of each point with shape
                (N1 + N2 ..., C). C is features channel number. Default: None.

        Returns:
            Return new points coordinates and features:
                - new_xyz  (Tensor): Target points coordinates with shape
                    (N1 + N2 ..., 3).
                - new_features (Tensor): Target points features with shape
                    (M1 + M2 ..., sum_k(mlps[k][-1])).
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            grouped_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)  # (M1 + M2, Cin, nsample)
            grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
            new_features = self.mlps[k](grouped_features) # (M1 + M2 ..., Cout, nsample)

            if self.pool_method == 'max':
                new_features = new_features.max(-1).values
            elif self.pool_method == 'avg':
                new_features = new_features.mean(-1)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)

        return new_xyz, new_features


class VectorPoolLocalInterpolateModule(nn.Module):
    def __init__(self, mlp, num_voxels, max_neighbour_distance, nsample, neighbor_type, use_xyz=True,
                 neighbour_distance_multiplier=1.0, xyz_encoding_type='concat'):
        """
        Args:
            mlp:
            num_voxels:
            max_neighbour_distance:
            neighbor_type: 1: ball, others: cube
            nsample: find all (-1), find limited number(>0)
            use_xyz:
            neighbour_distance_multiplier:
            xyz_encoding_type:
        """
        super().__init__()
        self.num_voxels = num_voxels  # [num_grid_x, num_grid_y, num_grid_z]: number of grids in each local area centered at new_xyz
        self.num_total_grids = self.num_voxels[0] * self.num_voxels[1] * self.num_voxels[2]
        self.max_neighbour_distance = max_neighbour_distance
        self.neighbor_distance_multiplier = neighbour_distance_multiplier
        self.nsample = nsample
        self.neighbor_type = neighbor_type
        self.use_xyz = use_xyz
        self.xyz_encoding_type = xyz_encoding_type

        if mlp is not None:
            if self.use_xyz:
                mlp[0] += 9 if self.xyz_encoding_type == 'concat' else 0
            shared_mlps = []
            for k in range(len(mlp) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp[k + 1]),
                    nn.ReLU()
                ])
            self.mlp = nn.Sequential(*shared_mlps)
        else:
            self.mlp = None

        self.num_avg_length_of_neighbor_idxs = 1000

    def forward(self, support_xyz, support_features, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt):
        """
        Args:
            support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            support_features: (N1 + N2 ..., C) point-wise features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        with torch.no_grad():
            dist, idx, num_avg_length_of_neighbor_idxs = pointnet2_utils.three_nn_for_vector_pool_by_two_step(
                support_xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt,
                self.max_neighbour_distance, self.nsample, self.neighbor_type,
                self.num_avg_length_of_neighbor_idxs, self.num_total_grids, self.neighbor_distance_multiplier
            )
        self.num_avg_length_of_neighbor_idxs = max(self.num_avg_length_of_neighbor_idxs, num_avg_length_of_neighbor_idxs.item())

        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / torch.clamp_min(norm, min=1e-8)

        empty_mask = (idx.view(-1, 3)[:, 0] == -1)
        idx.view(-1, 3)[empty_mask] = 0

        interpolated_feats = pointnet2_utils.three_interpolate(support_features, idx.view(-1, 3), weight.view(-1, 3))
        interpolated_feats = interpolated_feats.view(idx.shape[0], idx.shape[1], -1)  # (M1 + M2 ..., num_total_grids, C)
        if self.use_xyz:
            near_known_xyz = support_xyz[idx.view(-1, 3).long()].view(-1, 3, 3)  # ( (M1 + M2 ...)*num_total_grids, 3)
            local_xyz = (new_xyz_grid_centers.view(-1, 1, 3) - near_known_xyz).view(-1, idx.shape[1], 9)
            if self.xyz_encoding_type == 'concat':
                interpolated_feats = torch.cat((interpolated_feats, local_xyz), dim=-1)  # ( M1 + M2 ..., num_total_grids, 9+C)
            else:
                raise NotImplementedError

        new_features = interpolated_feats.view(-1, interpolated_feats.shape[-1])  # ((M1 + M2 ...) * num_total_grids, C)
        new_features[empty_mask, :] = 0
        if self.mlp is not None:
            new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
            new_features = self.mlp(new_features)

            new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features


class VectorPoolAggregationModule(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 num_local_voxel=(3, 3, 3),
                 local_aggregation_type: str = 'local_interpolation',
                 num_reduced_channels: int = 32,
                 num_channels_of_local_aggregation: int = 32,
                 post_mlps=(128,),
                 max_neighbor_distance=None,
                 neighbor_nsample: int = -1,
                 neighbor_type: int = 0,
                 neighbor_distance_multiplier: int = 2.0):
        super().__init__()
        self.num_local_voxel = num_local_voxel
        self.total_voxels = self.num_local_voxel[0] * self.num_local_voxel[1] * self.num_local_voxel[2]
        self.local_aggregation_type = local_aggregation_type
        assert self.local_aggregation_type in ['local_interpolation', 'voxel_avg_pool', 'voxel_random_choice']
        self.input_channels = input_channels
        self.num_reduced_channels = input_channels if num_reduced_channels is None else num_reduced_channels
        self.num_channels_of_local_aggregation = num_channels_of_local_aggregation
        self.max_neighbour_distance = max_neighbor_distance
        self.neighbor_nsample = neighbor_nsample
        self.neighbor_type = neighbor_type  # 1: ball, others: cube

        if self.local_aggregation_type == 'local_interpolation':
            self.local_interpolate_module = VectorPoolLocalInterpolateModule(
                mlp=None, num_voxels=self.num_local_voxel,
                max_neighbour_distance=self.max_neighbour_distance,
                nsample=self.neighbor_nsample,
                neighbor_type=self.neighbor_type,
                neighbour_distance_multiplier=neighbor_distance_multiplier,
            )
            num_c_in = (self.num_reduced_channels + 9) * self.total_voxels
        else:
            self.local_interpolate_module = None
            num_c_in = (self.num_reduced_channels + 3) * self.total_voxels

        num_c_out = self.total_voxels * self.num_channels_of_local_aggregation

        self.separate_local_aggregation_layer = nn.Sequential(
            nn.Conv1d(num_c_in, num_c_out, kernel_size=1, groups=self.total_voxels, bias=False),
            nn.BatchNorm1d(num_c_out),
            nn.ReLU()
        )

        post_mlp_list = []
        c_in = num_c_out
        for cur_num_c in post_mlps:
            post_mlp_list.extend([
                nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(cur_num_c),
                nn.ReLU()
            ])
            c_in = cur_num_c
        self.post_mlps = nn.Sequential(*post_mlp_list)

        self.num_mean_points_per_grid = 20
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def extra_repr(self) -> str:
        ret = f'radius={self.max_neighbour_distance}, local_voxels=({self.num_local_voxel}, ' \
              f'local_aggregation_type={self.local_aggregation_type}, ' \
              f'num_c_reduction={self.input_channels}->{self.num_reduced_channels}, ' \
              f'num_c_local_aggregation={self.num_channels_of_local_aggregation}'
        return ret

    def vector_pool_with_voxel_query(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        use_xyz = 1
        pooling_type = 0 if self.local_aggregation_type == 'voxel_avg_pool' else 1

        new_features, new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid = pointnet2_utils.vector_pool_with_voxel_query_op(
            xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt,
            self.num_local_voxel[0], self.num_local_voxel[1], self.num_local_voxel[2],
            self.max_neighbour_distance, self.num_reduced_channels, use_xyz,
            self.num_mean_points_per_grid, self.neighbor_nsample, self.neighbor_type,
            pooling_type
        )
        self.num_mean_points_per_grid = max(self.num_mean_points_per_grid, num_mean_points_per_grid.item())

        num_new_pts = new_features.shape[0]
        new_local_xyz = new_local_xyz.view(num_new_pts, -1, 3)  # (N, num_voxel, 3)
        new_features = new_features.view(num_new_pts, -1, self.num_reduced_channels)  # (N, num_voxel, C)
        new_features = torch.cat((new_local_xyz, new_features), dim=-1).view(num_new_pts, -1)

        return new_features, point_cnt_of_grid

    @staticmethod
    def get_dense_voxels_by_center(point_centers, max_neighbour_distance, num_voxels):
        """
        Args:
            point_centers: (N, 3)
            max_neighbour_distance: float
            num_voxels: [num_x, num_y, num_z]

        Returns:
            voxel_centers: (N, total_voxels, 3)
        """
        R = max_neighbour_distance
        device = point_centers.device
        x_grids = torch.arange(-R + R / num_voxels[0], R - R / num_voxels[0] + 1e-5, 2 * R / num_voxels[0], device=device)
        y_grids = torch.arange(-R + R / num_voxels[1], R - R / num_voxels[1] + 1e-5, 2 * R / num_voxels[1], device=device)
        z_grids = torch.arange(-R + R / num_voxels[2], R - R / num_voxels[2] + 1e-5, 2 * R / num_voxels[2], device=device)
        x_offset, y_offset, z_offset = torch.meshgrid(x_grids, y_grids, z_grids, indexing='ij')  # shape: [num_x, num_y, num_z]
        xyz_offset = torch.cat((
            x_offset.contiguous().view(-1, 1),
            y_offset.contiguous().view(-1, 1),
            z_offset.contiguous().view(-1, 1)), dim=-1
        )
        voxel_centers = point_centers[:, None, :] + xyz_offset[None, :, :]
        return voxel_centers

    def vector_pool_with_local_interpolate(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        """
        Args:
            xyz: (N, 3)
            xyz_batch_cnt: (batch_size)
            features: (N, C)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size)
        Returns:
            new_features: (M, total_voxels * C)
        """
        voxel_centers = self.get_dense_voxels_by_center(
            point_centers=new_xyz, max_neighbour_distance=self.max_neighbour_distance, num_voxels=self.num_local_voxel
        )  # (M1 + M2 + ..., total_voxels, 3)
        voxel_features = self.local_interpolate_module.forward(
            support_xyz=xyz, support_features=features, xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz, new_xyz_grid_centers=voxel_centers, new_xyz_batch_cnt=new_xyz_batch_cnt
        )  # ((M1 + M2 ...) * total_voxels, C)

        voxel_features = voxel_features.contiguous().view(-1, self.total_voxels * voxel_features.shape[-1])
        return voxel_features

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features, **kwargs):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        N, C = features.shape
        assert C % self.num_reduced_channels == 0, \
            f'the input channels ({C}) should be an integral multiple of num_reduced_channels({self.num_reduced_channels})'

        features = features.view(N, -1, self.num_reduced_channels).sum(dim=1)

        if self.local_aggregation_type in ['voxel_avg_pool', 'voxel_random_choice']:
            vector_features, point_cnt_of_grid = self.vector_pool_with_voxel_query(
                xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features,
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )
        elif self.local_aggregation_type == 'local_interpolation':
            vector_features = self.vector_pool_with_local_interpolate(
                xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features,
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )  # (M1 + M2 + ..., total_voxels * C)
        else:
            raise NotImplementedError

        vector_features = vector_features.permute(1, 0)[None, :, :]  # (1, num_voxels * C, M1 + M2 ...)

        new_features = self.separate_local_aggregation_layer(vector_features)

        new_features = self.post_mlps(new_features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)
        return new_xyz, new_features


@SA_MODULES.register_module()
class VectorPoolAggregationModuleMSG(BaseModule):
    def __init__(self,
                 in_channels: int = 1,
                 local_aggregation_type: str = 'local_interpolation',
                 num_reduced_channels = 2,
                 num_channels_of_local_aggregation = 32,
                 msg_post_mlps: Tuple[int] = (32),
                 radius_of_neighbor_with_roi: int = None,
                 group_cfg: List[Dict] = [dict(num_local_voxel = [2, 2, 2],
                                               max_neighbor_distance = 0.2,
                                               neighbor_nsample = -1,
                                               post_mlps = [32, 32])],
                 **kwargs):
        super(VectorPoolAggregationModuleMSG, self).__init__()
        self.radius_of_neighbor_with_roi = radius_of_neighbor_with_roi
        self.num_groups = len(group_cfg)
        self.layers = []

        c_in = 0
        for k, cur_config in enumerate(group_cfg):
            cur_vector_pool_module = VectorPoolAggregationModule(
                input_channels=in_channels,
                num_local_voxel=cur_config['num_local_voxel'],
                post_mlps=cur_config['post_mlps'],
                max_neighbor_distance=cur_config['max_neighbor_distance'],
                neighbor_nsample=cur_config['neighbor_nsample'],
                local_aggregation_type=local_aggregation_type,
                num_reduced_channels=num_reduced_channels,
                num_channels_of_local_aggregation=num_channels_of_local_aggregation,
                neighbor_distance_multiplier=2.0
            )
            self.__setattr__(f'layer_{k}', cur_vector_pool_module)
            c_in += cur_config['post_mlps'][-1]

        c_in += 3  # use_xyz

        shared_mlps = []
        for cur_num_c in msg_post_mlps:
            shared_mlps.extend([
                nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(cur_num_c),
                nn.ReLU()
            ])
            c_in = cur_num_c
        self.msg_post_mlps = nn.Sequential(*shared_mlps)

    def forward(self,
                xyz: torch.Tensor,
                xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor,
                new_xyz_batch_cnt: torch.Tensor,
                features: Optional[torch.Tensor] = None,
                rois: List[torch.Tensor] = None):

        if self.radius_of_neighbor_with_roi is not None:
            sample_xyz, sample_features = [], []
            cnt_start = 0
            for bs_idx, batch_rois in enumerate(rois):
                batch_xyz = xyz[cnt_start:cnt_start + xyz_batch_cnt[bs_idx]]
                _, valid_mask = sample_points_with_roi(
                    rois=batch_rois, points=batch_xyz,
                    sample_radius_with_roi=self.radius_of_neighbor_with_roi,
                    num_max_points_of_part=200000)

                sample_xyz.append(batch_xyz[valid_mask])
                if sample_features is not None:
                    sample_features.append(features[cnt_start:cnt_start + xyz_batch_cnt[bs_idx]][valid_mask])
                cnt_start += xyz_batch_cnt[bs_idx]
                xyz_batch_cnt[bs_idx] = valid_mask.sum()

            xyz = torch.cat(sample_xyz, dim=0)
            features = None if features is None else torch.cat(sample_features, dim=0)

        features_list = []
        for k in range(self.num_groups):
            cur_xyz, cur_features = self.__getattr__(f'layer_{k}')(
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
            features_list.append(cur_features)

        features = torch.cat(features_list, dim=-1)
        features = torch.cat((cur_xyz, features), dim=-1)
        features = features.permute(1, 0)[None, :, :]  # (1, C, N)
        new_features = self.msg_post_mlps(features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)  # (N, C)

        return cur_xyz, new_features
