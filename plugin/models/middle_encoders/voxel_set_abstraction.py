# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.ops import SparseConvTensor
from mmcv.runner import BaseModule
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.models.builder import MIDDLE_ENCODERS
from mmdet3d.ops.pointnet_modules.builder import build_sa_module

from ...ops.pointnet2 import pointnet2_utils
from ...utils import bilinear_interpolate_torch, sample_points_with_roi


@MIDDLE_ENCODERS.register_module()
class VoxelSetAbstraction(BaseModule):
    """Voxel set abstraction module for PVRCNN and PVRCNN++.

    Args:
        num_keypoints (int): The number of key points sampled from
            raw points cloud.
        fused_out_channel (int): Key points feature output channels
            num after fused. Default to 128.
        voxel_size (list[float]): Size of voxels. Defaults to
            [0.05, 0.05, 0.1].
        point_cloud_range (list[float]): Point cloud range. Defaults to
            [0, -40, -3, 70.4, 40, 1].
        voxel_sa_cfgs_list (List[dict or ConfigDict], optional): List of SA
            module cfg. Used to gather key points features from multi-wise
            voxel features. Default to None.
        rawpoints_sa_cfgs (dict or ConfigDict, optional): SA module cfg.
            Used to gather key points features from raw points. Default to
            None.
        bev_feat_channel (int): Bev features channels num.
            Default to 256.
        bev_scale_factor (int): Bev features scale factor. Default to 8.
        voxel_center_as_source (bool): Whether used voxel centers as points
            cloud key points. Defaults to False.
        norm_cfg (dict[str]): Config of normalization layer. Default
            used dict(type='BN1d', eps=1e-5, momentum=0.1).
        bias (bool | str, optional): If specified as `auto`, it will be
            decided by `norm_cfg`. `bias` will be set as True if
            `norm_cfg` is None, otherwise False. Default: 'auto'.
    """

    def __init__(self,
                 num_keypoints: int,
                 fused_out_channel: int = 128,
                 voxel_size: list = [0.05, 0.05, 0.1],
                 point_cloud_range: list = [0, -40, -3, 70.4, 40, 1],
                 voxel_sa_cfgs_list: Optional[list] = None,
                 rawpoints_sa_cfgs: Optional[dict] = None,
                 bev_feat_channel: int = 256,
                 bev_scale_factor: int = 8,
                 voxel_center_as_source: bool = False,
                 norm_cfg: dict = dict(type='BN2d', eps=1e-5, momentum=0.1),
                 bias: str = 'auto',
                 sample_cfg: Dict = dict(method='FPS')) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.fused_out_channel = fused_out_channel
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_center_as_source = voxel_center_as_source
        self.sample_cfg = sample_cfg

        gathered_channel = 0
        if bev_feat_channel is not None and bev_scale_factor is not None:
            self.bev_cfg = Config(
                dict(bev_feat_channels=bev_feat_channel,
                     bev_scale_factor=bev_scale_factor))
            gathered_channel += bev_feat_channel
        else:
            self.bev_cfg = None

        if rawpoints_sa_cfgs is not None:
            self.rawpoints_sa_layer = build_sa_module(rawpoints_sa_cfgs)
            # gathered_channel += sum([x[-1] for x in rawpoints_sa_cfgs.mlp_channels])
            gathered_channel += rawpoints_sa_cfgs.msg_post_mlps[-1]
        else:
            self.rawpoints_sa_layer = None

        if voxel_sa_cfgs_list is not None:
            self.voxel_sa_configs_list = voxel_sa_cfgs_list
            self.voxel_sa_layers = nn.ModuleList()
            for voxel_sa_config in voxel_sa_cfgs_list:
                cur_layer = build_sa_module(voxel_sa_config)
                self.voxel_sa_layers.append(cur_layer)
                # gathered_channel += sum([x[-1] for x in voxel_sa_config.mlp_channels])
                gathered_channel += voxel_sa_config.msg_post_mlps[-1]
        else:
            self.voxel_sa_layers = None

        self.point_feature_fusion_layer = nn.Sequential(
            nn.Linear(gathered_channel, fused_out_channel, bias=False),
            nn.BatchNorm1d(fused_out_channel),
            nn.ReLU(),
        )

    def interpolate_from_bev_features(self, keypoints: torch.Tensor, keypoints_cnt: torch.Tensor,
                                      bev_features: torch.Tensor) -> torch.Tensor:
        """Gather key points features from bev feature map by interpolate.

        Args:
            keypoints (torch.Tensor): Sampled key points with shape
                (N1 + N2 + ..., NDim).
            bev_features (torch.Tensor): Bev feature map from the first
                stage with shape (B, C, H, W).

        Returns:
            torch.Tensor: Key points features gather from bev feature
                map with shape (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / self.bev_cfg.bev_scale_factor
        y_idxs = y_idxs / self.bev_cfg.bev_scale_factor

        point_bev_features_list = []
        cnt_start = 0
        for k, cur_bev_features in enumerate(bev_features):
            cur_x_idxs = x_idxs[cnt_start:cnt_start + keypoints_cnt[k], ...]
            cur_y_idxs = y_idxs[cnt_start:cnt_start + keypoints_cnt[k], ...]
            point_bev_features = bilinear_interpolate_torch(cur_bev_features.permute(1, 2, 0), cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)
            cnt_start += keypoints_cnt[k]

        return torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)

    def get_voxel_centers(self, coors: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Get voxel centers coordinate.

        Args:
            coors (torch.Tensor): Coordinates of voxels shape is Nx(1+NDim),
                where 1 represents the batch index.
            scale_factor (float): Scale factor.

        Returns:
            torch.Tensor: Voxel centers coordinate with shape (N, 3).
        """
        assert coors.shape[1] == 4
        voxel_centers = coors[:, [3, 2, 1]].float()  # (xyz)
        voxel_size = torch.tensor(self.voxel_size, device=voxel_centers.device).float() * scale_factor
        pc_range = torch.tensor(self.point_cloud_range[:3], device=voxel_centers.device).float()
        voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range

        xyz_batch_cnt = coors.new_zeros(coors[:, 0].max() + 1).int()
        for bs_idx, _ in enumerate(xyz_batch_cnt):
            xyz_batch_cnt[bs_idx] = (coors[:, 0] == bs_idx).sum()
        
        if coors[:xyz_batch_cnt[0], 0].sum() != 0:
            print(uuuuuuu)

        return voxel_centers, xyz_batch_cnt

    def sectorized_proposal_centric_sampling(self, points: torch.Tensor, roi_boxes: torch.Tensor):
        """
        Args:
            points: (N, 3)
            roi_boxes: (M, 7 + C)

        Returns:
            sampled_points: (N_out, 3)
        """
        sampled_points, _ = sample_points_with_roi(
            rois=roi_boxes, points=points,
            sample_radius_with_roi=self.sample_cfg.sample_radius_with_roi,
            num_max_points_of_part=self.sample_cfg.get('num_points_of_each_sample_part', 200000))

        sampled_points = pointnet2_utils.sector_fps(
            points=sampled_points, num_sampled_points=self.num_keypoints,
            num_sectors=self.sample_cfg.num_sectors)
        return sampled_points

    def sample_key_points(self, points: List[torch.Tensor], coors: torch.Tensor, rois: List[torch.Tensor]) -> torch.Tensor:
        """Sample key points from raw points cloud.

        Args:
            points (List[torch.Tensor]): Point cloud of each sample.
            coors (torch.Tensor): Coordinates of voxels shape is Nx(1+NDim),
                where 1 represents the batch index.

        Returns:
            torch.Tensor: (B, M, 3) Key points of each sample.
                M is num_keypoints.
        """
        assert points is not None or coors is not None
        if self.voxel_center_as_source:
            _src_points, _ = self.get_voxel_centers(coors=coors, scale_factor=1)
            batch_size = coors[-1, 0].item() + 1
            src_points = [_src_points[coors[:, 0] == b] for b in range(batch_size)]
        else:
            batch_size = len(points)
            src_points = [p[..., :3] for p in points]

        keypoints_list, keypoints_cnt = [], []
        for bs_idx, points_to_sample in enumerate(src_points):
            if self.sample_cfg.method == 'FPS':
                cur_pt_idxs = pointnet2_utils.furthest_point_sample(
                    points_to_sample.unsqueeze(0).contiguous(), self.num_keypoints).long()[0]

                num_points = points_to_sample.shape[0]
                if num_points < self.num_keypoints:
                    times = int(self.num_keypoints / num_points) + 1
                    non_empty = cur_pt_idxs[:num_points]
                    cur_pt_idxs = non_empty.repeat(times)[:self.num_keypoints]

                keypoints = points_to_sample[cur_pt_idxs]

            elif self.sample_cfg.method == 'SPC':
                keypoints = self.sectorized_proposal_centric_sampling(
                    points_to_sample, rois[bs_idx])
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)
            keypoints_cnt.append(keypoints.shape[0])
        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)

        return keypoints, keypoints.new_tensor(keypoints_cnt).int()

    def forward(
        self,
        points: List[torch.Tensor] = None,
        bev_feature: List[torch.Tensor] = None,
        voxel_feature: List[SparseConvTensor] = None,
        voxel_coords = None,
        rpn_results_list: List[LiDARInstance3DBoxes] = None,
    ) -> Dict:
        """Extract point-wise features from multi-input.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.
            feats_dict (dict): Contains features from the first
                stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.

        Returns:
            dict: Contain Point-wise features, include:
                - keypoints (torch.Tensor): Sampled key points.
                - keypoint_features (torch.Tensor): Gathered key points
                    features from multi input.
                - fusion_keypoint_features (torch.Tensor): Fusion
                    keypoint_features by point_feature_fusion_layer.
        """
        if isinstance(bev_feature, List):
            bev_feature = bev_feature[0]

        rpn_rois = []
        if rpn_results_list is not None:
            for rpn_results in rpn_results_list:
                roi_boxes = rpn_results[0].tensor.clone()
                roi_boxes[:, :3] = rpn_results[0].gravity_center
                rpn_rois.append(roi_boxes)

        if not self.voxel_center_as_source:
            voxels_coors = None
        key_xyz, key_xyz_cnt = self.sample_key_points(points, voxels_coors, rpn_rois)    # [N1 + N2 + ..., D]

        point_features_list = []
        if self.bev_cfg is not None:
            point_bev_features = self.interpolate_from_bev_features(key_xyz, key_xyz_cnt, bev_feature)
            point_features_list.append(point_bev_features)

        if self.rawpoints_sa_layer is not None:
            batch_points = torch.cat(points, dim=0) # stacked pts [N1 + N2 + ..., D]
            raw_xyz = batch_points[:, :3].contiguous()
            raw_pts_feature = batch_points[:, 3:].contiguous() if batch_points.shape[1] > 3 else None
            raw_xyz_cnt = raw_xyz.new_tensor([len(p) for p in points]).int()

            _, pooled_features = self.rawpoints_sa_layer(
                xyz=raw_xyz,
                xyz_batch_cnt=raw_xyz_cnt,
                new_xyz=key_xyz,
                new_xyz_batch_cnt=key_xyz_cnt,
                features=raw_pts_feature,
                rois=rpn_rois)
            point_features_list.append(pooled_features)

        if self.voxel_sa_layers is not None:
            for k, voxel_sa_layer in enumerate(self.voxel_sa_layers):
                # indices not consecutive order in spconv
                cur_coords = voxel_feature[k].indices
                cur_features = voxel_feature[k].features.contiguous()

                xyz, xyz_cnt = self.get_voxel_centers(
                    coors=cur_coords,
                    scale_factor=self.voxel_sa_configs_list[k].downsample_factor)

                print(cur_coords)
                print(cur_features)
                print(xyz)
                print(xyz_cnt)
                _, pooled_features = voxel_sa_layer(
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_cnt,
                    new_xyz=key_xyz,
                    new_xyz_batch_cnt=key_xyz_cnt,
                    features=cur_features,
                    rois=rpn_rois)
                point_features_list.append(pooled_features)

        point_features = torch.cat(point_features_list, dim=-1)
        fusion_point_features = self.point_feature_fusion_layer(point_features) # B, L, C

        keypoints_xyz, cnt_start = F.pad(key_xyz, pad=(1, 0), value=1), 0
        for i, xyz_cnt in enumerate(key_xyz_cnt):
            keypoints_xyz[cnt_start:cnt_start + xyz_cnt, 0] = i
            cnt_start += xyz_cnt

        return dict(
            keypoints=keypoints_xyz,
            keypoint_features=point_features,
            fusion_keypoint_features=fusion_point_features,
        )