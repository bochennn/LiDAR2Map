# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import BaseModule

from mmdet3d.models.builder import ROI_EXTRACTORS
from mmdet3d.core.bbox.box_np_ops import rotation_3d_in_axis
from mmdet3d.ops.pointnet_modules.builder import build_sa_module


@ROI_EXTRACTORS.register_module()
class Batch3DRoIGridExtractor(BaseModule):
    """Grid point wise roi-aware Extractor.

    Args:
        grid_size (int): The number of grid points in a roi bbox.
            Defaults to 6.
        roi_layer (dict, optional): Config of sa module to get
            grid points features. Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
    """

    def __init__(self,
                 grid_size: int = 6,
                 roi_layer: dict = None,
                 init_cfg: dict = None) -> None:
        super(Batch3DRoIGridExtractor, self).__init__(init_cfg=init_cfg)
        self.roi_grid_pool_layer = build_sa_module(roi_layer)
        self.grid_size = grid_size

    def forward(self, features: torch.Tensor, keypoints: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """Forward roi extractor to extract grid points feature.

        Args:
            feats (torch.Tensor): Key points features.
            coordinate (torch.Tensor): Key points coordinates.
            batch_inds (torch.Tensor): Input batch indexes.
            rois (torch.Tensor): Detection results of rpn head.

        Returns:
            torch.Tensor: Grid points features.
        """
        xyz = keypoints[:, 1:4].contiguous()
        batch_size = int(keypoints[-1, 0].item() + 1)
        xyz_batch_cnt = keypoints.new_zeros(batch_size).int()
        for k in range(batch_size):
            xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()

        rois_batch_inds = rois[:, 0].int()
        # (N1+N2+..., 6x6x6, 3)
        roi_grid = self.get_dense_grid_points(rois[:, 1:])

        new_xyz = roi_grid.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (rois_batch_inds == k).sum() * roi_grid.shape[1]

        _, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz,
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=features)  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(-1, self.grid_size,
                                               self.grid_size, self.grid_size,
                                               pooled_features.shape[-1])
        # (BxN, 6, 6, 6, C)
        return pooled_features

    def get_dense_grid_points(self, rois: torch.Tensor) -> torch.Tensor:
        """Get dense grid points from rois.

        Args:
            rois (torch.Tensor): Detection results of rpn head.

        Returns:
            torch.Tensor: Grid points coordinates.
        """
        rois_bbox = rois.clone()
        rois_bbox[:, 2] += rois_bbox[:, 5] / 2
        faked_features = rois_bbox.new_ones(
            (self.grid_size, self.grid_size, self.grid_size))
        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(rois_bbox.size(0), 1, 1).float()
        dense_idx = ((dense_idx + 0.5) / self.grid_size)
        dense_idx[..., :3] -= 0.5

        roi_ctr = rois_bbox[:, :3]
        roi_dim = rois_bbox[:, 3:6]
        roi_grid_points = dense_idx * roi_dim.view(-1, 1, 3)
        roi_grid_points = rotation_3d_in_axis(
            roi_grid_points, rois_bbox[:, 6], axis=2)
        roi_grid_points += roi_ctr.view(-1, 1, 3)

        return roi_grid_points
