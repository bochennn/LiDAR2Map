from typing import Dict, Tuple

import torch
from mmdet3d.models.builder import VOXEL_ENCODERS
from mmdet3d.models.voxel_encoders.voxel_encoder import \
    DynamicSimpleVFE as _DynamicSimpleVFE
from mmdet3d.models.voxel_encoders.voxel_encoder import \
    DynamicVFE as _DynamicVFE
from mmdet3d.models.voxel_encoders.voxel_encoder import \
    HardSimpleVFE as _HardSimpleVFE


@VOXEL_ENCODERS.register_module(force=True)
class HardSimpleVFE(_HardSimpleVFE):
    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
        **kwargs: Dict
    ):
        voxel_feats = super(HardSimpleVFE, self).forward(
            features=voxels, num_points=num_points, coors=coors)

        return dict(
            batch_size=batch_size,
            voxel_features=voxel_feats,
            coors=coors
        )


@VOXEL_ENCODERS.register_module(force=True)
class DynamicSimpleVFE(_DynamicSimpleVFE):

    def __init__(
        self,
        voxel_size: Tuple[float] = (0.2, 0.2, 4),
        point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
        **kwargs: Dict
    ):
        super(DynamicSimpleVFE, self).__init__(
            voxel_size=voxel_size, point_cloud_range=point_cloud_range)

    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
        **kwargs
    ):
        voxel_feats, voxel_coors = super(
            DynamicSimpleVFE, self).forward(features=voxels, coors=coors)

        return dict(
            batch_size=batch_size,
            voxel_features=voxel_feats,
            coors=voxel_coors
        )


@VOXEL_ENCODERS.register_module(force=True)
class DynamicVFE(_DynamicVFE):

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Tuple[int] = (64),
        with_distance: bool = False,
        with_cluster_center: bool = True,
        with_voxel_center: bool = True,
        voxel_size: Tuple[float] = (0.2, 0.2, 4),
        point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
        norm_cfg: Dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode: str = 'max',
        fusion_layer: Dict = None,
        return_point_feats: bool = False,
        **kwargs: Dict
    ):
        super(DynamicVFE, self).__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            with_distance=with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            fusion_layer=fusion_layer,
            return_point_feats=return_point_feats
        )

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2] + 1e-9) / self.vz)
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1] + 1e-9) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0] + 1e-9) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
        **kwargs
    ):
        voxel_feats, voxel_coors = super(DynamicVFE, self).forward(features=voxels, coors=coors)

        return dict(
            batch_size=batch_size,
            voxel_features=voxel_feats,
            coors=voxel_coors
        )