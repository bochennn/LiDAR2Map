import torch
from mmdet3d.models.builder import VOXEL_ENCODERS
from mmdet3d.models.voxel_encoders.voxel_encoder import \
    DynamicSimpleVFE as _DynamicSimpleVFE
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
        **kwargs
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
