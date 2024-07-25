from typing import Dict, List, Optional

import torch
from mmcv.runner import force_fp32
from mmdet3d.models.builder import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from torch.nn import functional as F


@MODELS.register_module()
class BEVFusion(MVXTwoStageDetector):

    def __init__(
        self,
        pts_voxel_layer: Optional[dict] = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        pts_fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        pts_bbox_head: Optional[dict] = None,
        img_roi_head: Optional[dict] = None,
        img_rpn_head: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        pretrained: Optional[dict] = None,
        init_cfg: Optional[dict] = None
    ):
        super(BEVFusion, self).__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained, init_cfg
        )
        self.view_transform = view_transform

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points: List[torch.Tensor]) -> Dict:
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxel_dict = dict(voxels=[], coors=[], num_points=[])
        for i, res in enumerate(points):
            if self.pts_voxel_layer.max_num_points == -1:
                res_coors = self.pts_voxel_layer(res)
                voxel_dict['voxels'].append(res)
            else:
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
                voxel_dict['voxels'].append(res_voxels)
                voxel_dict['num_points'].append(res_num_points)
            voxel_dict['coors'].append(F.pad(res_coors, (1, 0), mode='constant', value=i))

        voxel_dict['voxels'] = torch.cat(voxel_dict['voxels'], dim=0)
        voxel_dict['coors'] = torch.cat(voxel_dict['coors'], dim=0)
        if len(voxel_dict['num_points']) > 0:
            voxel_dict['num_points'] = torch.cat(voxel_dict['num_points'], dim=0)
        voxel_dict['batch_size'] = len(points)
        return voxel_dict
