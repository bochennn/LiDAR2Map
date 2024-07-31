from typing import Dict, List, Optional

import torch
from mmcv.runner import force_fp32
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.models.builder import DETECTORS, build_head, build_middle_encoder
from mmdet3d.models.detectors.mvx_two_stage import \
    MVXTwoStageDetector as _MVXTwoStageDetector
from torch.nn import functional as F


@DETECTORS.register_module(force=True)
class MVXTwoStageDetector(_MVXTwoStageDetector):

    def __init__(
        self,
        pts_encoder: Optional[Dict] = None,
        pts_roi_head: Optional[Dict] = None,
        **kwargs
    ):
        super(MVXTwoStageDetector, self).__init__(**kwargs)
        if pts_roi_head is not None:
            self.pts_roi_head = build_head(pts_roi_head)
        if pts_encoder is not None:
            self.pts_encoder = build_middle_encoder(pts_encoder)

    @property
    def with_pts_roi_head(self):
        """bool: Whether the detector has a RoI Head in 3D branch."""
        return hasattr(self, 'pts_roi_head') and self.pts_roi_head is not None

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

    def extract_pts_feat(self, pts: List[torch.Tensor], img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxel_dict = self.voxelize(pts)
        feats_dict = self.pts_voxel_encoder(**voxel_dict)
        pts_middle_feature = self.pts_middle_encoder(**feats_dict)
        pts_feature = self.pts_backbone(pts_middle_feature)
        if self.with_pts_neck:
            pts_feature = self.pts_neck(pts_feature)
        return pts_feature

    def forward_pts_train(self,
                          pts_feats: List[torch.Tensor],
                          gt_bboxes_3d: List[LiDARInstance3DBoxes],
                          gt_labels_3d: List[torch.Tensor],
                          img_metas: List[Dict],
                          gt_bboxes_ignore = None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        losses = self.pts_bbox_head.loss(gt_bboxes_3d=gt_bboxes_3d,
                                         gt_labels_3d=gt_labels_3d,
                                         preds_dicts=outs)
        if self.with_pts_roi_head:
            proposal_list = self.pts_bbox_head.get_bboxes(outs, img_metas)

        return losses

    def forward_test(
        self,
        points: List[torch.Tensor] = None,
        img_metas: List = None,
        img: torch.Tensor = None,
        rescale: bool = False,
        **kwargs
    ):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for _ in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
