from typing import Dict, List, Optional, Tuple

import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.models.builder import DETECTORS, build_head
from mmdet3d.models.detectors.centerpoint import CenterPoint as _CenterPoint

from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module(force=True)
class CenterPoint(MVXTwoStageDetector, _CenterPoint):

    def __init__(
        self,
        pts_roi_head: Optional[Dict] = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        **kwargs
    ):
        super(CenterPoint, self).__init__(
            train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)

        if pts_roi_head is not None:
            pts_roi_head.update(train_cfg=train_cfg.get('rcnn'),
                                test_cfg=test_cfg.get('rcnn'))
            self.pts_roi_head = build_head(pts_roi_head)

    @property
    def with_pts_roi_head(self) -> bool:
        """bool: Whether the detector has a RoI Head in 3D branch."""
        return hasattr(self, 'pts_roi_head') and self.pts_roi_head is not None

    def extract_pts_feat(
        self,
        pts: List[torch.Tensor],
        img_feats: Optional[torch.Tensor] = None,
        img_metas: Optional[List] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Extract features of points."""
        voxel_dict = self.voxelize(pts)
        feats_dict = self.pts_voxel_encoder(**voxel_dict)
        pts_middle_feature, voxel_feature = self.pts_middle_encoder(**feats_dict)
        pts_feature = self.pts_backbone(pts_middle_feature)
        if self.with_pts_neck:
            pts_feature = self.pts_neck(pts_feature)
        return pts_feature, \
            dict(voxel_feats=voxel_feature,
                 voxel_coors=feats_dict['coors'])

    def forward_train(
        self,
        points: List[torch.Tensor] = None,
        img_metas: List[Dict] = None,
        gt_bboxes_3d: List[LiDARInstance3DBoxes] = None,
        gt_labels_3d: List[torch.Tensor] = None,
        gt_bboxes_ignore: List[torch.Tensor] = None,
        **kwargs: Optional[Dict]
    ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        pts_feats, extra_feats = self.extract_pts_feat(points)

        preds_dicts = self.pts_bbox_head(pts_feats)
        losses = self.pts_bbox_head.loss(gt_bboxes_3d=gt_bboxes_3d,
                                         gt_labels_3d=gt_labels_3d,
                                         preds_dicts=preds_dicts)
        if self.with_pts_roi_head:
            rpn_results_list = self.pts_bbox_head.get_bboxes(preds_dicts, img_metas)
            losses.update(self.pts_roi_head.forward_train(rpn_results_list,
                                                          gt_bboxes_3d, gt_labels_3d,
                                                          points=points,
                                                          pts_feats=pts_feats,
                                                          **extra_feats))
        return losses

    def forward_test(
        self,
        points: List[torch.Tensor],
        img_metas: List[Dict],
        img: Optional[torch.Tensor] = None,
        rescale: Optional[bool] = False
    ) -> List[Dict]:
        """Test function without augmentaiton."""
        pts_feats, extra_feats = self.extract_pts_feat(points)

        bbox_list = [dict() for _ in range(len(img_metas))]

        bbox_pts = self.simple_test_pts(
            pts_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        if self.with_pts_roi_head:
            pass
        return bbox_list
