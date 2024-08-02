from typing import Dict, List, Optional

import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.models.builder import DETECTORS, build_head, build_middle_encoder
from mmdet3d.models.detectors.centerpoint import CenterPoint as _CenterPoint

from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module(force=True)
class CenterPoint(MVXTwoStageDetector, _CenterPoint):

    def __init__(
        self,
        pts_encoder: Optional[Dict] = None,
        pts_roi_head: Optional[Dict] = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        **kwargs
    ):
        super(CenterPoint, self).__init__(
            train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)
        if pts_roi_head is not None:
            pts_roi_head.update(
                train_cfg=train_cfg.get('rcnn'),
                test_cfg=test_cfg.get('rcnn'))
            self.pts_roi_head = build_head(pts_roi_head)
        if pts_encoder is not None:
            self.pts_encoder = build_middle_encoder(pts_encoder)

    @property
    def with_pts_roi_head(self):
        """bool: Whether the detector has a RoI Head in 3D branch."""
        return hasattr(self, 'pts_roi_head') and self.pts_roi_head is not None

    def extract_pts_feat(self, pts: List[torch.Tensor]):
        """Extract features of points."""
        voxel_dict = self.voxelize(pts)
        feats_dict = self.pts_voxel_encoder(**voxel_dict)
        pts_middle_feature = self.pts_middle_encoder(**feats_dict)
        pts_feature = self.pts_backbone(pts_middle_feature)
        if self.with_pts_neck:
            pts_feature = self.pts_neck(pts_feature)
        return pts_feature

    def forward_train(
        self,
        points: List[torch.Tensor] = None,
        img_metas: List[Dict] = None,
        gt_bboxes_3d: List[LiDARInstance3DBoxes] = None,
        gt_labels_3d: List[torch.Tensor] = None,
        gt_bboxes_ignore: List[torch.Tensor] = None,
        **kwargs: Dict
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
        pts_feats = self.extract_pts_feat(points)

        print(type(pts_feats), len(pts_feats))

        losses = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                        gt_labels_3d, gt_bboxes_ignore,
                                        points, img_metas)
        return losses

    def forward_pts_train(self,
                          pts_feats: List[torch.Tensor],
                          gt_bboxes_3d: List[LiDARInstance3DBoxes],
                          gt_labels_3d: List[torch.Tensor],
                          gt_bboxes_ignore: List[torch.Tensor] = None,
                          img_metas: List[Dict] = None,
                          **kwargs):
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
            rpn_results_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
            self.pts_encoder(pts_feats, rpn_results_list, **kwargs)

            # [[instance_box, scores, types], ..]
            self.pts_roi_head.loss(dict(), rpn_results_list,
                                   gt_bboxes_3d, gt_labels_3d)
            raise NotImplementedError

        return losses

