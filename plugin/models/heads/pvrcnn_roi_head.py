# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
from mmcv.ops import SparseConvTensor
from mmdet3d.core import AssignResult, SamplingResult, bbox3d2roi
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.models.builder import HEADS, build_head, build_middle_encoder
from mmdet3d.models.roi_heads.base_3droi_head import Base3DRoIHead
from mmdet.core.bbox.builder import build_assigner, build_sampler
from torch.nn import functional as F


@HEADS.register_module()
class PVRCNNRoiHead(Base3DRoIHead):
    """RoI head for PV-RCNN.

    Args:
        num_classes (int): The number of classes. Defaults to 3.
        semantic_head (dict, optional): Config of semantic head.
            Defaults to None.
        bbox_roi_extractor (dict, optional): Config of roi_extractor.
            Defaults to None.
        bbox_head (dict, optional): Config of bbox_head. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
    """

    def __init__(self,
                 pts_encoder: Optional[Dict] = None,
                 semantic_head: Optional[Dict] = None,
                 bbox_roi_extractor: Optional[Dict] = None,
                 bbox_head: Optional[Dict] = None,
                 train_cfg: Optional[Dict] = None,
                 test_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None):
        super(PVRCNNRoiHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        self.pts_encoder = build_middle_encoder(pts_encoder)
        self.semantic_head = build_head(semantic_head)
        self.bbox_roi_extractor = build_head(bbox_roi_extractor)

        self.init_assigner_sampler()

    @property
    def with_semantic(self):
        """bool: whether the head has semantic branch"""
        return hasattr(self, 'semantic_head') and self.semantic_head is not None

    def init_bbox_head(self, bbox_head: Dict):
        """Initialize the box head."""
        self.bbox_head = HEADS.build(bbox_head)

    def init_mask_head(self, mask_roi_extractor: Dict, mask_head: Dict):
        """Initialize maek head."""
        pass

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            if isinstance(self.train_cfg.assigner, dict):
                self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            elif isinstance(self.train_cfg.assigner, list):
                self.bbox_assigner = [
                    build_assigner(res) for res in self.train_cfg.assigner
                ]
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)

    def loss(
        self,
        rpn_results_list: List[LiDARInstance3DBoxes],
        gt_bboxes_3d: List[LiDARInstance3DBoxes],
        gt_labels_3d: List[torch.Tensor],
        points: List[torch.Tensor] = None,
        pts_feats: List[torch.Tensor] = None,
        voxel_feats: List[SparseConvTensor] = None,
        voxel_coors: List = None,
        **kwargs
    ) -> Dict:
        """Training forward function of PVRCNNROIHead.

        Args:
            feats_dict (dict): Contains point-wise features.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: losses from each head.

            - loss_semantic (torch.Tensor): loss of semantic head.
            - loss_bbox (torch.Tensor): loss of bboxes.
            - loss_cls (torch.Tensor): loss of object classification.
            - loss_corner (torch.Tensor): loss of bboxes corners.
        """
        feats_dict = self.pts_encoder(points, pts_feats, voxel_feats,
                                      voxel_coors, rpn_results_list)

        losses = dict()
        if self.with_semantic:
            semantic_results = self._semantic_forward_train(
                feats_dict['keypoint_features'], feats_dict['keypoints'],
                gt_bboxes_3d, gt_labels_3d)
            losses.update(loss_semantic=semantic_results['loss_semantic'])

        if self.with_bbox:
            sample_results = self._assign_and_sample(rpn_results_list,
                                                     gt_bboxes_3d, gt_labels_3d)
            bbox_results = self._bbox_forward_train(
                semantic_results['seg_preds'],
                feats_dict['fusion_keypoint_features'],
                feats_dict['keypoints'], sample_results)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def predict(
        self,
        rpn_results_list: List[LiDARInstance3DBoxes],
        points: List[torch.Tensor] = None,
        pts_feats: List[torch.Tensor] = None,
        img_metas: List[Dict] = None,
        voxel_feats: List[SparseConvTensor] = None,
        voxel_coors: List = None,
        **kwargs):
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains point-wise features.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert self.with_semantic, 'Semantic head must be implemented.'
        if len(rpn_results_list[0][0]) == 0:
            return rpn_results_list

        feats_dict = self.pts_encoder(points, pts_feats, voxel_feats,
                                      voxel_coors, rpn_results_list)

        semantic_results = self.semantic_head(feats_dict['keypoint_features'])
        point_features = feats_dict['fusion_keypoint_features'] * \
            semantic_results['seg_preds'].sigmoid().max(dim=-1, keepdim=True).values
        rois = bbox3d2roi([res[0].tensor for res in rpn_results_list])
        labels_3d = [res[2] for res in rpn_results_list]
        bbox_results = self._bbox_forward(point_features,
                                          feats_dict['keypoints'], rois)

        results_list = self.bbox_head.get_results(rois,
                                                  bbox_results['bbox_scores'],
                                                  bbox_results['bbox_reg'],
                                                  labels_3d, img_metas,
                                                  self.test_cfg)
        return results_list

    def forward_train(self):
        """Forward function during training."""
        pass

    def _bbox_forward_train(self, seg_preds: torch.Tensor,
                            keypoint_features: torch.Tensor,
                            keypoints: torch.Tensor,
                            sampling_results: SamplingResult) -> dict:
        """Forward training function of roi_extractor and bbox_head.

        Args:
            seg_preds (torch.Tensor): Point-wise semantic features.
            keypoint_features (torch.Tensor): key points features
                from points encoder.
            keypoints (torch.Tensor): Coordinate of key points.
            sampling_results (:obj:`SamplingResult`): Sampled results used
                for training.

        Returns:
            dict: Forward results including losses and predictions.
        """
        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        keypoint_features = keypoint_features * seg_preds.sigmoid().max(
            dim=-1, keepdim=True).values
        bbox_results = self._bbox_forward(keypoint_features, keypoints, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['bbox_scores'],
                                        bbox_results['bbox_reg'],
                                        rois, *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, keypoint_features: torch.Tensor,
                      keypoints: torch.Tensor, rois: torch.Tensor) -> dict:
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.

        Args:
            rois (Tensor): Roi boxes.
            keypoint_features (torch.Tensor): key points features
                from points encoder.
            keypoints (torch.Tensor): Coordinate of key points.
            rois (Tensor): Roi boxes.

        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        pooled_keypoint_features = self.bbox_roi_extractor(
            keypoint_features, keypoints, rois)
        bbox_score, bbox_reg = self.bbox_head(pooled_keypoint_features)

        bbox_results = dict(bbox_scores=bbox_score, bbox_reg=bbox_reg)
        return bbox_results

    def _assign_and_sample(
        self,
        proposal_list: List[Dict],
        gt_bboxes_3d: List[LiDARInstance3DBoxes],
        gt_labels_3d: List[torch.Tensor],
    ) -> List[SamplingResult]:
        """Assign and sample proposals for training.

        Args:
            proposal_list (list[:obj:`InstancesData`]): Proposals produced by
                rpn head.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.

        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        sampling_results = []
        # bbox assign
        for batch_idx in range(len(proposal_list)):
            cur_proposal_list = dict(
                bboxes_3d=proposal_list[batch_idx][0],
                labels_3d=proposal_list[batch_idx][2])
            cur_boxes = cur_proposal_list['bboxes_3d']
            cur_labels_3d = cur_proposal_list['labels_3d']

            cur_gt_instances_3d = dict(
                bboxes_3d=gt_bboxes_3d[batch_idx].tensor.to(cur_boxes.device),
                labels_3d=gt_labels_3d[batch_idx])
            cur_gt_bboxes = cur_gt_instances_3d['bboxes_3d']
            cur_gt_labels = cur_gt_instances_3d['labels_3d']

            batch_num_gts = 0
            # 0 is bg
            batch_gt_indis = cur_gt_labels.new_full((len(cur_boxes), ), 0)
            batch_max_overlaps = cur_boxes.tensor.new_zeros(len(cur_boxes))
            # -1 is bg
            batch_gt_labels = cur_gt_labels.new_full((len(cur_boxes), ), -1)

            # each class may have its own assigner
            if isinstance(self.bbox_assigner, list):
                for i, assigner in enumerate(self.bbox_assigner):
                    gt_per_cls = (cur_gt_labels == i)
                    pred_per_cls = (cur_labels_3d == i)
                    cur_assign_res = assigner.assign(
                        cur_proposal_list[pred_per_cls],
                        cur_gt_instances_3d[gt_per_cls])
                    # gather assign_results in different class into one result
                    batch_num_gts += cur_assign_res.num_gts
                    # gt inds (1-based)
                    gt_inds_arange_pad = gt_per_cls.nonzero(
                        as_tuple=False).view(-1) + 1
                    # pad 0 for indice unassigned
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=0)
                    # pad -1 for indice ignore
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=-1)
                    # convert to 0~gt_num+2 for indices
                    gt_inds_arange_pad += 1
                    # now 0 is bg, >1 is fg in batch_gt_indis
                    batch_gt_indis[pred_per_cls] = gt_inds_arange_pad[
                        cur_assign_res.gt_inds + 1] - 1
                    batch_max_overlaps[
                        pred_per_cls] = cur_assign_res.max_overlaps
                    batch_gt_labels[pred_per_cls] = cur_assign_res.labels

                assign_result = AssignResult(batch_num_gts, batch_gt_indis,
                                             batch_max_overlaps,
                                             batch_gt_labels)
            else:  # for single class
                assign_result = self.bbox_assigner.assign(
                    cur_proposal_list, cur_gt_instances_3d)
            # sample boxes
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes,
                                                       cur_gt_labels)
            sampling_results.append(sampling_result)
        return sampling_results

    def _semantic_forward_train(
        self,
        keypoint_features: torch.Tensor,
        keypoints: torch.Tensor,
        gt_bboxes_3d: List[LiDARInstance3DBoxes],
        gt_labels_3d: List[torch.Tensor]
    ) -> Dict:
        """Train semantic head.

        Args:
            keypoint_features (torch.Tensor): key points features
                from points encoder.
            keypoints (torch.Tensor): Coordinate of key points.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.

        Returns:
            dict: Segmentation results including losses
        """
        semantic_results = self.semantic_head(keypoint_features)
        semantic_targets = self.semantic_head.get_targets(
            keypoints, gt_bboxes_3d, gt_labels_3d)
        loss_semantic = self.semantic_head.loss(semantic_results,
                                                semantic_targets)
        semantic_results.update(loss_semantic)
        return semantic_results
