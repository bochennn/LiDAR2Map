from typing import List

import torch
from mmcv.runner import force_fp32
from mmdet3d.models.builder import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from torch.nn import functional as F

from ...datasets.evaluate import batch_iou_torch
from ..heads.bev_fpd import BEV_FPD
from ..layers import PositionGuidedFusion
from ..loss import (SimpleLoss, compute_feature_distill_loss,
                    compute_logit_distill_loss)
from ..view_transform import LiftSplat


@MODELS.register_module()
class LiDAR2Map(MVXTwoStageDetector):

    def __init__(
        self,
        data_conf = None,
        embedded_dim=16,
        direction_dim=36,
        pts_voxel_layer = None,
        pts_voxel_encoder = None,
        pts_middle_encoder = None,
        pts_backbone = None,
        pts_neck = None,
        img_backbone = None,
        img_neck = None,
        **kwargs
    ):
        super(LiDAR2Map, self).__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, None,
            img_backbone, pts_backbone, img_neck, pts_neck,
        )
        self.view_transform = LiftSplat(data_conf)

        self.PGF2M = PositionGuidedFusion(128, 128)

        self.lidar_bevfpd = BEV_FPD(inC=128, outC=data_conf['num_channels'], instance_seg=False,
                                    embedded_dim=embedded_dim, direction_pred=False,
                                    direction_dim=direction_dim + 1)
        self.fusion_bevfpd = BEV_FPD(inC=128, outC=data_conf['num_channels'], instance_seg=False,
                                     embedded_dim=embedded_dim, direction_pred=False,
                                     direction_dim=direction_dim + 1)

        self.loss_fn = SimpleLoss(2.13)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
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

    def extract_pts_feat(
        self,
        pts: torch.Tensor,
        img_feats: torch.Tensor = None,
        img_metas: List = None
    ):
        """Extract features of points."""
        voxel_dict = self.voxelize(pts)
        feats_dict = self.pts_voxel_encoder(**voxel_dict)
        pts_middle_feature = self.pts_middle_encoder(feats_dict['voxel_feats'],
                                                     feats_dict['voxel_coors'],
                                                     voxel_dict['batch_size'])
        pts_feature = self.pts_backbone(pts_middle_feature)
        pts_feature = self.pts_neck(pts_feature)

        if isinstance(pts_feature, list):
            pts_feature = pts_feature[0]

        return pts_feature, pts_middle_feature

    def extract_img_feat(self, img: torch.Tensor, img_metas: List):

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)

        img_feats = self.img_backbone(img)  # multi scale: 1/8, 1/16, 1/32
        img_feats = self.img_neck(img_feats)

        bev_feats = self.view_transform(img_feats, img_metas)
        return bev_feats

    def forward_train(
        self,
        points: List[torch.Tensor] = None,
        img: torch.Tensor = None,
        img_metas: List = None,
        gt_semantic_seg: torch.Tensor = None
    ):
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats, voxel_feats = self.extract_pts_feat(points, img_feats, img_metas)
        fusion_feats = self.PGF2M(img_feats, pts_feats)

        pts_semantic_seg, student_feats = self.lidar_bevfpd(pts_feats)
        fusion_semantic_seg, teacher_feats = self.fusion_bevfpd(fusion_feats)

        loss_feature_distill = compute_feature_distill_loss(student_feats, teacher_feats, voxel_feats)
        loss_logit_distill = compute_logit_distill_loss(pts_semantic_seg, fusion_semantic_seg)

        seg_loss = self.loss_fn(pts_semantic_seg, gt_semantic_seg)
        fusion_seg_loss = self.loss_fn(fusion_semantic_seg, gt_semantic_seg)

        semantic_iou = batch_iou_torch(pts_semantic_seg, gt_semantic_seg).mean(dim=0)

        loss_dict = dict(
            loss_feature=loss_feature_distill,
            loss_logit=loss_logit_distill,
            loss_semantic_lidar=seg_loss,
            loss_semantic_fusion=fusion_seg_loss,
        )

        for idx, cls_iou in enumerate(semantic_iou):
            loss_dict[f'pts_seg_iou_cls_{idx}'] = cls_iou

        return loss_dict

    def simple_test(
        self,
        points: torch.Tensor,
        img: torch.Tensor = None,
        img_metas: List = None,
        **kwargs
    ):
        pts_feats, _ = self.extract_pts_feat(points)
        batch_semantic,_ = self.lidar_bevfpd(pts_feats) # B, cls, H, W

        seg_list = [dict(pts_semantic_seg=seg)
            for seg in onehot_encoding(batch_semantic, dim=1)]

        return seg_list


def onehot_encoding(logits: torch.Tensor, dim: int = 0):
    """ """
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot
