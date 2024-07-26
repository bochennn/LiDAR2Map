from typing import List, Optional

import torch
from mmdet3d.models.builder import MODELS, build_head
import torch.nn.functional as F

from ...datasets.evaluate import onehot_iou_torch
from ..losses import feature_distill_loss, logit_distill_loss, logits_loss
from .bevfusion import BEVFusion


@MODELS.register_module()
class LiDAR2Map(BEVFusion):

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
        pts_seg_head: Optional[dict] = None,
        fusion_seg_head: Optional[dict] = None,
        **kwargs
    ):
        super(LiDAR2Map, self).__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, view_transform, img_neck, pts_neck)

        self.pts_seg_head = build_head(pts_seg_head)
        self.fusion_seg_head = build_head(fusion_seg_head)

    def extract_pts_feat(self, pts: torch.Tensor):
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
        """ """
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

        bev_feats, depth = self.view_transform(img_feats, img_metas)
        return bev_feats, depth

    def forward_train(
        self,
        points: List[torch.Tensor] = None,
        img: torch.Tensor = None,
        img_metas: List = None,
        gt_semantic_seg: torch.Tensor = None
    ):
        img_feats, depth = self.extract_img_feat(img, img_metas)
        pts_feats, low_level_feats = self.extract_pts_feat(points)
        fusion_feats = self.pts_fusion_layer(img_feats, pts_feats)

        pts_semantic_logits, student_feats = self.pts_seg_head(pts_feats)
        fusion_semantic_logits, teacher_feats = self.fusion_seg_head(fusion_feats)

        loss_dict = self.loss(gt_semantic_seg, pts_semantic_logits, fusion_semantic_logits,
                              student_feats, teacher_feats, low_level_feats)

        return loss_dict

    def loss(self,
        gt_semantic_seg: torch.Tensor,  # onehot [B, cls, H, W]
        pts_semantic_logits: torch.Tensor,
        fusion_semantic_logits: torch.Tensor,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
        low_level_feats: torch.Tensor,
        loss_feats_weight: float = 0.4,
        loss_logits_weight: float = 1.5,
    ):
        pts_logits_loss = logits_loss(pts_semantic_logits, gt_semantic_seg)
        fusion_logits_loss = logits_loss(fusion_semantic_logits, gt_semantic_seg)

        loss_distill_feats = feature_distill_loss(student_feats, teacher_feats, low_level_feats)
        loss_distill_logits = logit_distill_loss(pts_semantic_logits, fusion_semantic_logits)

        pts_iou = onehot_iou_torch(_encode_onehot(pts_semantic_logits), gt_semantic_seg).mean(dim=0)

        loss_dict = dict(
            loss_seg_lidar=pts_logits_loss,
            loss_seg_fusion=fusion_logits_loss,
            loss_feats=loss_distill_feats,
            loss_logits=loss_distill_logits,
        )

        for idx, cls_iou in enumerate(pts_iou):
            loss_dict[f'pts_seg_iou_cls_{idx}'] = cls_iou.detach()
        return loss_dict

    def simple_test(self, points: torch.Tensor, img_metas, img, **kwargs):
        pts_feats, _ = self.extract_pts_feat(points)
        batch_semantic,_ = self.pts_seg_head(pts_feats) # B, cls, H, W

        seg_list = [dict(pts_semantic_seg=seg)
            for seg in _encode_onehot(batch_semantic, dim=1)]

        return seg_list


def _encode_onehot(logits: torch.Tensor) -> torch.Tensor:
    """ """
    _, cls, _, _ = logits.shape
    return F.one_hot(F.softmax(logits, dim=1).argmax(dim=1),
                     num_classes=cls).permute(0, 3, 1, 2)
