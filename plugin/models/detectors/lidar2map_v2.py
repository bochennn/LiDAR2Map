from typing import Optional

from mmdet3d.models.builder import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


@MODELS.register_module()
class LiDAR2MapV2(MVXTwoStageDetector):

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
        super(LiDAR2MapV2, self).__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained, init_cfg
        )
        self.view_transform = view_transform

    @property
    def with_view_transform(self):
        """bool: Whether the detector has a view transform layer."""
        return hasattr(self,
                       'view_transform') and self.view_transform is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        if self.with_view_transform:
            bev_feats = self.view_transform()
        # print(len(img_feats), img_feats[0].shape)

        return img_feats

    def forward_train(
        self,
        points=None,
        img_metas=None,
        # gt_bboxes_3d=None,
        # gt_labels_3d=None,
        # gt_labels=None,
        # gt_bboxes=None,
        img=None,
        # proposals=None,
        # gt_bboxes_ignore=None
        **kwargs
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
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        print(img_metas[0].keys())

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()

        print(img_feats.shape)
        print(pts_feats.shape)
        print(xx)
        # if pts_feats:
        #     losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
        #                                         gt_labels_3d, img_metas,
        #                                         gt_bboxes_ignore)
        #     losses.update(losses_pts)
        # if img_feats:
        #     losses_img = self.forward_img_train(
        #         img_feats,
        #         img_metas=img_metas,
        #         gt_bboxes=gt_bboxes,
        #         gt_labels=gt_labels,
        #         gt_bboxes_ignore=gt_bboxes_ignore,
        #         proposals=proposals)
        #     losses.update(losses_img)
        return losses

