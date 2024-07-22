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
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained, init_cfg
        )



