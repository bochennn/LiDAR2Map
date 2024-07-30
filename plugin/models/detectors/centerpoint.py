from mmdet3d.models.detectors.centerpoint import CenterPoint as _CenterPoint
from mmdet3d.models.builder import DETECTORS
# import torch
# from typing import List
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module(force=True)
class CenterPoint(_CenterPoint, MVXTwoStageDetector):

    pass
    # def forward_test(
    #     self,
    #     points: torch.Tensor = None,
    #     img_metas: List = None,
    #     img: torch.Tensor = None,
    #     rescale: bool = False,
    #     **kwargs
    # ):
    #     img_feats, pts_feats = self.extract_feat(
    #         points, img=img, img_metas=img_metas)

    #     bbox_list = [dict() for i in range(len(img_metas))]
    #     if pts_feats and self.with_pts_bbox:
    #         bbox_pts = self.simple_test_pts(
    #             pts_feats, img_metas, rescale=rescale)
    #         for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
    #             result_dict['pts_bbox'] = pts_bbox
    #     if img_feats and self.with_img_bbox:
    #         bbox_img = self.simple_test_img(
    #             img_feats, img_metas, rescale=rescale)
    #         for result_dict, img_bbox in zip(bbox_list, bbox_img):
    #             result_dict['img_bbox'] = img_bbox
    #     return bbox_list
