from typing import List

import numpy as np
import torch
from mmdet3d.models.builder import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from torch import nn

from ..loss import compute_feature_distill_loss, compute_logit_distill_loss
from ..utils.base import BEV_FPD
from ..view_transform import LiftSplat


@MODELS.register_module()
class LiDAR2Map(MVXTwoStageDetector):

    def __init__(
        self,
        data_conf = None,
        embedded_dim=16,
        direction_pred=True,
        direction_dim=36,
        pts_voxel_layer = None,
        pts_voxel_encoder = None,
        pts_middle_encoder = None,
        pts_neck = None,
        img_backbone = None,
        img_neck = None,
        **kwargs
    ):
        super(LiDAR2Map, self).__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, None,
            img_backbone, None, img_neck, pts_neck,
        )
        self.view_transform = LiftSplat(data_conf)

        self.PGF2M = PosGuidedFeaFusion(128, 128)

        self.lidar_bevfpd = BEV_FPD(inC=128, outC=data_conf['num_channels'], instance_seg=False,
                                    embedded_dim=embedded_dim, direction_pred=direction_pred,
                                    direction_dim=direction_dim + 1)
        self.fusion_bevfpd = BEV_FPD(inC=128, outC=data_conf['num_channels'], instance_seg=False,
                                     embedded_dim=embedded_dim, direction_pred=False,
                                     direction_dim=direction_dim + 1)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)
        batch_size = coors[-1, 0] + 1

        pts_middle_feature = self.pts_middle_encoder(voxel_features, coors, batch_size)
        pts_feature = self.pts_neck(pts_middle_feature)
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
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        trans = img.new_tensor(np.stack([m['lidar2cam'][:, :3, 3] for m in img_metas]))
        rots = img.new_tensor(np.stack([m['lidar2cam'][:, :3, :3] for m in img_metas]))
        intrins = img.new_tensor(np.stack([m['cam2img'] for m in img_metas]))
        post_trans = img.new_tensor(np.stack([m['img_aug_matrix'][:, :3, 3] for m in img_metas]))
        post_rots = img.new_tensor(np.stack([m['img_aug_matrix'][:, :3, :3] for m in img_metas]))
        bev_feats = self.view_transform(img_feats, trans, rots, intrins, post_trans, post_rots)

        return bev_feats

    def forward(
        self, 
        points: List[torch.Tensor] = None,
        img: torch.Tensor = None,
        img_metas: List = None,
        flag='training',
        **kwargs
    ):
        """ """

        if flag == 'training':
            camera_feature = self.extract_img_feat(img, img_metas)
            lidar_feature, voxel_feature = self.extract_pts_feat(points)
            fusion_feature = self.PGF2M(camera_feature, lidar_feature)

            semantic, embedding, direction, student_feature = self.lidar_bevfpd(lidar_feature)
            fusion_semantic, fusion_embedding, fusion_direction, teacher_feature = self.fusion_bevfpd(fusion_feature)

            loss_feature_distill = compute_feature_distill_loss(student_feature, teacher_feature, voxel_feature)
            loss_logit_distill = compute_logit_distill_loss(semantic, fusion_semantic)

            return semantic, embedding, direction, loss_feature_distill, loss_logit_distill, fusion_semantic, fusion_embedding, fusion_direction
        else:
            # lidar_feature, _ = self.lidar2bev(lidar_data, lidar_mask)
            lidar_feature, _ = self.extract_pts_feat(points)
            semantic, embedding, direction, _ = self.lidar_bevfpd(lidar_feature)

            return semantic, embedding, direction


class PosGuidedFeaFusion(nn.Module):
    def __init__(self, cam_channel, lidar_channel):
        super(PosGuidedFeaFusion, self).__init__()
        self.fuse_posconv = nn.Sequential(
            nn.Conv2d(cam_channel + 2, cam_channel,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cam_channel)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(cam_channel+lidar_channel, cam_channel,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cam_channel)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cam_channel, cam_channel,
                      kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(cam_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(cam_channel, cam_channel,
                      kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(cam_channel),
            nn.Sigmoid()
        )

    def forward(self, fea_cam, fea_lidar):
        # add coord for camera
        x_range = torch.linspace(-1, 1, fea_cam.shape[-1], device=fea_cam.device)
        y_range = torch.linspace(-1, 1, fea_cam.shape[-2], device=fea_cam.device)
        y, x = torch.meshgrid(y_range, x_range, indexing='ij')

        y = y.expand([fea_cam.shape[0], 1, -1, -1])
        x = x.expand([fea_cam.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        cat_feature = torch.cat((fea_cam, fea_lidar), dim=1)
        fuse_out = self.fuse_conv(cat_feature)

        fuse_out = self.fuse_posconv(torch.cat((fuse_out, coord_feat), dim=1))
        attention_map = self.attention(fuse_out)
        out = fuse_out*attention_map + fea_cam

        return out
