# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet3d.models.builder import NECKS
from torch.cuda.amp.autocast_mode import autocast

from ..layers.depth_net import DepthNet
from .lss_v2 import LSSViewTransformerV2


@NECKS.register_module()
class LSSViewTransformerBEVDepth(LSSViewTransformerV2):

    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformerBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.out_channels, self.D, **depthnet_cfg)

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 4, 4).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input, stereo_metas=None):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input, stereo_metas)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        bev_feat, depth = self.view_transform(input, depth, tran_feat)
        return bev_feat, depth
