from typing import List

import torch
from mmdet3d.models.builder import MODELS
from torch import nn


@MODELS.register_module()
class ConvFuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


@MODELS.register_module()
class PositionGuidedFusion(nn.Module):
    def __init__(
        self,
        img_feats_channel: int,
        pts_feats_channel: int
    ):
        super(PositionGuidedFusion, self).__init__()
        self.fuse_posconv = nn.Sequential(
            nn.Conv2d(img_feats_channel + 2, img_feats_channel,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(img_feats_channel)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(img_feats_channel + pts_feats_channel, img_feats_channel,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(img_feats_channel)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(img_feats_channel, img_feats_channel,
                      kernel_size=1, padding=0, stride=1),
            nn.LayerNorm([img_feats_channel, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(img_feats_channel, img_feats_channel,
                      kernel_size=1, padding=0, stride=1),
            nn.LayerNorm([img_feats_channel, 1, 1]),
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
        out = fuse_out * attention_map + fea_cam

        return out
