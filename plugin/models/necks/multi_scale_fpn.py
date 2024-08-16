from typing import Dict, List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import build_upsample_layer
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class MSFPN(BaseModule):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        upsample_scales: List[int] = [2, 2, 2],
        out_indices: List[int] = [0, 1, 2],
        upsample_cfg=dict(type='bilinear', align_corners=True),
        norm_cfg: Dict = dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
    ):
        super(MSFPN, self).__init__()

        self.in_channels = in_channels
        self.out_indices = out_indices
        backbone_end_level = len(in_channels) - 1

        self.upsample_layer = nn.ModuleList()
        for i, upsample_scale in enumerate(upsample_scales):
            self.upsample_layer.append(
                build_upsample_layer(
                    upsample_cfg, scale_factor=upsample_scale))

        self.fpn_convs = nn.ModuleList()
        for i in range(backbone_end_level):
            self.fpn_convs.append(nn.Sequential(
                ConvModule(in_channels[i] + (
                    in_channels[i + 1] if i == backbone_end_level - 1 else out_channels[i + 1]),
                           out_channels[i],
                           kernel_size=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg,
                           inplace=False),
                ConvModule(out_channels[i],
                           out_channels[i],
                           kernel_size=3,
                           padding=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg,
                           inplace=False),
            ))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [inputs[i] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1): # 1, 0
            x = self.upsample_layer[i - used_backbone_levels](laterals[i + 1])
            laterals[i] = self.fpn_convs[i](torch.cat([laterals[i], x], dim=1))

        if len(self.upsample_layer) > len(self.fpn_convs):
            laterals.insert(0, self.upsample_layer[0](laterals[0]))

        # build outputs
        outs = [laterals[i] for i in self.out_indices]
        return tuple(outs)