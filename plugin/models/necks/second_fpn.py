from typing import Dict

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet3d.models.builder import NECKS
from mmdet3d.models.necks.second_fpn import SECONDFPN as _SECONDFPN


@NECKS.register_module()
class MultiScaleFPN(_SECONDFPN):

    def __init__(
        self, in_channels, out_channels,
        start_level: int = 0,
        end_level: int = -1,
        no_norm_on_lateral: bool = False,
        norm_cfg: Dict = None,
        **kwargs
    ):
        super(MultiScaleFPN, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            norm_cfg=norm_cfg, **kwargs)

        if end_level == -1:
            self.backbone_end_level = len(in_channels) - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            # assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i] + (out_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels[i]),
                out_channels[i],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not no_norm_on_lateral else None,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels[i],
                out_channels[i],
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):

        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -2, -1):
            x = self.deblocks[i + 1](laterals[i + 1])

            if i >= 0:
                laterals[i] = torch.cat([laterals[i], x], dim=1)
                laterals[i] = self.lateral_convs[i](laterals[i])
                laterals[i] = self.fpn_convs[i](laterals[i])
            else:
                laterals[i] = x



        return []