from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
# from mmdet3d.models.necks.second_fpn import SECONDFPN as _SECONDFPN
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class MultiScaleFPN(BaseModule):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        upsample_strides=[1, 2, 4],
        out_indices: List[int] = [0, 1, 2],
        start_level: int = 0,
        end_level: int = -1,
        no_norm_on_lateral: bool = False,
        conv_cfg: Dict = dict(type='Conv2d', bias=False),
        norm_cfg: Dict = dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride: bool = False,
        init_cfg=None
    ):
        super(MultiScaleFPN, self).__init__()

        if end_level == -1:
            self.backbone_end_level = len(in_channels) - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            # assert num_outs == end_level - start_level
        self.start_level = start_level
        # self.end_level = end_level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_indices = out_indices

        self.deblocks = nn.ModuleList()
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=out_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            self.deblocks.append(deblock)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = build_conv_layer(
                conv_cfg,
                in_channels=in_channels[i] + out_channels[i],
                out_channels=out_channels[i],
                kernel_size=1,
                # norm_cfg=norm_cfg if not no_norm_on_lateral else None,
            )
            fpn_conv = build_conv_layer(
                conv_cfg,
                in_channels=out_channels[i],
                out_channels=out_channels[i],
                kernel_size=3,
                padding=1,
                # norm_cfg=norm_cfg,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]


    def forward(self, inputs):

        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        for i in range(len(self.deblocks) - 1, -1, -1): # 2, 1, 0
            x = self.deblocks[i](laterals[i])

            if i > 0:
                laterals[i - 1] = torch.cat([laterals[i - 1], x], dim=1)
                laterals[i - 1] = self.lateral_convs[i - 1](laterals[i - 1])
                laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])
            else:
                laterals.insert(0, x)

        # build outputs
        outs = [laterals[i] for i in self.out_indices]
        return tuple(outs)