from typing import Dict, Tuple

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import BACKBONES
from mmdet.models.backbones.resnet import ResLayer, BasicBlock


@BACKBONES.register_module()
class ResSECOND(BaseModule):
    """Backbone network for DSVT. The difference between `ResSECOND` and
    `SECOND` is that the basic block in this module contains residual layers.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        blocks_nums (list[int]): Number of blocks in each stage.
        layer_strides (list[int]): Strides of each stage.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 128,
                 out_channels: Tuple[int] = [128, 128, 256],
                 layer_nums: Tuple[int] = [1, 2, 2],
                 layer_strides: Tuple[int] = [2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg: Dict = None,
                 **kwargs) -> None:
        super(ResSECOND, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            blocks.append(ResLayer(
                block=BasicBlock,
                inplanes=in_filters[i],
                planes=out_channels[i],
                stride=layer_strides[i],
                num_blocks=layer_num,
                norm_cfg=norm_cfg,
                downsample_first=True))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)
