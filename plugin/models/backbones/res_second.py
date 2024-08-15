from typing import Dict, Tuple

import torch
import torch.nn as nn
from mmdet3d.models.backbones.second import SECOND
from mmdet3d.models.builder import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock


@BACKBONES.register_module()
class ResSECOND(SECOND):
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
                 blocks_nums: Tuple[int] = [1, 2, 2],
                 layer_strides: Tuple[int] = [2, 2, 2],
                 init_cfg: Dict = None) -> None:
        super(ResSECOND, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(blocks_nums)
        assert len(out_channels) == len(blocks_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, block_num in enumerate(blocks_nums):
            cur_layers = [
                BasicBlock(
                    in_filters[i],
                    out_channels[i],
                    stride=layer_strides[i],
                    downsample=True)
            ]
            for _ in range(block_num):
                cur_layers.append(
                    BasicBlock(out_channels[i], out_channels[i]))
            blocks.append(nn.Sequential(*cur_layers))
        self.blocks = nn.Sequential(*blocks)

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

