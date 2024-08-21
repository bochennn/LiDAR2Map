import torch.nn as nn
from mmdet3d.models.builder import MIDDLE_ENCODERS
from spconv.pytorch import SparseConv3d, SparseConvTensor, SparseSequential

from ..layers.scatter_former_layer import AttnPillarPool, ScatterFormerLayer3x
from .sparse_encoder import SparseEncoder


@MIDDLE_ENCODERS.register_module()
class ScatterFormer(SparseEncoder):

    def __init__(
        self,
        output_channels: int = 128,
        attn_window_size: int = 20,
        **kwargs
    ):
        super(ScatterFormer, self).__init__(output_channels=output_channels, **kwargs)

        #  [472, 472, 11] -> [236, 236, 6]
        self.post_former_layer = SparseSequential()
        self.post_former_layer.add_module(
            f'post_former_layer0',
            SparseSequential(
                ScatterFormerLayer3x(output_channels, nhead=4, num_layers=3,
                                     win_size=attn_window_size,
                                     indice_key='scatter_former'),
                nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01),
                SparseConv3d(output_channels, output_channels, 3, stride=(1, 2, 2), padding=1,
                             bias=False, indice_key='spconv_down')))

        self.post_former_layer.add_module(
            f'post_former_layer1',
            SparseSequential(
                ScatterFormerLayer3x(output_channels, nhead=4, num_layers=3,
                                     win_size=attn_window_size,
                                     indice_key='scatter_former'),
                nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01),
                AttnPillarPool(output_channels, self.sparse_shape[0] // 8)))

    def forward(self, voxel_features, coors, batch_size):
        """ """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        out = self.post_former_layer(out)
        spatial_features = out.dense()

        spatial_features = spatial_features.flatten(1, 2)
        return spatial_features, [encode_features[i] for i in self.out_indices]
