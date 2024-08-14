from typing import Dict, Tuple

from mmcv.runner import auto_fp16
from mmdet3d.models.builder import MIDDLE_ENCODERS
from mmdet3d.models.middle_encoders.sparse_encoder import \
    SparseEncoder as _SparseEncoder
from mmdet3d.ops import make_sparse_convmodule
from spconv.pytorch import SparseConvTensor


@MIDDLE_ENCODERS.register_module(force=True)
class SparseEncoder(_SparseEncoder):

    def __init__(self,
                 out_indices: Tuple = [],
                 encoder_channels: Tuple = None,
                 out_padding: Tuple = 0,
                 norm_cfg: Dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 **kwargs):
        super(SparseEncoder, self).__init__(
            encoder_channels=encoder_channels, norm_cfg=norm_cfg, **kwargs)

        self.out_indices = out_indices
        self.conv_out = make_sparse_convmodule(
            encoder_channels[-1][-1],
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=out_padding,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features, [encode_features[i] for i in self.out_indices]

