from typing import List

from mmcv.runner import auto_fp16
from mmdet3d.models.builder import MIDDLE_ENCODERS
from mmdet3d.models.middle_encoders.sparse_encoder import \
    SparseEncoder as _SparseEncoder
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


@MIDDLE_ENCODERS.register_module(force=True)
class SparseEncoder(_SparseEncoder):

    def __init__(self, out_indices: List = [], **kwargs):
        super(SparseEncoder, self).__init__(**kwargs)
        self.out_indices = out_indices

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

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features, [encode_features[i] for i in self.out_indices]

