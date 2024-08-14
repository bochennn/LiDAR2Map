from configs.centerpoint.hv01_second_secfpn import VOXEL_GRID_SIZE

_base_ = ['./cbgs_dv01_second_secfpn.py']

model = dict(
    pts_middle_encoder=dict(
        _delete_=True,
        type='ScatterFormer',
        in_channels=4,
        output_channels=128,
        sparse_shape=[VOXEL_GRID_SIZE[2] + 1, VOXEL_GRID_SIZE[1], VOXEL_GRID_SIZE[0]],
        encoder_channels=((16, 16, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, (0, 1, 1)), (0, 0)),
        attn_window_size=20,
        block_type='basicblock'
    ),
    pts_backbone=dict(
        in_channels=128,
    )
)