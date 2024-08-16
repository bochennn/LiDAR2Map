from configs.centerpoint.hv01_second_secfpn import VOXEL_GRID_SIZE, VOXEL_SIZE, POINT_CLOUD_RANGE

_base_ = ['./cbgs_dv01_second_secfpn.py']

model = dict(
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        feat_channels=[64, 64],
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE),
    pts_middle_encoder=dict(
        _delete_=True,
        type='ScatterFormer',
        in_channels=64,
        base_channels=64,
        output_channels=128,
        sparse_shape=[VOXEL_GRID_SIZE[2] + 1, VOXEL_GRID_SIZE[1], VOXEL_GRID_SIZE[0]],
        encoder_channels=((64, 64, 128), (128, 128, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, (0, 1, 1)), (0, 0)),
        out_padding=(1, 0, 0),
        attn_window_size=20,
        block_type='basicblock'
    ),
    pts_backbone=dict(
        in_channels=128,
    )
)