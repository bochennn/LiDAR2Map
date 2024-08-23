_base_ = ['./cbgs_dv01_second_secfpn.py']

model = dict(
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        feat_channels=(64, 64)
    ),
    pts_middle_encoder=dict(
        in_channels=64,
        base_channels=64,
        encoder_channels=((64, 64, 128), (128, 128, 128), (128, 128, 128), (128, 128)),
    ),
)