from configs.centerpoint.hv01_second_secfpn import POINT_CLOUD_RANGE, VOXEL_SIZE

_base_ = ['./hv01_second_secfpn.py']

model = dict(
    pts_voxel_layer=dict(max_num_points=-1, max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        # type='DynamicSimpleVFE',
        type='DynamicVFE',
        feat_channels=(64,),
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE),
    pts_middle_encoder=dict(
        in_channels=64,
        base_channels=64,
        encoder_channels=((64, 64, 128), (128, 128, 128), (128, 128, 128), (128, 128)),
    ),
    # pts_backbone=dict(type='ResSECOND'),
)