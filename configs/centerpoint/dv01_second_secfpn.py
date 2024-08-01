from configs.centerpoint.hv01_second_secfpn_baseline import POINT_CLOUD_RANGE, VOXEL_SIZE

_base_ = ['./hv01_second_secfpn_baseline.py']

model = dict(
    pts_voxel_layer=dict(max_num_points=-1, max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicSimpleVFE',
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE),
)