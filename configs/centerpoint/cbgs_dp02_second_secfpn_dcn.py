# from configs.centerpoint.hv01_second_secfpn_baseline import POINT_CLOUD_RANGE, VOXEL_SIZE

_base_ = ['./dv01_second_secfpn.py']

VOXEL_SIZE = [0.2, 0.2, 8]
POINT_CLOUD_RANGE = [-82.4, -76.8, -3.0, 122.4, 76.8, 5.0]
VOXEL_GRID_SIZE = [
    int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0] + 1e-9) / VOXEL_SIZE[0]),
    int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1] + 1e-9) / VOXEL_SIZE[1]),
    int((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2] + 1e-9) / VOXEL_SIZE[2]),
]

model = dict(
    pts_voxel_layer=dict(max_num_points=-1, max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicSimpleVFE',
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE),
)