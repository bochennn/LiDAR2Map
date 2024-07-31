_base_ = ['./hv01_second_center_head.py']

voxel_size = [0.1, 0.1, 0.2]
point_cloud_range = [-82.4, -57.6, -3.0, 122.4, 57.6, 5.0]

model = dict(
    pts_voxel_layer=dict(max_num_points=-1, max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicSimpleVFE',
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
)