_base_ = ['./hv01_second_center_head.py']

model = dict(
    pts_voxel_layer=dict(max_num_points=-1, max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(type='DynamicSimpleVFE'),
)