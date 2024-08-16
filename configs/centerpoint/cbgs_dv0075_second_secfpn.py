from configs.centerpoint.hv01_second_secfpn import data

_base_ = ['./cbgs_dv01_second_secfpn.py']

VOXEL_SIZE = [0.075, 0.075, 8.0]
POINT_CLOUD_RANGE = [-76, -76.8, -3.0, 116.0, 76.8, 5.0]
VOXEL_GRID_SIZE = [
    int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0] + 1e-9) / VOXEL_SIZE[0]),
    int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1] + 1e-9) / VOXEL_SIZE[1]),
    int((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2] + 1e-9) / VOXEL_SIZE[2]),
]

model = dict(
    pts_voxel_layer=dict(
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        feat_channels=(64),
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE),
    pts_middle_encoder=dict(
        in_channels=64,
        base_channels=64,
        output_channels=256,
        encoder_channels=((64, 64, 128), (128, 128, 256), (256, 256, 256), (256, 256)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0)),
        out_padding=(1, 0, 0),
        sparse_shape=[VOXEL_GRID_SIZE[2] + 1, VOXEL_GRID_SIZE[1], VOXEL_GRID_SIZE[0]],
    ),
    pts_bbox_head=dict(
        bbox_coder=dict(
            pc_range=POINT_CLOUD_RANGE[:2],
            voxel_size=VOXEL_SIZE[:2])),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=POINT_CLOUD_RANGE,
            voxel_size=VOXEL_SIZE,
        )
    )
)

search = lambda lst, val: next(d for d in lst if d.get('type') == val)

search(data['train']['pipeline'], 'PointsRangeFilter').update(
    point_cloud_range=POINT_CLOUD_RANGE)
search(data['train']['pipeline'], 'ObjectRangeFilter').update(
    point_cloud_range=POINT_CLOUD_RANGE)

# data = dict(
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=eval_pipeline),
#     test=dict(pipeline=eval_pipeline)
# )