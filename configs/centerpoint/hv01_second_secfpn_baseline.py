from configs._base_.datasets.zd_od128 import CLASS_NAMES
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/cyclic.py',
    '../_base_/datasets/zd_od128.py'
]
custom_imports = dict(imports=['plugin.models'], allow_failed_imports=False)

VOXEL_SIZE = [0.1, 0.1, 0.2]
# _point_cloud_range = [-82.4, -57.6, -3.0, 122.4, 57.6, 5.0]
POINT_CLOUD_RANGE = [-82.4, -76.8, -3.0, 122.4, 76.8, 5.0]
# 2048, 1536, 40
VOXEL_GRID_SIZE = [
    int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0] + 1e-9) / VOXEL_SIZE[0]),
    int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1] + 1e-9) / VOXEL_SIZE[1]),
    int((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2] + 1e-9) / VOXEL_SIZE[2]),
]

BATCH_SIZE = 2
BASE_LR = 1e-4
EPOCHS = 36

###############################################################################
# MODEL
###############################################################################
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=10, max_voxels=(90000, 120000),
        voxel_size=VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[VOXEL_GRID_SIZE[2] + 1, VOXEL_GRID_SIZE[1], VOXEL_GRID_SIZE[0]],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        class_names=CLASS_NAMES,
        tasks=[ # exact same order with CLASS_NAMES
            dict(num_class=1, class_names=['car']),
            dict(num_class=3, class_names=['pickup_truck', 'truck', 'construction_vehicle']),
            dict(num_class=1, class_names=['bus']),
            dict(num_class=3, class_names=['tricycle', 'motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['person', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=POINT_CLOUD_RANGE[:2],
            out_size_factor=8,
            voxel_size=VOXEL_SIZE[:2],
            post_center_range=[-85.2, -80.0, -10.0, 125.2, 80.0, 10.0],
            max_num=500,
            score_threshold=0.1,
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=POINT_CLOUD_RANGE,
            grid_size=VOXEL_GRID_SIZE,
            voxel_size=VOXEL_SIZE,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-85.2, -80.0, -10.0, 125.2, 80.0, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=VOXEL_SIZE[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2))
)

###############################################################################
# DATASET
###############################################################################
train_pipeline = [
    dict(type='LoadPointsFromFile',
         load_dim=4, use_dim=4, convert_ego=True),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter',
         point_cloud_range=POINT_CLOUD_RANGE,
         min_point_cloud_range=[-1, -1, 4, 1]),
    dict(type='ObjectRangeFilter',
         point_cloud_range=POINT_CLOUD_RANGE),
    dict(type='ObjectNameFilter', classes=CLASS_NAMES),
    dict(type='DefaultFormatBundle3D', class_names=CLASS_NAMES),
    dict(type='Collect3D',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
eval_pipeline = [
    dict(type='LoadPointsFromFile',
            load_dim=4, use_dim=4, convert_ego=True),
    dict(type='PointsRangeFilter',
            point_cloud_range=POINT_CLOUD_RANGE,
            min_point_cloud_range=[-1, -1, 4, 1]),
    dict(type='DefaultFormatBundle3D', class_names=CLASS_NAMES),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=BATCH_SIZE,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=eval_pipeline),
    test=dict(pipeline=eval_pipeline)
)

optimizer = dict(lr=BASE_LR)
runner = dict(max_epochs=EPOCHS)
evaluation = dict(pipeline=eval_pipeline)