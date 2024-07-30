_base_ = [
    '../_base_/schedules/cosine.py', '../_base_/default_runtime.py',
    '../_base_/datasets/zd-od-128.py'
]
custom_imports = dict(
    imports=['plugin.models.detectors.centerpoint'],
    allow_failed_imports=False)

voxel_size = [0.1, 0.1, 0.2]
pts_range = [-82.4, -57.6, -3.0, 122.4, 57.6, 5.0]
grid_size = [
    int((pts_range[3] - pts_range[0]) / voxel_size[0]),
    int((pts_range[4] - pts_range[1]) / voxel_size[1]),
    int((pts_range[5] - pts_range[2]) / voxel_size[2]),
]

class_names = [
    'car', 'pickup_truck', 'truck', 'construction_vehicle', 'bus',
    'tricycle', 'motorcycle', 'bicycle', 'person', 'traffic_cone'
]

model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size,
        max_voxels=(90000, 12000), point_cloud_range=pts_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[grid_size[2] + 1, grid_size[1], grid_size[0]],
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
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
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
        tasks=[
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
            pc_range=pts_range[:2],
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            post_center_range=[-85.2, -65.2, -10.0, 125.2, 65.2, 10.0],
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
            point_cloud_range=pts_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-85.2, -65.2, -10.0, 125.2, 65.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))

train_pipeline = [
    dict(type='LoadPointsFromFile',
         load_dim=4, use_dim=4, convert_ego=True),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=pts_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

eval_pipeline = [
    dict(type='LoadPointsFromFile',
         load_dim=4, use_dim=4, convert_ego=True),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=pts_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=eval_pipeline)
)

optimizer = dict(lr=0.001)
runner = dict(max_epochs=64)