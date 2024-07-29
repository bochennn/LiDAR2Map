_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/cosine.py',
    '../_base_/datasets/nus-3d.py'
]
custom_imports = dict(
    imports=['plugin.models.detectors.lidar2map'],
    allow_failed_imports=False)

class_names = ['divider', 'ped_crossing', 'boundary'] # 'divider', 'ped_crossing', 'boundary'

pts_range = [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0]
voxel_size = [0.15, 0.15, 8]
bev_grid_size = [
    int((pts_range[4] - pts_range[1]) // voxel_size[1]),
    int((pts_range[3] - pts_range[0]) // voxel_size[0]),
]
depth_range = [1.0, 30.0, 1.0]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='LiDAR2Map',
    pts_voxel_layer=dict(
        max_num_points=-1,
        point_cloud_range=pts_range,
        voxel_size=voxel_size,
        max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicPillarFeatureNet',
        in_channels=4,
        voxel_size=voxel_size,
        point_cloud_range=pts_range),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=bev_grid_size),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],
        out_channels=[64, 64],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        )),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='LSSViewTransformerV2',
        grid_config=dict(
            x=[pts_range[0], pts_range[3], voxel_size[0]],
            y=[pts_range[1], pts_range[4], voxel_size[1]],
            z=[pts_range[2], pts_range[5], voxel_size[2]],
            depth=depth_range,
        ),
        input_size=(448, 800),
        downsample=8,
        in_channels=256,
        out_channels=128),
    pts_fusion_layer=dict(
        type='PositionGuidedFusion',
        img_feats_channel=128,
        pts_feats_channel=128),
    pts_seg_head=dict(
        type='BEV_FPD',
        in_channels=128,
        out_channels=len(class_names) + 1),
    fusion_seg_head=dict(
        type='BEV_FPD',
        in_channels=128,
        out_channels=len(class_names) + 1)
)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=3),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, color_type='color'),
    dict(type='PrepareImageInputs',
         pre_scale=(0.5, 0.5), pre_crop=(-2, 0),
         rand_scale=(0.95, 1.05), rand_rotation=(-5.4, 5.4)),
    dict(type='LoadAnnotations3D', class_names=class_names,
         pts_range=pts_range, bev_grid_size=bev_grid_size, with_seg=True),
    dict(type='PointsRangeFilter', point_cloud_range=pts_range),
    dict(type='DefaultFormatBundle3D', class_names=[], with_label=False),
    dict(type='Collect3D',
         keys=['img', 'points', 'gt_semantic_seg'],
         meta_keys=['cam2img', 'lidar2cam', 'img_aug_matrix'])
]

eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=3),
    dict(type='LoadAnnotations3D', class_names=class_names,
         pts_range=pts_range, bev_grid_size=bev_grid_size, with_seg=True),
    dict(type='PointsRangeFilter', point_cloud_range=pts_range),
    dict(type='DefaultFormatBundle3D', class_names=[], with_label=False),
    dict(type='Collect3D', keys=['points', 'gt_semantic_seg'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='NuScenesSegmentDataset',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality),
    val=dict(
        type='NuScenesSegmentDataset',
        pipeline=eval_pipeline,
        classes=class_names,
        modality=input_modality)
)

optimizer = dict(lr=0.001)
runner = dict(max_epochs=32)

evaluation = dict(pipeline=eval_pipeline)