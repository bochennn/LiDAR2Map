_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/cosine.py',
    '../_base_/datasets/nus-3d.py'
]
custom_imports = dict(
    imports=['plugin.models.detectors.lidar2map'], allow_failed_imports=False)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

class_names = []

data_root = '/data/sfs_turbo/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')

model = dict(
    type='LiDAR2Map',
    data_conf=dict(
        num_channels=2,
        image_size=(512, 960),
        xbound=[-30.0, 30.0, 0.15],
        ybound=[-15.0, 15.0, 0.15],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        thickness=5,
        angle_class=36,
        cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        # 'final_dim': (512, 960),
    )
    # data_preprocessor=dict(
    #     type='Det3DDataPreprocessor',
    #     mean=[123.675, 116.28, 103.53],
    #     std=[58.395, 57.12, 57.375],
    #     bgr_to_rgb=False),
    # pts_voxel_layer=dict(
    #     max_num_points=-1,
    #     point_cloud_range=point_cloud_range,
    #     voxel_size=voxel_size,
    #     max_voxels=(-1, -1)
    # ),
    # pts_voxel_encoder=dict(
    #     type='DynamicPillarFeatureNet',
    #     in_channels=4,
    #     voxel_size=voxel_size,
    #     point_cloud_range=point_cloud_range),
)

train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5,
         file_client_args=file_client_args),
    dict(type='LoadPointsFromMultiSweeps',
         sweeps_num=3,
         file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=True,
         color_type='color'),
    dict(type='PrepareImageInputs',
         # input_size=(512, 960),
         pre_scale=(0.6, 0.6),
         pre_crop=(-28, 0),
        #  rand_scale=(0.95, 1.05),
        #  rand_rotation=(-5.4, 5.4),
         rand_flip=False),
    dict(type='LoadAnnotations3D',
         with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D',
         class_names=class_names,
         with_label=False),
    dict(type='Collect3D',
         keys=['img', 'points'],
         meta_keys=['cam2img', 'lidar2cam', 'img_aug_matrix'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality
    )
)

# lr = 0.001
# optimizer = dict(lr=lr)