_base_ = ['./dv01_second_center_head.py']

voxel_size = [0.1, 0.1, 0.2]
point_cloud_range = [-82.4, -57.6, -3.0, 122.4, 57.6, 5.0]

model = dict(
    pts_encoder=dict(
        type='VoxelSetAbstraction',
        num_keypoints=2048,
        fused_out_channel=128,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        voxel_sa_cfgs_list=[
            dict(
                type='StackedSAModuleMSG',
                in_channels=16,
                scale_factor=1,
                radius=(0.4, 0.8),
                sample_nums=(16, 16),
                mlp_channels=((16, 16), (16, 16)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=32,
                scale_factor=2,
                radius=(0.8, 1.2),
                sample_nums=(16, 32),
                mlp_channels=((32, 32), (32, 32)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=64,
                scale_factor=4,
                radius=(1.2, 2.4),
                sample_nums=(16, 32),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=64,
                scale_factor=8,
                radius=(2.4, 4.8),
                sample_nums=(16, 32),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True)
        ],
        rawpoints_sa_cfgs=dict(
            type='StackedSAModuleMSG',
            in_channels=1,
            radius=(0.4, 0.8),
            sample_nums=(16, 16),
            mlp_channels=((16, 16), (16, 16)),
            use_xyz=True),
        bev_feat_channel=256,
        bev_scale_factor=8),
    pts_roi_head=dict(
        type='PVRCNNRoiHead',
        semantic_head=dict(
            type='ForegroundSegmentationHead',
            in_channels=640,
            extra_width=0.1,
            loss_seg=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                activated=True,
                loss_weight=1.0)),
        bbox_roi_extractor=dict(
            type='Batch3DRoIGridExtractor',
            grid_size=6,
            roi_layer=dict(
                type='StackedSAModuleMSG',
                in_channels=128,
                radius=(0.8, 1.6),
                sample_nums=(16, 16),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True,
                pool_mod='max'),
        ),
        bbox_head=dict(
            type='PVRCNNBBoxHead',
            in_channels=128,
            grid_size=6,
            num_classes=3,
            class_agnostic=True,
            shared_fc_channels=(256, 256),
            reg_channels=(256, 256),
            cls_channels=(256, 256),
            dropout_ratio=0.3,
            with_corner_loss=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))),
    train_cfg=dict(
        rcnn=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25
        )
    )
)