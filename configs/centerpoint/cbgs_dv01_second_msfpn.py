_base_ = ['./cbgs_dv01_second_secfpn.py']

model = dict(
    pts_backbone=dict(
        out_channels=[256, 256, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2]),
    pts_neck=dict(
        _delete_=True,
        type='MultiScaleFPN',
        in_channels=[256, 256, 256],
        out_channels=[256, 256, 256],
        upsample_scales=[2, 2, 2],
        out_indices=[0, 1, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='bilinear', align_corners=False)),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=2, class_names=['person', 'traffic_cone']),
            dict(num_class=3, class_names=['tricycle', 'motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['car', 'pickup_truck']),
            dict(num_class=3, class_names=['bus', 'truck', 'construction_vehicle'])
        ],
        out_size_factor=[4, 8, 8, 16],
    )
)