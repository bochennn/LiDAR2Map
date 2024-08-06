from configs._base_.datasets.zd_od128 import CLASS_NAMES
_base_ = ['./cbgs_dv01_second_secfpn_dcn.py']

model = dict(
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[256, 256, 512],
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='MultiScaleFPN',
        in_channels=[256, 256, 512],
        out_channels=[512, 512, 512],
        upsample_strides=[2, 2, 2],
        out_indices=[0, 1, 2],
        conv_cfg=dict(type='Conv2d', bias=False),
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='MultiScaleCenterHead',
        in_channels=[512, 512, 512],
        tasks=[
            [dict(num_class=2, class_names=['person', 'traffic_cone'])],
            [
                dict(num_class=3, class_names=['tricycle', 'motorcycle', 'bicycle']),
                dict(num_class=3, class_names=['car', 'pickup_truck', 'construction_vehicle'])
            ],
            [dict(num_class=2, class_names=['bus', 'truck'])],
        ],
        out_size_factor=[4, 8, 16],
    )
)