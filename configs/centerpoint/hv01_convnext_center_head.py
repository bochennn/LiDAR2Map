_base_ = ['./hv01_second_center_head.py']

model = dict(
    pts_backbone=dict(
        type='SECONDSCConvNeXt',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False))
)