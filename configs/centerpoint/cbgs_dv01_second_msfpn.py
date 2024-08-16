_base_ = ['./cbgs_dv01_second_secfpn.py']

model = dict(
    pts_backbone=dict(
        out_channels=(128, 256, 256),
        layer_nums=(5, 5, 3),
        layer_strides=(1, 2, 2)),
    pts_neck=dict(
        _delete_=True,
        type='MSFPN',
        in_channels=(128, 256, 256),
        out_channels=(512, 512, 512),
        upsample_scales=(2, 2, 2),
        out_indices=(1, 2, 2, 0, 0),
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='bilinear', align_corners=False)),
    pts_bbox_head=dict(
        out_size_factor=[8, 16, 16, 4, 4],
    )
)