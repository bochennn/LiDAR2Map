_base_ = ['./cbgs_dv01_second_secfpn.py']

model = dict(
    pts_bbox_head=dict(
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), iou=(1, 2)),
        loss_iou=dict(type='IoU3DLoss')
    )
)