from configs._base_.datasets.zd_od128 import CLASS_NAMES, DATASET_TYPE, INFO_ROOT, data
from configs.centerpoint.hv01_second_secfpn import train_pipeline

_base_ = ['./dv01_second_secfpn.py']

data = dict(
    train=dict(
        _delete_=True,
        type='CBGSWrapper',
        dataset=dict(
            type=DATASET_TYPE, classes=CLASS_NAMES,
            data_root=INFO_ROOT,
            ann_files=data['train']['ann_files'],
            pipeline=train_pipeline),
    ),
)
runner = dict(max_epochs=18)