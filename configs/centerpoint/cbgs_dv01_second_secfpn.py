from configs._base_.datasets.zd_od128 import data as _data
from configs.centerpoint.hv01_second_secfpn import train_pipeline

_base_ = ['./dv01_second_secfpn.py']

data = dict(
    train=dict(
        _delete_=True,
        type='CBGSWrapper',
        dataset=dict(**_data['train'], pipeline=train_pipeline),
    ),
)

runner = dict(max_epochs=8)