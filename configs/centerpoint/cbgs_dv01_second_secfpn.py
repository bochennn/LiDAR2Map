from configs._base_.datasets.zd_od128 import CLASS_NAMES, DATASET_TYPE, INFO_ROOT
from configs.centerpoint.hv01_second_secfpn_baseline import train_pipeline

_base_ = ['./dv01_second_secfpn.py']

data = dict(
    train=dict(
        _delete_=True,
        type='CBGSWrapper',
        dataset=dict(
            type=DATASET_TYPE, classes=CLASS_NAMES,
            data_root=INFO_ROOT,
            ann_files=[
                f'{INFO_ROOT}/E03-CITY-20240702_infos_clip_272_frames_10737.pkl',
                f'{INFO_ROOT}/E03-HY-20240702_infos_clip_1250_frames_49491.pkl'],
            pipeline=train_pipeline)
    )
)
runner = dict(max_epochs=18)