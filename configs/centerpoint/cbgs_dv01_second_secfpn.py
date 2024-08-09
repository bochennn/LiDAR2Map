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
            # ann_files=data['train']['ann_files'],
            ann_files=[
                f'{INFO_ROOT}/E03-CITY-20240702_infos_clip_2_frames_78.pkl',
                f'{INFO_ROOT}/E03-HY-20240702_infos_clip_1_frames_39.pkl'],
            pipeline=train_pipeline),
    ),
    val=dict(
        ann_files=[
                f'{INFO_ROOT}/CITY-3D-0529_infos_clip_1_frames_40.pkl',
                f'{INFO_ROOT}/HY-3D-0529_infos_clip_1_frames_40.pkl']),
    test=dict(
        ann_files=[
            f'{INFO_ROOT}/NON-E03-CITY-20240702_infos_clip_1_frames_40.pkl',
            f'{INFO_ROOT}/NON-E03-HY-20240702_infos_clip_1_frames_40.pkl'])
)
runner = dict(max_epochs=18)