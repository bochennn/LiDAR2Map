CLASS_NAMES = [
    'car', 'pickup_truck', 'truck', 'construction_vehicle', 'bus',
    'tricycle', 'motorcycle', 'bicycle', 'person', 'traffic_cone'
]

DATASET_TYPE = 'ZDriveDataset'
INFO_ROOT = 'data/zdrive'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT,
        ann_files=[
            f'{INFO_ROOT}/CITY-3D-0529_infos_clip_1134_frames_45116.pkl',
            f'{INFO_ROOT}/HY-3D-0529_interval_15_clip_396_frames_15504.pkl']),
    val=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT,
        ann_files=[
            f'{INFO_ROOT}/E03-CITY-20240702_interval_2_clip_136_frames_5371.pkl',
            f'{INFO_ROOT}/E03-HY-20240702_interval_30_clip_42_frames_1670.pkl'],
        test_mode=True),
    test=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT,
        ann_files=[
            f'{INFO_ROOT}/NON-E03-CITY-20240702_interval_2_clip_468_frames_18548.pkl',
            f'{INFO_ROOT}/NON-E03-HY-20240702_interval_30_clip_187_frames_7396.pkl'],
        test_mode=True)
)

evaluation = dict(interval=2)