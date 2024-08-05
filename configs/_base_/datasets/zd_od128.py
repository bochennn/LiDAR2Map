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
            f'{INFO_ROOT}/E03-CITY-20240702_infos_clip_272_frames_10737.pkl',
            f'{INFO_ROOT}/E03-HY-20240702_infos_clip_1250_frames_49491.pkl']),
    val=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT,
        ann_files=[
            f'{INFO_ROOT}/CITY-3D-0529_minibatch_val.pkl',
            f'{INFO_ROOT}/HY-3D-0529_minibatch_val.pkl'],
        test_mode=True),
    test=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT,
        ann_files=[
            f'{INFO_ROOT}/NON-E03-CITY-20240702_minibatch_test.pkl',
            f'{INFO_ROOT}/NON-E03-HY-20240702_minibatch_test.pkl'],
        test_mode=True)
)

evaluation = dict(interval=2)