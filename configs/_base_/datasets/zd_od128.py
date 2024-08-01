CLASS_NAMES = [
    'car', 'pickup_truck', 'truck', 'construction_vehicle', 'bus',
    'tricycle', 'motorcycle', 'bicycle', 'person',  'traffic_cone'
]

DATASET_TYPE = 'ZDriveDataset'
INFO_ROOT = 'data/zdrive'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT, ann_file=f'{INFO_ROOT}/zdrive_infos_train.pkl'),
    val=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT, ann_file=f'{INFO_ROOT}/zdrive_infos_val.pkl',
        test_mode=True),
    test=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT, ann_file=f'{INFO_ROOT}/zdrive_infos_val.pkl',
        test_mode=True)
)

evaluation = dict(interval=2)