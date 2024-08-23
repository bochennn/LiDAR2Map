CLASS_NAMES = [
    'car', 'pickup_truck', 'truck', 'construction_vehicle', 'bus',
    'tricycle', 'motorcycle', 'bicycle', 'person', 'traffic_cone'
]

DATASET_TYPE = 'ZDriveDataset'
INFO_ROOT = '/data/sfs_turbo/dataset/data_info'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT,
        ann_files=[
            f'{INFO_ROOT}/CITY-ONLY-3D-L2+-NON-E03_interval_4_clip_883_frame_35041.pkl',
            f'{INFO_ROOT}/HIGHWAY-23D-L2+-NON-E03_interval_18_clip_389_frame_15377.pkl']),
    val=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT,
        ann_files=[
            f'{INFO_ROOT}/HIGHWAY-23D-L2+-NON-E03_val_clip_217_frame_8566.pkl'],
        test_mode=True),
    test=dict(
        type=DATASET_TYPE, classes=CLASS_NAMES,
        data_root=INFO_ROOT,
        ann_files=[
            f'{INFO_ROOT}/HIGHWAY-23D-L2+-NON-E03_test_clip_1083_frame_42647.pkl'],
        test_mode=True)
)

evaluation = dict(interval=2)