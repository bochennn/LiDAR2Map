from collections import OrderedDict


label_map = {
    17: 'empty',
    0: 'noise',
    16: 'vegetation',
    15: 'manmade',
    14: 'terrain',
    13: 'sidewalk',
    12: 'other_flat',
    11: 'driveable_surface',
    10: 'truck',
    9: 'trailer',
    6: 'motorcycle',
    5: 'construction_vehicle',
    4: 'car',
    3: 'bus',
    2: 'bicycle',
    8: 'traffic_cone',
    1: 'barrier',
    7: 'pedestrian'
}


eval_occupancy_config = OrderedDict(
    {
        "gt_data_path": 0,
        "pred_data_path": 0,
        "ignore_label": 17,
        "range_max": [51.2, 51.2, 3],
        "range_min": [-51.2, -51.2, -5],
        "grid_size": [100, 100, 8],
        "label_map": label_map
    }
)
