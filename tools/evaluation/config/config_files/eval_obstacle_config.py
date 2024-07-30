from collections import OrderedDict
import sys

from shapely.geometry import Polygon


__all__ = ["eval_obstacle_config", "Category", "MatchMethod"]


class Category:
    Bus = "Bus"
    Car = "Car"
    Cone = "Cone"
    Cyclist = "Cyclist"
    Person = "Person"
    Truck = "Truck"


class MatchMethod:
    iou = "iou"
    iop = "iop"
    distance = "distance"
    LET_iou = "LET_iou"


# key region for hesai lidar
key_region_hesai = Polygon([[30, 6], [30, -6], [0, -6], [0, 6]])

# key region for front camera
key_region_fc = Polygon([[0, 0], [60, 60], [60, -60]])

# global roi for m2 lidar
global_roi_m2 = Polygon([[0, 0], [120, 207.8], [120, -207.8]])


eval_obstacle_config = OrderedDict({
    "iou_threshold": {
        Category.Bus: 0.3,
        Category.Car: 0.5,
        Category.Cone: 0.2,
        Category.Cyclist: 0.3,
        Category.Person: 0.3,
        Category.Truck: 0.5
        },
    "distance_threshold": {
        Category.Bus: {"lat": 1, "lon": 4},
        Category.Car: {"lat": 1, "lon": 4},
        Category.Cone: {"lat": 1, "lon": 4},
        Category.Cyclist: {"lat": 1, "lon": 4},
        Category.Person: {"lat": 1, "lon": 4},
        Category.Truck: {"lat": 1, "lon": 4},
        },

    # "global_roi": global_roi_m2,
    "score_threshold": 0.3,
    "match_method": MatchMethod.iou,
    # "match_method": MatchMethod.LET_iou,
    "target_category": [
        Category.Bus,
        Category.Car,
        Category.Cone,
        Category.Cyclist,
        Category.Person,
        Category.Truck
    ],
    "distance_filter": {
        "0-30": [0, 30],
        "30-50": [30, 50],
        "50-70": [50, 70],
        "50-inf": [50, sys.maxsize],
        # "key_region": key_region_fc,
        "key_region": key_region_hesai,
        "overall": []
    },

    "gt_data_path": "example_data/eval_obstacle/gt_files",
    # "pred_data_path": "/home/wuchuanpan/Projects/BiTrack/data/zdrive/clip_1689126018500_forward/tracking",
    # "pred_data_path": "/mnt/data/autolabel/example_data/clip_1693892849299/det_for_tracking",
    # "pred_data_path": "/mnt/data/autolabel/code/autolabel_3d/tracking/kpi_test/perception/tracking",
    "pred_data_path": "example_data/eval_obstacle/pred_files",

    # "out_path": "/home/wuchuanpan/PycharmProjects/experiment/data/518700/eval_ret",
    # "lidar_path": "/home/wuchuanpan/PycharmProjects/experiment/data/518700/lidar",
    "failed_sample_visualize": False,
    "all_sample_visualize": False,
    # number of processes for parallel calculation
    "process_num": 1,
})
