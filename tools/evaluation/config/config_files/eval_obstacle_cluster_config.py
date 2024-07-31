from collections import OrderedDict
# import sys

from shapely.geometry import Polygon

from .eval_obstacle_config import Category, MatchMethod


__all__ = ["eval_obstacle_cluster_config"]


global_roi_delivery_vehicle = Polygon([[0, 0], [5.77, 5], [50, 5], [50, -5], [5.77, -5]])


eval_obstacle_cluster_config = OrderedDict({
    "iop_threshold": {
        Category.Bus: 0.3,
        Category.Car: 0.3,
        Category.Cone: 0.3,
        Category.Cyclist: 0.3,
        Category.Person: 0.3,
        Category.Truck: 0.3
        },


    "global_roi": global_roi_delivery_vehicle,
    "score_threshold": 0.3,
    "match_method": MatchMethod.iop,
    "min_distance_threshold": 0.3,
    "distance_percentage_threshold": 0.1,
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
        "overall": []
    },
    "gt_data_path": "/mnt/data/lidar_detection/test_datasets/20230717_clip155/gt_split",
    "pred_data_path": "/mnt/data/tmp/20240507_delivery_vehicle/kpi_res_all",
    "out_path": "/mnt/data/lidar_detection/results/test_dataset_20231028_tracking_improve/output_refine_size_position_when_delta_less_30/eval_ret",
    "failed_sample_visualize": False,
    "all_sample_visualize": False
})
