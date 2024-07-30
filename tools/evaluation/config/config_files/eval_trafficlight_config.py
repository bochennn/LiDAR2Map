__all__ = ["eval_trafficlight_config"]


eval_trafficlight_config = {
    "iou_threshold": {
        "vertical_light": 0.5,
        "horizontal_light": 0.5,
        "single_light": 0.5
    },
    "target_shape": [
        "vertical_light",
        "horizontal_light",
        "single_light"
    ],
    "img_width": 160,
    "img_height": 160,
    "gt_data_path": "/mnt/data/trafficlight/test_datasets/20240124_front_wide/annotation_formated",
    "pred_data_path": "/home/wuchuanpan/Projects/trafficlight_projects/trafficlight_0126/trafficlight/detect/yolov7/runs/detect/exp4/jsons",
    "img_data_path": "/mnt/data/trafficlight/test_datasets/20240124_front_wide/yolo_fw_test_data_0124/images"
}
