import os
import pprint

from config.config import ConfigParser
from tasks.eval_tasks.obstacle_eval import ObstacleEval
from log_mgr import logger


if __name__ == "__main__":
    task_type = "3d_object"
    config = ConfigParser.parse(task_type)

    root_data_path = "/mnt/data/lidar_detection/results/m2test/detector_2024020219/inference_ret"
    root_out_path = "/mnt/data/lidar_detection/results/m2test/detector_2024020219/evaluate_ret"
    gt_data_path = "/mnt/data/lidar_detection/test_datasets/m2test_0118/annotation_m2"
    config["gt_data_path"] = gt_data_path
    for epoch in range(20, 41):
        test_name = "epoch_{}".format(epoch)
        pred_data_path = os.path.join(root_data_path, test_name, "detection")
        out_path = os.path.join(root_out_path, test_name)
        os.makedirs(out_path, exist_ok=True)
        config["test_name"] = test_name
        config["pred_data_path"] = pred_data_path
        config["out_path"] = out_path
        logger.info(pprint.pformat(config))
        task = ObstacleEval(config)
        task.start()
