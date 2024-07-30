from config.config_files.eval_obstacle_config import eval_obstacle_config
from config.config_files.eval_trafficlight_config import eval_trafficlight_config
from config.config_files.eval_obstacle_cluster_config import eval_obstacle_cluster_config


def obstacle_config_process(config):
    iou_threshold = config["iou_threshold"]
    distance_threshold = config["distance_threshold"]
    target_categories = config["target_category"]
    config["iou_threshold"] = {cate: iou_threshold[cate] for cate in target_categories}
    config["distance_threshold"] = {cate: distance_threshold[cate] for cate in target_categories}
    return config


def obstacle_cluster_config_process(config):
    iop_threshold = config["iop_threshold"]
    target_categories = config["target_category"]
    config["iop_threshold"] = {cate: iop_threshold[cate] for cate in target_categories}
    return config


class ConfigParser:
    config_table = {
        "3d_object": eval_obstacle_config,
        "trafficlight": eval_trafficlight_config,
        "cluster_object": eval_obstacle_cluster_config,
    }

    config_process_table = {
        "3d_object": obstacle_config_process,
        "cluster_object": obstacle_cluster_config_process
    }

    @staticmethod
    def parse(task_type):
        config = ConfigParser.config_table.get(task_type)
        process_func = ConfigParser.config_process_table.get(task_type)
        if process_func is not None:
            config = process_func(config)
        return config
