from functools import partial
from itertools import groupby
from collections import OrderedDict

import pandas as pd

from tasks.eval_tasks.eval_base import EvalBase
from utils.result_formatter import print_result
from objects.obstacle.objs.obstacle_match_obj import ObstacleMatchObj
# from utils import timeit
from objects.obstacle.objs.obstacle_clip_gt import ObstacleClipGt
from objects.obstacle.objs.obstacle_clip_pred import ObstacleClipPred
# from objects.obstacle.objs.obstacle_point_cloud_obj import ObstacleClipPointCloud
# from tasks.eval_tasks.sub_tasks.obstacle_eval_tracking import ObstacleEvalTrack
# from tasks.eval_tasks.sub_tasks.obstacle_eval_bbox import ObstacleEvalBbox
from tasks.eval_tasks.match_tasks.obstacle_matcher import Matcher, FrameMatchStatus
from tasks.eval_tasks.sub_tasks.obstacle_eval_visualization import ObstacleFailedVisual
# from tasks.qa_tasks.annotation_qa import AnnotationQa
# from tasks.corner_case_task.heading_obnormal_case import HeadingAbnormalChecker
# from tasks.eval_tasks.sub_tasks.obstacle_eval_data_distribution import ObstacleSampleDistribution
# from log_mgr import logger


class ObstacleEvalCluster(EvalBase):
    def __init__(self, config):
        super().__init__(config,
                         ObstacleClipGt(config["gt_data_path"]),
                         ObstacleClipPred(config["pred_data_path"], config["score_threshold"]),
                         ObstacleMatchObj)
        self.out_path = config["out_path"] if "out_path" in config else None
        self.lidar_path = config["lidar_path"] if "lidar_path" in config else None
        self.test_name = config["test_name"] if "test_name" in config else None
        self.iop_thrs = config["iop_threshold"]
        self.min_distance_thrs = config["min_distance_threshold"]
        self.distance_percentage = config["distance_percentage_threshold"]
        self.thrs_map = {"iop_threshold": self.iop_thrs,
                         "min_dist_thrs": self.min_distance_thrs,
                         "dist_percentage": self.distance_percentage}
        self.category = list(self.iop_thrs.keys())
        self.distance = config["distance_filter"]
        self.match_method = config["match_method"]
        self.global_range = config["global_roi"] if "global_roi" in config else None
        self.failed_sample_vis = config["failed_sample_visualize"]
        self.all_sample_vis = config["all_sample_visualize"]
        self.match_obj = self.init_match_obj(self.match_method)
        self.match_pair_list = None
        self.target_tracks = None

    def init_match_obj(self, match_method):
        return partial(self.base_match_obj,
                       threshold_map=self.thrs_map,
                       match_method=match_method,
                       global_range=self.global_range)

    def get_match_pairs(self):
        matcher = Matcher(self.gt_obj, self.pred_obj, match_method=self.match_method, category=self.category)
        match_info_list = matcher.match()
        return matcher.create_match_pairs(match_info_list, self.match_obj)

    def cluster_metrics(self, category, distance_name):
        distance_value = self.distance.get(distance_name)
        match_pair_list = [match for match in self.match_pair_list
                           if match.get_gt_category() == category and match.within_distance_region(distance_value)]
        match_pair_list = sorted(match_pair_list, key=lambda x: x.gt_instant.get_uuid())
        gt_num = 0
        tp_num = 0

        for gt_uuid, target_match_pair_list in groupby(match_pair_list, key=lambda x: x.gt_instant.get_uuid()):
            gt_num += 1
            for match in target_match_pair_list:
                if match.is_tp():
                    tp_num += 1
                    break
        fn_num = gt_num - tp_num
        recall = tp_num / (gt_num + 1e-6) * 100
        return OrderedDict(
            {
                "category": category,
                "distance": distance_name,
                "tp": tp_num,
                "fn": fn_num,
                "gt": gt_num,
                "recall": round(recall, 2)
            }
        )

    def result_process(self, results):
        result_table = pd.DataFrame(results)
        dimension_name = ["category", "distance"]
        gt_table = result_table.pivot(index="category", columns="distance", values="gt")
        for metric_name in result_table.columns:
            if metric_name not in dimension_name:
                metric_table = result_table.pivot(index="category", columns="distance", values=metric_name)
                metric_table.sort_index(inplace=True)
                if metric_name in ["recall"]:
                    metric_table.loc["mean"] = metric_table[gt_table > 0].mean().round(2)
                elif metric_name in ["tp", "gt", "fn"]:
                    metric_table.loc["total"] = metric_table.sum()
                sub_ret_str = print_result(metric_table, metric_name)
                print(sub_ret_str)

    def evaluating_cluster_obstacles(self):
        results = []
        for category in self.category:
            for distance_name in self.distance:
                results.append(self.cluster_metrics(category, distance_name))
        self.result_process(results)

    def visualize_failed_samples(self):
        match_pair_list = [match for match in self.match_pair_list if match.within_distance_region([]) and
                           match.get_category() in self.category]
        ObstacleFailedVisual(match_pair_list,
                             self.out_path,
                             self.lidar_path,
                             target="failed").start()

    def run(self):
        self.match_pair_list = self.get_match_pairs()
        self.evaluating_cluster_obstacles()
        self.visualize_failed_samples()
