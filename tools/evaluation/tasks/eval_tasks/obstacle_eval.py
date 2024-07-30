# import shutil
from functools import partial

from tasks.eval_tasks.eval_base import EvalBase
from objects.obstacle.objs.obstacle_match_obj import ObstacleMatchObj
from utils import timeit
from objects.obstacle.objs.obstacle_clip_gt import ObstacleClipGt
from objects.obstacle.objs.obstacle_clip_pred import ObstacleClipPred
# from objects.obstacle.objs.obstacle_point_cloud_obj import ObstacleClipPointCloud
from tasks.eval_tasks.sub_tasks.obstacle_eval_tracking import ObstacleEvalTrack
from tasks.eval_tasks.sub_tasks.obstacle_eval_bbox import ObstacleEvalBbox
from tasks.eval_tasks.match_tasks.obstacle_matcher import Matcher, FrameMatchStatus
from tasks.eval_tasks.sub_tasks.obstacle_eval_visualization import ObstacleFailedVisual
# from tasks.qa_tasks.annotation_qa import AnnotationQa
# from tasks.corner_case_task.heading_obnormal_case import HeadingAbnormalChecker
from tasks.eval_tasks.sub_tasks.obstacle_eval_data_distribution import ObstacleSampleDistribution
from log_mgr import logger


class ObstacleEval(EvalBase):
    def __init__(self, config):
        super().__init__(config,
                         ObstacleClipGt(config["gt_data_path"]),
                         ObstacleClipPred(config["pred_data_path"], config["score_threshold"]),
                         ObstacleMatchObj)
        self.out_path = config["out_path"] if "out_path" in config else None
        self.lidar_path = config["lidar_path"] if "lidar_path" in config else None
        self.test_name = config["test_name"] if "test_name" in config else None
        self.iou_thrs = config["iou_threshold"]
        self.distance_thrs = config["distance_threshold"]
        self.thrs_map = {"iou": self.iou_thrs,
                         "distance": self.distance_thrs}
        self.category = list(self.iou_thrs.keys())
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
        matcher = Matcher(self.gt_obj, self.pred_obj, match_method=self.match_method, category=self.category,
                          process_num=self.config["process_num"])
        match_info_list = matcher.match()
        return matcher.create_match_pairs(match_info_list, self.match_obj)

    @timeit
    def evaluating_bbox(self):
        return ObstacleEvalBbox(self.match_pair_list,
                                self.category,
                                self.distance,
                                self.thrs_map,
                                self.out_path,
                                self.test_name,
                                process_num=self.config["process_num"]).run()

    def evaluating_track(self):
        return ObstacleEvalTrack(self.match_pair_list).run()

    def evaluating_data_distribution(self):
        ObstacleSampleDistribution(self.match_pair_list, self.out_path, self.lidar_path).start()

    def visualize_all_samples(self):
        if self.lidar_path is None:
            logger.warning("lidar path is None, visualize func will not be processed")
        else:
            match_pair_list = [match for match in self.match_pair_list if match.within_distance_region([]) and
                               match.get_category() in self.iou_thrs]
            ObstacleFailedVisual(match_pair_list,
                                 self.out_path,
                                 self.lidar_path,
                                 target="all").start()

    def visualize_failed_samples(self):
        if self.lidar_path is None:
            logger.warning("lidar path is None, visualize func will not be processed")
        else:
            match_pair_list = [match for match in self.match_pair_list if match.within_distance_region([]) and
                               match.get_category() in self.iou_thrs]
            ObstacleFailedVisual(match_pair_list,
                                 self.out_path,
                                 self.lidar_path,
                                 target="failed").start()

    def run(self):
        self.match_pair_list = self.get_match_pairs()
        logger.info("number of match pairs: {}".format(len(self.match_pair_list)))
        self.evaluating_bbox()
        self.evaluating_track()
        if self.failed_sample_vis:
            self.visualize_failed_samples()
        if self.all_sample_vis:
            self.visualize_all_samples()
