from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from ...log_mgr import logger
from ...objects.ego.ego_clip import EgoClip
from ...objects.obstacle.objs.obstacle_clip_pred import ObstacleClipPred
from ...objects.obstacle.objs.pose_match_obj import PoseMatchObj
from ...objects.obstacle.parsers.attribute_tool import Attr as obstacle_attr
from ...objects.tf import TfClip
from ...utils import timeit
from ...utils.distance import points_distance
from ...utils.index_match import ts_match
from ..eval_tasks.eval_base import EvalBase
from ..eval_tasks.sub_tasks.obstacle_eval_heading import ObstacleEvalHeading
from ..eval_tasks.sub_tasks.obstacle_eval_location import ObstacleEvalLocation
from ..eval_tasks.sub_tasks.obstacle_eval_tracking import ObstacleEvalTrack
from ..eval_tasks.sub_tasks.obstacle_eval_velocity import ObstacleEvalVelocity

plt.ticklabel_format(style='plain', useOffset=False, axis='y')


class ObstacleEvalPose(EvalBase):
    @timeit
    def __init__(self, config):
        super().__init__(config,
                         EgoClip(config, data_path=config["data"]["gt"]["data_path"]),
                         ObstacleClipPred(config["data"]["pred"]["data_path"]),
                         PoseMatchObj,
                         config["match_method"])
        self.target_objs = \
            OrderedDict({"fusion": self.pred_obj,
                         "poly-mot": self.get_extra_obj(config["data"]["pred"].get("new_track")),
                         "radar_3_frames": self.get_extra_obj(config["data"]["pred"].get("radar_3_frames")),
                         "radar_10_frames": self.get_extra_obj(config["data"]["pred"].get("radar_10_frames")),
                         "detection_on_mc": self.get_extra_obj(config["data"]["pred"].get("detection_with_mc")),
                         "detection": self.get_extra_obj(config["data"]["pred"].get("detection_path")),
                         "recognition": self.get_extra_obj(config["data"]["pred"].get("recognition_path")),
                         "radar_front": self.get_extra_obj(config["data"]["pred"].get("radar_front")),
                         "radar_front_left": self.get_extra_obj(config["data"]["pred"].get("radar_front_left")),
                         "radar_front_right": self.get_extra_obj(config["data"]["pred"].get("radar_front_right")),
                         "radar_rear": self.get_extra_obj(config["data"]["pred"].get("radar_rear")),
                         "radar_rear_left": self.get_extra_obj(config["data"]["pred"].get("radar_rear_left")),
                         "radar_rear_right": self.get_extra_obj(config["data"]["pred"].get("radar_rear_right")),
                         "radar_front_raw": self.get_extra_obj(config["data"]["pred"].get("radar_front_raw")),
                         "radar_rear_raw": self.get_extra_obj(config["data"]["pred"].get("radar_rear_raw"))})
        self.out_path = config["out_path"]
        self.match_pair_list = []
        self.ego_obj = EgoClip(config, data_path=config["data"]["ego"]["data_path"])
        self.tf_obj = TfClip(config["data"]["ego"]["data_path"])
        # self.pc_obj = ObstacleClipPointCloud(config["data"]["pc"]["data_path"],
        #                                      config["out_path"])

    @staticmethod
    def get_extra_obj(data_path, obj=ObstacleClipPred):
        if data_path is None:
            return None
        else:
            return obj(data_path)

    def get_target_objs(self):
        pred_names = []
        pred_objs = []
        for name, obj in self.target_objs.items():
            if obj is not None:
                pred_names.append(name)
                pred_objs.append(obj)
        return pred_names, pred_objs

    @staticmethod
    def get_match_attr_func(gt_obj, target_obj):
        if hasattr(gt_obj, obstacle_attr.utm_pos_x) and hasattr(target_obj, obstacle_attr.utm_pos_x):
            logger.info("both objs have utm coordination position, use utm position to match")
            return lambda x: x.get_center_2d_utm()
        elif hasattr(gt_obj, obstacle_attr.lidar_pos_x) and hasattr(target_obj, obstacle_attr.lidar_pos_x):
            logger.info("both objs have lidar coordination position, use lidar position to match")
            return lambda x: x.get_center_2d_lidar()
        elif hasattr(gt_obj, obstacle_attr.utm_pos_x) and hasattr(target_obj, obstacle_attr.lidar_pos_x):
            return lambda x: x.get_center_2d_lidar()
        else:
            raise ValueError("None of utm position or lidar position was found")

    def get_match_pair_list(self):
        match_pair_list = []
        names, objs = self.get_target_objs()
        ts_lists = [obj.get_ts_list() for obj in objs]
        ts_lists = [self.gt_obj.get_ts_list(), self.ego_obj.get_ts_list()] + ts_lists
        matched_ts_list = ts_match(ts_lists, verbose=True)
        loc2lidar = self.tf_obj.local_to_sensor(self.tf_obj.sensor_name.lidar_main)
        match_funcs = [self.get_match_attr_func(self.gt_obj.data, obj.data) for obj in objs]
        target_track_id = dict()
        for gt_ts, ego_ts, *pred_ts_group in matched_ts_list:
            gt_instant = self.gt_obj.get_frame_obj_by_ts(gt_ts)
            ego_instant = self.ego_obj.get_frame_obj_by_ts(ego_ts)
            tf_instant = self.tf_obj.get_frame_obj_by_nearest_ts(gt_ts)
            world2loc = tf_instant.world_to_loc()
            world2lidar = loc2lidar.dot(world2loc)
            gt_instant.cal_lidar_pose(world2lidar)
            ego_instant.cal_lidar_pose(world2lidar)
            matched_pred_instants = []
            logger.info("="*20)
            logger.info("name: gt, ts: {}, lidar_pos_x: {}, lidar_pos_y: {}".format(gt_ts,
                                                                                    gt_instant.get_lidar_pos_x(),
                                                                                    gt_instant.get_lidar_pos_y()))
            logger.info("ego_speed: [{}, {}]".format(ego_instant.get_lidar_abs_vel_x(), ego_instant.get_lidar_abs_vel_y()))
            for idx, ts in enumerate(pred_ts_group):
                tf_instant = self.tf_obj.get_frame_obj_by_nearest_ts(ts)
                world2loc = tf_instant.world_to_loc()
                world2lidar = loc2lidar.dot(world2loc)
                match_func = match_funcs[idx]
                pred_instants = objs[idx].get_frame_obj_by_ts(ts).get_instant_objects()
                distance_list = [points_distance(match_func(pred_instant), match_func(gt_instant))
                                 for pred_instant in pred_instants]
                matched_index = np.argmin(distance_list)
                matched_distance = distance_list[matched_index]
                if matched_distance < 3:
                    match_instant = pred_instants[matched_index]
                else:
                    continue
                # match_instant = min(pred_instants, key=lambda x: points_distance(match_func(x),
                #                                                                  match_func(gt_instant)))
                match_instant.cal_lidar_pose(world2lidar, source_ego=ego_instant)
                matched_pred_instants.append((names[idx], match_instant, points_distance(match_func(match_instant),
                                                                                         match_func(gt_instant))))
                logger.info("name: {}, ts:{}, lidar_pos_x: {}, lidar_pos_y: {}".format(names[idx],
                                                                                 ts,
                                                                                 match_instant.get_lidar_pos_x(),
                                                                                 match_instant.get_lidar_pos_y()))
                logger.info("utm velocity x: {}, utm velocity y: {}".format(match_instant.get_utm_abs_vel_x(),
                                                                      match_instant.get_utm_abs_vel_y()))
                target_track_id[names[idx]] = match_instant.get_track_id()
            match_pair_list.append(self.match_obj(gt_instant, matched_pred_instants))
        logger.info("track_id in different ret: {}".format(target_track_id))
        return match_pair_list

    def evaluate_target_track(self, target_name):
        obstacle_match_pair_list = []
        for match_obj in self.match_pair_list:
            obstacle_match_pair_list.append(match_obj.create_obstacle_match(target_name))
        ObstacleEvalTrack(obstacle_match_pair_list).run()

    def evaluate_track(self):
        logger.info("=" * 60)
        logger.info("tracking of fusion:\n")
        self.evaluate_target_track("fusion")
        logger.info("="*60)
        logger.info("tracking of recognition\n")
        self.evaluate_target_track("recognition")

    def run(self):
        self.match_pair_list = self.get_match_pair_list()
        ObstacleEvalLocation(self.match_pair_list, self.out_path).run()
        ObstacleEvalVelocity(self.match_pair_list, self.out_path).run()
        ObstacleEvalHeading(self.match_pair_list, self.out_path).run()
        self.evaluate_track()
