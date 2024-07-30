import os
from functools import partial

import simplejson as json

from objects.obstacle.objs.obstacle_clip_pred import ObstacleClipPred
from objects.obstacle.objs.obstacle_match_obj import ObstacleMatchObj
from objects.obstacle.parsers.attribute_tool import Attr
from tasks.eval_tasks.match_tasks.obstacle_matcher import Matcher, FrameMatchStatus
from tasks.eval_tasks.sub_tasks.obstacle_eval_tracking import ObstacleEvalTrack
from tasks.eval_tasks.sub_tasks.obstacle_track_frames_combine import TrackFrameCombine
from utils.pointcloud_ops import numpy_to_bin


class SizeConsistentChecker:
    def __init__(self, config):
        self.pred_data_path = config["data"]["pred"]["data_path"]
        self.tracking_data_path = config["data"]["tracking"]["data_path"]
        self.pose_data_path = config["data"]["pose"]["data_path"]
        self.out_path = config["out_path"]
        self.raw_lidar_path = config["lidar_path"]
        self.out_lidar_path = self.init_out_lidar_path()
        self.out_det_path = self.init_out_det_path()
        self.out_abnormal_path = self.init_out_abnormal_path()
        self.target_category = ["Car", "Truck", "Bus"]
        self.det_obj = ObstacleClipPred(self.pred_data_path)
        self.tracking_obj = ObstacleClipPred(self.tracking_data_path)
        self.match_obj = partial(ObstacleMatchObj,
                                 threshold_map=config["thrs_map"],
                                 match_method="iou")
        self.match_pair_list = []

    @staticmethod
    def init_target_path(target_path):
        os.makedirs(target_path, exist_ok=True)
        return target_path

    def init_out_lidar_path(self):
        return self.init_target_path(os.path.join(self.out_path, "lidar"))

    def init_out_det_path(self):
        return self.init_target_path(os.path.join(self.out_path, "label", "detection"))

    def init_out_abnormal_path(self):
        return self.init_target_path(os.path.join(self.out_path, "label", "fn"))

    @staticmethod
    def is_value_inconsistent(main_value, compare_value, thrs=0.3):
        return abs(main_value - compare_value) > main_value * thrs

    def size_inconsistent(self, track):
        abnormal_frame = []
        for crt_idx in range(1, track.get_track_frame_num() - 1):
            prev_idx = crt_idx - 1
            next_idx = crt_idx + 1
            prev_instant = track.data[prev_idx].pred_instant
            crt_instant = track.data[crt_idx].pred_instant
            next_instant = track.data[next_idx].pred_instant
            if prev_instant is not None and crt_instant is not None and next_instant is not None:
                if crt_instant.get_category() not in self.target_category:
                    continue
                if self.is_value_inconsistent(crt_instant.get_length(), prev_instant.get_length()) or \
                   self.is_value_inconsistent(crt_instant.get_width(), prev_instant.get_width()) or \
                   self.is_value_inconsistent(crt_instant.get_length(), next_instant.get_length()) or \
                   self.is_value_inconsistent(crt_instant.get_width(), next_instant.get_width()):
                    abnormal_frame.append(crt_idx)
        return abnormal_frame

    @staticmethod
    def dump_visual_objs(out_path, visual_objs, ts, track_id):
        with open(os.path.join(out_path, "{}_{}.json".format(ts, track_id)), 'w') as f:
            json.dump(visual_objs, f)

    @staticmethod
    def dump_lidar_points(out_path, lidar_points, ts, track_id):

        numpy_to_bin(lidar_points, os.path.join(out_path, "{}_{}.bin".format(ts, track_id)))

    def link_lidar_file(self, ts, track_id):
        raw_lidar_file = os.path.join(self.raw_lidar_path, "{}.bin".format(ts))
        target_lidar_file = os.path.join(self.out_lidar_path, "{}_{}.bin".format(ts, track_id))
        os.symlink(raw_lidar_file, target_lidar_file)

    def search_size_inconsistent_case(self, tracks):
        target_tracks = []
        abnormal_index_list = []
        for track_id, track in tracks.items():
            if track.get_track_frame_num() > 5:
                abnormal_index = self.size_inconsistent(track)
                if len(abnormal_index) > 0:
                    target_tracks.append(track)
                    abnormal_index_list.append(abnormal_index)

        for track, abnormal_index in zip(target_tracks, abnormal_index_list):
            instant_list = [match.pred_instant for match in track.data if match.pred_valid()]
            visual_objs, lidar_points, chosen_ts = \
                TrackFrameCombine(instant_list, self.pose_data_path, self.raw_lidar_path).start()
            normal_objs = [visual_objs[idx] for idx, instant in enumerate(instant_list) if idx not in abnormal_index]
            abnormal_objs = [visual_objs[idx] for idx in abnormal_index]
            self.dump_visual_objs(self.out_det_path, normal_objs, chosen_ts, track.get_track_id())
            self.dump_visual_objs(self.out_abnormal_path, abnormal_objs, chosen_ts, track.get_track_id())
            # self.dump_lidar_points(self.out_lidar_path, lidar_points, chosen_ts, track.get_track_id())
            self.link_lidar_file(chosen_ts, track.get_track_id())

    def start(self):
        matcher = Matcher(self.tracking_obj, self.det_obj, category=self.target_category)
        match_info_list = matcher.match()
        self.match_pair_list = matcher.create_match_pairs(match_info_list, self.match_obj)
        tracks = ObstacleEvalTrack.extract_track_from_match(self.match_pair_list, "gt")
        self.search_size_inconsistent_case(tracks)
