import os
import simplejson as json
import pickle
from functools import partial
from collections import OrderedDict
import time

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from objects.obstacle.objs.obstacle_clip_pred import ObstacleClipPred
from tasks.eval_tasks.match_tasks.obstacle_matcher import Matcher
from tasks.eval_tasks.sub_tasks.obstacle_eval_tracking import ObstacleEvalTrack
from objects.obstacle.parsers.attribute_tool import Attr
from objects.obstacle.objs.obstacle_match_obj import ObstacleMatchObj
from utils.plot import get_ax4
from utils.transform import vector_transform
from log_mgr import logger


threshold_map = {"iou": {"Bus": 0.3,
                         "Car": 0.5,
                         "Cyclist": 0.3,
                         "Person": 0.3,
                         "Truck": 0.5}}


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


palette = {"refined": (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
           "forward": (0.45, 0.45, 0.45),
           "backward": (0.45, 0.45, 0.45)}

style_map = {"refined": "Solid",
             "forward": "Dashed",
             "backward": "Dashed"}


class VelocityRefiner:
    # def __init__(self, forward_data_path, backward_data_path, pose_data_path, ts_map_data_path, clip_name, root_out_path):
    def __init__(self, forward_data_path, backward_data_path, pose_data_path, clip_name, root_out_path):
        self.forward_data_path = forward_data_path
        self.backward_data_path = backward_data_path
        self.pose_data_path = pose_data_path
        self.forward_obj = ObstacleClipPred(forward_data_path)
        self.backward_obj = ObstacleClipPred(backward_data_path)
        self.pose_data = self.get_pose_data(pose_data_path)
        # self.ts_map = self.get_ts_map(ts_map_data_path)
        # self.backward_obj = self.align_backward_ts(self.backward_obj)
        # self.backward_obj = self.recover_velocity(self.backward_obj)
        self.match_obj = self.init_match_obj()
        self.forward_instant_map = self.get_forward_instant_map()
        self.refined_out_path = self.init_out_path(clip_name, root_out_path)

    @staticmethod
    def init_out_path(clip_name, root_out_path):
        out_path = os.path.join(root_out_path, clip_name, "refined_fusion")
        os.makedirs(out_path, exist_ok=True)
        return out_path

    @staticmethod
    def get_pose_data(pose_data_path):
        pose_record = dict()
        for file_name in sorted(os.listdir(pose_data_path)):
            if file_name.endswith(".json"):
                ts = os.path.splitext(file_name)[0]
                with open(os.path.join(pose_data_path, file_name), 'r') as f:
                    pose_data = json.load(f)
                pose_record[ts] = pose_data
        return pose_record

    def get_forward_instant_map(self):
        forward_map = OrderedDict()
        self.forward_obj.data["forward_utm_abs_vel_x"] = self.forward_obj.data["utm_abs_vel_x"]
        self.forward_obj.data["forward_utm_abs_vel_y"] = self.forward_obj.data["utm_abs_vel_y"]
        self.forward_obj.data["backward_utm_abs_vel_x"] = None
        self.forward_obj.data["backward_utm_abs_vel_y"] = None
        self.forward_obj.data["refined_utm_abs_vel_x"] = self.forward_obj.data["utm_abs_vel_x"]
        self.forward_obj.data["refined_utm_abs_vel_y"] = self.forward_obj.data["utm_abs_vel_y"]
        self.forward_obj.data["refined_lidar_abs_vel_x"] = self.forward_obj.data["lidar_abs_vel_x"]
        self.forward_obj.data["refined_lidar_abs_vel_y"] = self.forward_obj.data["lidar_abs_vel_y"]

        for ts in self.forward_obj.get_ts_list():
            frame_obj = self.forward_obj.get_frame_obj_by_ts(ts)
            for instant in frame_obj.get_instant_objects():
                forward_map[instant.get_uuid()] = instant
        return forward_map

    @staticmethod
    def init_match_obj():
        return partial(ObstacleMatchObj, threshold_map=threshold_map)

    def align_backward_ts(self, backward_obj):
        target_data = backward_obj.data
        target_data.index = target_data.index.map(lambda x: float(self.ts_map["rever2ori"]["{:.6f}".format(x)]))
        target_data.sort_index(inplace=True)
        backward_obj.data = target_data
        backward_obj.ts_list = backward_obj.data.index.unique().tolist()
        return backward_obj

    def recover_velocity(self, backward_obj):
        target_data = backward_obj.data
        target_data[Attr.utm_abs_vel_x] = target_data[Attr.utm_abs_vel_x] * -1
        target_data[Attr.utm_abs_vel_y] = target_data[Attr.utm_abs_vel_y] * -1
        backward_obj.data = target_data
        return backward_obj

    def get_ts_map(self, ts_map_data_path):
        with open(ts_map_data_path, 'rb') as f:
            ts_map = pickle.load(f)
        return ts_map

    @staticmethod
    def extract_track_data(target_data):
        tracks = dict()
        for track_id, track_data in target_data.groupby([Attr.track_id]):
            tracks[track_id] = track_data
        return tracks

    @staticmethod
    def extract_data_from_track(forward_track, backward_track):
        forward_series_list = [match.gt_instant.data for match in forward_track.data]
        backward_series_list = [match.pred_instant.data for match in backward_track.data]
        return pd.concat(forward_series_list, axis=1).T, pd.concat(backward_series_list, axis=1).T

    @staticmethod
    def get_intersection_data(forward_track, backward_track):
        index_inter = forward_track.index.intersection(backward_track.index)
        return forward_track.loc[index_inter], backward_track.loc[index_inter]

    def velocity_fusion(self, forward_data, backward_data, target_attr):
        refine_attr = "refined_{}".format(target_attr)
        backward_attr = "backward_{}".format(target_attr)
        forward_data[refine_attr] = forward_data[target_attr]
        forward_data[backward_attr] = backward_data[target_attr]

        index_inter = forward_data.index.intersection(backward_data.index)
        forward_attr = forward_data.loc[index_inter, target_attr].to_numpy()
        backward_attr = backward_data.loc[index_inter, target_attr].to_numpy()

        delta_attr = np.abs(forward_attr - backward_attr)
        sort_args = np.argsort(delta_attr)
        argmin = sort_args[0]
        if argmin <= 2 and len(sort_args) > 1 and abs(delta_attr[sort_args[1]] - delta_attr[argmin]) <= 1:
            argmin = sort_args[1]

        target_index = index_inter[argmin]
        before_index = index_inter[max(0, argmin-1)]
        after_index = index_inter[min(argmin + 1, len(index_inter) - 1)]

        first_part_mask = index_inter[index_inter < target_index]
        forward_data.loc[first_part_mask, refine_attr] = backward_data.loc[first_part_mask, target_attr]

        middle_value = np.mean([forward_data.loc[target_index, refine_attr],
                                backward_data.loc[target_index, target_attr]])

        forward_data.loc[target_index, refine_attr] = np.mean([backward_data.loc[before_index, target_attr],
                                                               middle_value,
                                                               forward_data.loc[after_index, refine_attr]])
        return forward_data

    def plot_velocity(self, forward_track_data):
        frame_seq = list(range(len(forward_track_data)))
        forward_track_data["frame_seq"] = frame_seq

        backward_valid_mask = forward_track_data["backward_" + Attr.utm_abs_vel_x].notna()
        plot_data = pd.concat([
            pd.DataFrame({"frame_seq": frame_seq,
                          "vel_x": forward_track_data[Attr.utm_abs_vel_x].tolist(),
                          "vel_y": forward_track_data[Attr.utm_abs_vel_y].tolist(),
                          "type": "forward"}),
            pd.DataFrame({"frame_seq": forward_track_data[backward_valid_mask]["frame_seq"].tolist(),
                          "vel_x": forward_track_data[backward_valid_mask]["backward_" + Attr.utm_abs_vel_x].tolist(),
                          "vel_y": forward_track_data[backward_valid_mask]["backward_" + Attr.utm_abs_vel_y].tolist(),
                          "type": "backward"}),
            pd.DataFrame({"frame_seq": frame_seq,
                          "vel_x": forward_track_data["refined_" + Attr.utm_abs_vel_x].tolist(),
                          "vel_y": forward_track_data["refined_" + Attr.utm_abs_vel_y].tolist(),
                          "type": "refined"}),
        ])
        track_id = forward_track_data[Attr.track_id].unique().tolist()[0]
        fig, ax1, ax2, ax3, ax4 = get_ax4()
        fig.suptitle(track_id)

        order = ["refined", "forward", "backward"]
        sns.lineplot(plot_data, x="frame_seq", y="vel_x", hue="type", style="type", style_order=order, ax=ax1)
        sns.lineplot(plot_data, x="frame_seq", y="vel_y", hue="type", style="type", style_order=order, ax=ax2)
        sns.lineplot(forward_track_data, x="frame_seq", y=Attr.utm_pos_x, ax=ax3)
        sns.lineplot(forward_track_data, x="frame_seq", y=Attr.utm_pos_y, ax=ax4)
        plt.show()
        plt.close(fig)

    def is_static(self, forward_data):
        linear_v = np.linalg.norm([forward_data[Attr.utm_abs_vel_x].tolist(),
                                   forward_data[Attr.utm_abs_vel_y].tolist()], axis=0)
        return np.max(linear_v) < 1

    def velocity_update(self, refined_data):
        for index, row_data in refined_data.iterrows():
            ts_str = "{:.6f}".format(row_data["measure_ts"])
            lidar2world = self.pose_data[ts_str]["lidar2world"]
            world2lidar = np.linalg.inv(lidar2world)
            refined_utm_velocity = [row_data["refined_utm_abs_vel_x"],
                                    row_data["refined_utm_abs_vel_y"]]
            refined_lidar_vel_x, refined_lidar_vel_y = vector_transform(refined_utm_velocity, world2lidar)

            uuid = row_data["uuid"]
            instant = self.forward_instant_map[uuid]
            instant.setattr("forward_utm_abs_vel_x", row_data["utm_abs_vel_x"])
            instant.setattr("forward_utm_abs_vel_y", row_data["utm_abs_vel_y"])
            if not np.isnan(row_data["backward_utm_abs_vel_x"]):
                instant.setattr("backward_utm_abs_vel_x", row_data["backward_utm_abs_vel_x"])
                instant.setattr("backward_utm_abs_vel_y", row_data["backward_utm_abs_vel_y"])
            instant.setattr("refined_utm_abs_vel_x", row_data["refined_utm_abs_vel_x"])
            instant.setattr("refined_utm_abs_vel_y", row_data["refined_utm_abs_vel_y"])
            instant.setattr("refined_lidar_abs_vel_x", refined_lidar_vel_x)
            instant.setattr("refined_lidar_abs_vel_y", refined_lidar_vel_y)

    def save_refined_data(self):
        for ts in self.forward_obj.get_ts_list():
            frame_obj = self.forward_obj.get_frame_obj_by_ts(ts)
            frame_obj.to_visual_json_with_velocity(self.refined_out_path)

    def refine(self):
        # step1. init track
        matcher = Matcher(self.forward_obj, self.backward_obj, category=list(threshold_map["iou"].keys()))
        match_info_list = matcher.match()
        match_pair_list = matcher.create_match_pairs(match_info_list, self.match_obj)
        forward_track = ObstacleEvalTrack.extract_track_from_match(match_pair_list, "gt")
        backward_track = ObstacleEvalTrack.extract_track_from_match(match_pair_list, "pred")

        # step2. find matched track
        track_match_record = []
        succeed_track = 0
        valid_num = 0
        failed_track_info = []
        for f_track_id, f_track in forward_track.items():
            if f_track.length < 5:
                continue
            valid_num += 1
            tp_dict = f_track.get_tp_dict()
            if len(tp_dict) > 0:
                succeed_track += 1
                matched_backward_track_id = max(tp_dict.keys(), key=lambda x: tp_dict[x])
                track_match_record.append(self.extract_data_from_track(forward_track[f_track_id],
                                                                       backward_track[matched_backward_track_id]))
            else:
                failed_track_info.append([f_track_id, f_track.length, f_track.get_category()])

        # step3. velocity fusion
        for forward_data, backward_data in track_match_record:
            if self.is_static(forward_data):
                continue
            forward_data = self.velocity_fusion(forward_data, backward_data, Attr.utm_abs_vel_x)
            forward_data = self.velocity_fusion(forward_data, backward_data, Attr.utm_abs_vel_y)
            self.velocity_update(forward_data)
            self.plot_velocity(forward_data)
        self.save_refined_data()


if __name__ == "__main__":
    # forward_path = "/mnt/data/legacy_data_playback/data_quality_verigy/B611N0/clip_1689304521000/ret_server/label/fusion"
    # backward_path = "/mnt/data/legacy_data_playback/data_quality_verigy/B611N0/clip_1689304521000/backward_offset_100/fusion"
    # pose_data_path = "/mnt/data/legacy_data_playback/data_quality_verigy/B611N0/clip_1689304521000/pose"
    # ts_map_path = "/mnt/data/legacy_data_playback/data_quality_verigy/B611N0/clip_1689304521000/reversed_record/lidar_measure_time_map.pickle"
    # refiner = VelocityRefiner(forward_path,
    #                           backward_path,
    #                           pose_data_path,
    #                           ts_map_path,
    #                           "clip_1689304521000",
    #                           "/mnt/data/legacy_data_playback/data_quality_verigy/B611N0/clip_1689304521000")
    # refiner.refine()
    forward_path = "/mnt/data/autolabel/forward_backward_example/clip_1691641020400_forward/fusion"
    backward_path = "/mnt/data/autolabel/forward_backward_example/clip_1691641020400_backward/fusion"
    pose_data_path = "/mnt/data/autolabel/forward_backward_example/clip_1691641020400_pose"
    refiner = VelocityRefiner(forward_path,
                              backward_path,
                              pose_data_path,
                              clip_name="clip_1691641020400",
                              root_out_path="/mnt/data/autolabel/forward_backward_example/refined_ret")
    refiner.refine()
