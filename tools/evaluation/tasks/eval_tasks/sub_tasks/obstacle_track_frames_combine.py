import os

import numpy as np
import simplejson as json

from ....utils.transform import scalar_transform, vector_transform
from ....utils.pointcloud_ops import load_bin, points_crop


class TrackFrameCombine:
    def __init__(self, instant_list, pose_data_path, raw_lidar_path=None, crop_lidar_points=False):
        self.instant_list = instant_list
        self.chosen_index = 0
        self.raw_lidar_path = raw_lidar_path
        if crop_lidar_points and raw_lidar_path is None:
            raise AttributeError("raw_lidar_path must be provided when crop_lidar_points set True")
        self.pose_map = self.get_pose_data(pose_data_path)

    @staticmethod
    def get_pose_data(pose_data_path):
        pose_map = dict()
        for pose_file in sorted(os.listdir(pose_data_path)):
            if pose_file.endswith(".json"):
                ts = os.path.splitext(pose_file)[0]
                with open(os.path.join(pose_data_path, pose_file), 'r') as f:
                    frame_pose_data = json.load(f)
                pose_map[ts] = frame_pose_data["lidar2world"]
        return pose_map

    @staticmethod
    def get_obj_display_attr(instant):
        return "L: {:.2f}, W: {:.2f}, yaw: {:.2f}".format(instant.get_length(),
                                                          instant.get_width(),
                                                          instant.get_utm_yaw_degree())

    @staticmethod
    def convert_visual_obj(instant, transform):
        point = instant.get_obj_center()
        new_point = scalar_transform(point, transform)
        yaw = instant.get_lidar_yaw()
        direction = [np.cos(yaw), np.sin(yaw), 0]
        new_direction = vector_transform(direction, transform)
        new_yaw = np.arctan2(new_direction[1], new_direction[0])

        visual_obj = instant.get_visual_json_obj(TrackFrameCombine.get_obj_display_attr(instant))
        visual_obj["psr"]["position"] = {"x": new_point[0],
                                         "y": new_point[1],
                                         "z": new_point[2]}
        visual_obj["psr"]["rotation"]["z"] = new_yaw
        return visual_obj

    def convert_lidar_points(self, instant, transform=None):
        instant_ts = "{:.6f}".format(instant.get_ts())
        raw_lidar_file = os.path.join(self.raw_lidar_path, "{}.bin".format(instant_ts))
        lidar_points = load_bin(raw_lidar_file)
        corners = instant.get_corners_3d()
        target_points = points_crop(lidar_points,
                                    [np.min(corners[:, 0]), np.max(corners[:, 0])],
                                    [np.min(corners[:, 1]), np.max(corners[:, 1])],
                                    [np.min(corners[:, 2]), np.max(corners[:, 2])],
                                    )
        if transform is not None:
            target_points = [scalar_transform(point, transform) for point in target_points]
            target_points = np.array(target_points)
        return target_points

    def convert_frame_to_chosen_frame(self, instant, chosen_ts):
        instant_ts = "{:.6f}".format(instant.get_ts())
        other_lidar2world = self.pose_map[instant_ts]
        world2chosen_lidar = np.linalg.inv(self.pose_map[chosen_ts])
        transform = world2chosen_lidar.dot(other_lidar2world)

        lidar_points = self.convert_lidar_points(instant, transform)
        visual_obj = self.convert_visual_obj(instant, transform)
        return visual_obj, lidar_points

    def adjust_obj_pos(self, out_visual_objs, out_lidar_points, chosen_index):
        prev_obj = out_visual_objs[chosen_index]
        direction = None
        for idx in reversed(range(chosen_index)):
            crt_obj = out_visual_objs[idx]
            crt_position = crt_obj["psr"]["position"]
            crt_length = crt_obj["psr"]["scale"]["x"]
            prev_position = prev_obj["psr"]["position"]
            prev_length = prev_obj["psr"]["scale"]["x"]
            if direction is None:
                direction = "forward" if crt_position["x"] > prev_position["x"] else "backward"
            delta_x = crt_position["x"] - prev_position["x"]
            min_gap = 1.1 * ((crt_length + prev_length) / 2)
            if direction == "forward" and (0 < delta_x < min_gap or delta_x < 0):
                crt_position["x"] += (min_gap - delta_x)
                out_lidar_points[idx][:, 0] += (min_gap - delta_x)
            if direction == "backward" and (-min_gap < delta_x < 0 or delta_x > 0):
                crt_position["x"] -= (min_gap - delta_x)
                out_lidar_points[idx][:, 0] -= (min_gap - delta_x)
            prev_obj = crt_obj

        prev_obj = out_visual_objs[chosen_index]
        direction = "backward" if direction == "forward" else "forward"
        for idx in range(chosen_index + 1, len(out_visual_objs)):
            crt_obj = out_visual_objs[idx]
            crt_position = crt_obj["psr"]["position"]
            crt_length = crt_obj["psr"]["scale"]["x"]
            prev_position = prev_obj["psr"]["position"]
            prev_length = prev_obj["psr"]["scale"]["x"]
            delta_x = crt_position["x"] - prev_position["x"]
            min_gap = 1.1 * ((crt_length + prev_length) / 2)
            if direction == "forward" and (0 < delta_x < min_gap or delta_x < 0):
                crt_position["x"] += (min_gap - delta_x)
                out_lidar_points[idx][:, 0] += (min_gap - delta_x)
            if direction == "backward" and (-min_gap < delta_x < 0 or delta_x > 0):
                crt_position["x"] -= (min_gap - delta_x)
                out_lidar_points[idx][:, 0] -= (min_gap - delta_x)
            prev_obj = crt_obj
        return out_visual_objs, out_lidar_points

    def start(self):
        chosen_index = self.chosen_index
        chosen_instant = self.instant_list[chosen_index]
        out_visual_objs = []
        out_lidar_points = []
        chosen_ts = "{:.6f}".format(chosen_instant.get_ts())
        for idx, instant in enumerate(self.instant_list):
            if idx == chosen_index:
                chosen_lidar_points = self.convert_lidar_points(chosen_instant)
                out_visual_objs.append(chosen_instant.get_visual_json_obj(self.get_obj_display_attr(chosen_instant)))
                out_lidar_points.append(chosen_lidar_points)
            visual_obj, lidar_points = self.convert_frame_to_chosen_frame(instant, chosen_ts)
            out_visual_objs.append(visual_obj)
            out_lidar_points.append(lidar_points)
        out_visual_objs, out_lidar_points = self.adjust_obj_pos(out_visual_objs, out_lidar_points, chosen_index)
        out_points_array = []
        for points in out_lidar_points:
            out_points_array.extend(points)
        out_points_array = np.array(out_points_array)
        return out_visual_objs, out_points_array, chosen_ts
