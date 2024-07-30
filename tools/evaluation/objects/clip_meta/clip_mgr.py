import os
import json
from collections import defaultdict

import numpy as np
import cv2

from objects.calib.calibration_mgr import CalibMeta


def get_img_size(img_path):
    img_mat = cv2.imread(img_path)
    return img_mat.shape[:2]


class ClipMeta:
    def __init__(self, clip_path):
        self.clip_path = clip_path
        self.clip_name = self.get_clip_name(clip_path)
        self.clip_info = self.get_clip_info()
        self.sensor_mapping = self.get_sensor_mapping()
        self.frame_info = self.get_frame_info()
        self.img_info = self.get_img_info()
        self.calib_mgr = CalibMeta(clip_path)

    @staticmethod
    def get_clip_name(clip_path):
        return os.path.basename(clip_path)

    def get_clip_info(self):
        clip_info_file = os.path.join(self.clip_path, "info.json")
        with open(clip_info_file, 'r') as f:
            clip_info = json.load(f)
        return clip_info

    def get_sensor_mapping(self):
        mapping = dict()
        for id_name, common_name in self.clip_info["mapping"].items():
            mapping[common_name.strip().replace(" ", "_")] = id_name
        return mapping

    @staticmethod
    def search_files(root_path, post_fix, record_data):
        # key_name = "{sensor_name}_" + post_fix.strip(".")
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith(post_fix):
                    frame_id = os.path.splitext(file)[0]
                    sensor_name = os.path.basename(root)
                    # record_data[frame_id][key_name.format(sensor_name=sensor_name)] = os.path.join(root, file)
                    record_data[frame_id][sensor_name] = os.path.join(root, file)
        return record_data

    def get_frame_info(self):
        lidar_path = os.path.join(self.clip_path, "lidar")
        camera_path = os.path.join(self.clip_path, "camera")
        frame_info = defaultdict(dict)
        frame_info = self.search_files(lidar_path, ".pcd", frame_info)
        frame_info = self.search_files(camera_path, ".jpg", frame_info)
        return frame_info

    def get_img_info(self):
        img_info = dict()
        for frame_id, frame_data in self.frame_info.items():
            for sensor_name, file_path in frame_data.items():
                if "camera" in sensor_name:
                    img_h, img_w = get_img_size(file_path)
                    img_info[sensor_name] = {"img_w": img_w,
                                             "img_h": img_h}
            break
        return img_info

    def get_cam_extrinsic(self, cam_name):
        return self.calib_mgr.get_cam_extrinsic(cam_name)

    def get_cam_intrinsic(self, cam_name):
        return self.calib_mgr.get_cam_intrinsic(cam_name)

    def get_pad_cam_intrinsic(self, cam_name):
        intrinsic = self.get_cam_intrinsic(cam_name)
        return np.hstack([intrinsic, np.zeros((3, 1), dtype=np.float32)])

    def get_pcd_path(self, frame_id, lidar_name="lidar"):
        return self.frame_info[frame_id][lidar_name]

    def get_img_path(self, frame_id, cam_name):
        return self.frame_info[frame_id][cam_name]

    def get_img_w(self, cam_name="camera1"):
        return self.img_info[cam_name]["img_w"]

    def get_img_h(self, cam_name="camera1"):
        return self.img_info[cam_name]["img_h"]

    def get_cam_info(self, cam_name):
        return [self.get_cam_extrinsic(cam_name),
                self.get_pad_cam_intrinsic(cam_name),
                self.get_img_w(cam_name),
                self.get_img_h(cam_name)]
