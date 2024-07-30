import os
import json
from collections import defaultdict

from cyber_record.record import Record
import numpy as np

from .. import SensorName
from log_mgr import logger


class CalibMeta:
    def __init__(self, data_path):
        self.data_path = data_path
        self.calib_params = self.parse()

    def get_camera_calib_info(self):
        calib_info = defaultdict(dict)
        calib_path = os.path.join(self.data_path, "calib", "camera")
        for calib_file in os.listdir(calib_path):
            sensor_name = os.path.splitext(calib_file)[0]
            with open(os.path.join(calib_path, calib_file), 'r') as f:
                calib_data = json.load(f)
            extrinsic = np.array(calib_data["extrinsic"], dtype=np.float32).reshape((4, 4))
            intrinsic = np.array(calib_data["intrinsic"], dtype=np.float32).reshape((3, 3))
            calib_info[sensor_name]["extrinsic"] = extrinsic
            calib_info[sensor_name]["intrinsic"] = intrinsic
        return calib_info

    def get_cam_extrinsic(self, cam_name):
        return self.calib_params["camera"][cam_name]["extrinsic"]

    def get_cam_intrinsic(self, cam_name):
        return self.calib_params["camera"][cam_name]["intrinsic"]

    def get_pad_cam_intrinsic(self, cam_name):
        intrinsic = self.get_cam_intrinsic(cam_name)
        return np.hstack([intrinsic, np.zeros((3, 1), dtype=np.float32)])

    def parse_from_clip(self):
        camera_calib_info = self.get_camera_calib_info()
        return {"camera": camera_calib_info}

    def parse_from_record(self):
        pass

    def parse(self):
        return self.parse_from_clip()


if __name__ == "__main__":
    from pyquaternion import Quaternion
    import yaml
    import numpy as np
    lidar2cam1_file = "/home/wuchuanpan/Data/clips/2023_0712/clip_1689126848599/extrinsics/lidar2camera/lidar2frontmain.yaml"
    lidar2imu_file = "/home/wuchuanpan/Data/clips/2023_0712/clip_1689126848599/extrinsics/lidar2imu/lidar2imu.yaml"

    lidar2cam1_data = yaml.safe_load(open(lidar2cam1_file, 'r'))
    lidar2cam1_rot = np.array(lidar2cam1_data["transform"], dtype=np.float64).reshape((4, 4))

    lidar2imu_data = yaml.safe_load((open(lidar2imu_file, 'r')))
    l2i_rot_raw = lidar2imu_data["transform"]["rotation"]
    l2i_translation = lidar2imu_data["transform"]["translation"]
    l2i_rot = Quaternion([l2i_rot_raw["w"],
                          l2i_rot_raw["x"],
                          l2i_rot_raw["y"],
                          l2i_rot_raw["z"]])


    cam2world_transform = np.array([
        0.0145075,-0.999839,0.0106345,-0.124495,
        -0.00354034,-0.010687,-0.999937,-0.0651284,
        0.999888,0.0144688,-0.00369482,-0.207757,
        0,0,0,1]).reshape((4, 4))
    cam2world_rot = cam2world_transform[:3, :3]
    # print(l2i_rot.rotation_matrix)
    # print(cam2world_rot)

    # print(np.matmul(np.linalg.inv(cam2world_rot), l2i_rot.rotation_matrix))
    logger.info(np.linalg.inv(cam2world_rot) @ (l2i_rot.rotation_matrix))
    logger.info(lidar2cam1_rot)
    logger.info(np.linalg.inv(lidar2cam1_rot))

    tf_static_cam_lidar = Quaternion([-0.0057542, 0.0093682, -0.7044519, 0.7096666])
    logger.info(tf_static_cam_lidar.rotation_matrix)