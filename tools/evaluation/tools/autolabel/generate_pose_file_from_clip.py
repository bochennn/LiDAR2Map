import os
import json
import yaml
import math

import numpy as np



BUFF_SIZE_IN_SEC = 0.01


def rotation_matrix(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float64)


def quaternion_to_euler(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return yaw, pitch, roll


def rot_translate_combine(rotation, translation):
    assert rotation.shape == (3, 3)
    assert translation.shape == (3, 1)
    return np.vstack([np.hstack([rotation, translation]), [0, 0, 0, 1]])


def transform_matrix(rotation, translation):
    """
    Args:
        rotation: quaternion in order of (w, x, y, z)
        translation: translation in order of (x, y z)
    Returns:
        (4, 4) transformation matrix
    """
    yaw, _, _ = quaternion_to_euler(*rotation)
    rotation = rotation_matrix(yaw)
    translation = np.array([*translation], dtype=np.float64).reshape((3, 1))
    return rot_translate_combine(rotation, translation)


def get_lidar2imu_trans(clip_path):
    file_path = os.path.join(clip_path, "extrinsics", "lidar2imu", "lidar2imu.yaml")
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    rotation = data["transform"]["rotation"]
    translation = data["transform"]["translation"]
    lidar2imu = transform_matrix([rotation["w"], rotation["x"], rotation["y"], rotation["z"]],
                                 [translation["x"], translation["y"], translation["z"]])
    return lidar2imu


def get_imu2utm_trans(clip_path, target_ts):
    localization_path = os.path.join(clip_path, "localization.json")
    with open(localization_path, 'r') as f:
        pose_data = json.load(f)
    target_pose = min(pose_data, key=lambda x: abs(x["timestamp"] - target_ts))
    if abs(target_pose["timestamp"] - target_ts) > BUFF_SIZE_IN_SEC:
        print("pose not found, target_ts: {}, nearest_pose_ts: {}".format(target_ts, target_pose["timestamp"]))
        return None
    rotation = target_pose["pose"]["orientation"]
    translation = target_pose["pose"]["position"]
    imu2utm = transform_matrix([rotation["qw"], rotation["qx"], rotation["qy"], rotation["qz"]],
                               [translation["x"], translation["y"], translation["z"]])
    return imu2utm


def get_lidar2utm_trans(clip_path, target_ts):
    imu2utm = get_imu2utm_trans(clip_path, target_ts)
    lidar2imu = get_lidar2imu_trans(clip_path)
    return imu2utm.dot(lidar2imu)


def generate_pose_file_from_clip(clip_path, pose_out_path, offset_sec=0):
    lidar_ts_list_in_ns = [dir_name.split("_")[-1] for dir_name in os.listdir(clip_path)
                           if os.path.isdir(os.path.join(clip_path, dir_name)) and dir_name.startswith("sample_")]
    lidar2imu = get_lidar2imu_trans(clip_path)
    os.makedirs(pose_out_path, exist_ok=True)
    for ts_in_ns in lidar_ts_list_in_ns:
        measurement_ts = ts_in_ns[:10] + "." + ts_in_ns[10:]
        lidar_ts = ts_in_ns + "000"
        target_ts = float(measurement_ts) - offset_sec
        imu2world = get_imu2utm_trans(clip_path, target_ts)
        lidar2world = get_lidar2utm_trans(clip_path, target_ts)
        pose_info = {
            "lidar2novatel": lidar2imu.tolist(),
            "lidar2world": lidar2world.tolist(),
            "novatel2world": imu2world.tolist(),
            "lidar_query_tf_offset": offset_sec,
            "lidar_timestamp": int(lidar_ts),
            "measurement_time": float(measurement_ts)
        }
        pose_out_file_path = os.path.join(pose_out_path, "{}.json".format(measurement_ts))
        with open(pose_out_file_path, 'w') as f:
            json.dump(pose_info, f)


if __name__ == "__main__":
    generate_pose_file_from_clip(
        "/mnt/data/lidar_detection/test_datasets/20231028/clips/clip_1698469608001",
        "/mnt/data/lidar_detection/test_datasets/20231028/clips/clip_1698469608001/pose_info"
    )