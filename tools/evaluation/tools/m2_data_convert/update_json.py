import json
import os
import yaml
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from rich.progress import track

default_transform = {"transform":
                         {"rotation": {"x": -0.003570839465284601,
                                       "y": -0.008788895882283447,
                                       "z": 0.001723325361709939,
                                       "w": 0.9999535162018189},
                          "translation": {"x": 0.6495324358955036,
                                          "y": -0.004994488023347435,
                                          "z": -0.38720038058813294}}}


def get_json_list(data_path):
    json_name = os.listdir(data_path)
    json_paths = []
    for item in json_name:
        json_path = os.path.join(data_path, item)
        json_paths.append(json_path)
    return json_paths


def load_ex(params_path=None):
    if params_path is not None:
        with open(params_path) as f:
            params = yaml.safe_load(f)
    else:
        params = default_transform
    qx = params["transform"]["rotation"]["x"]
    qy = params["transform"]["rotation"]["y"]
    qz = params["transform"]["rotation"]["z"]
    qw = params["transform"]["rotation"]["w"]
    tx = params["transform"]["translation"]["x"]
    ty = params["transform"]["translation"]["y"]
    tz = params["transform"]["translation"]["z"]
    q = np.array([qx, qy, qz, qw])
    r_ = R.from_quat(q)
    rotation = r_.as_matrix()
    m2_2_pandar128 = np.eye(4)
    m2_2_pandar128[:3, :3] = rotation
    m2_2_pandar128[:3, 3] = np.array([tx, ty, tz])
    pandar128_2_m2 = np.linalg.inv(m2_2_pandar128)
    return pandar128_2_m2


def load_json(json_file):
    with open(json_file) as f:
        label_info = json.load(f)
    return label_info


def get_obj_info(obj):
    obj_x = obj["psr"]["position"]["x"]
    obj_y = obj["psr"]["position"]["y"]
    obj_z = obj["psr"]["position"]["z"]
    obj_rotation = obj["psr"]["rotation"]["z"]
    obj_center = np.array([obj_x, obj_y, obj_z, 1])
    return obj_center, obj_rotation


def cal_heding(heading_vector_inm2):
    x = heading_vector_inm2[0]
    y = heading_vector_inm2[1]
    heading = np.arctan((y / x))
    if x < 0:
        if heading > 0:
            heading -= np.pi
        else:
            heading += np.pi
    return heading


def handle_obj(label_info, T):
    for i, obj in enumerate(label_info):
        print(obj)
        obj_center, heding_inlidar = get_obj_info(obj)
        obj_center_m2 = T @ obj_center
        heading_vetor_inlidar = np.array([math.cos(heding_inlidar), math.sin(heding_inlidar), 0])
        heading_vector_inm2 = T[:3, :3] @ heading_vetor_inlidar
        #        heding_inm2 = cal_heding(heading_vector_inm2)
        obj["psr"]["position"]["x"] = obj_center_m2[0]
        obj["psr"]["position"]["y"] = obj_center_m2[1]
        obj["psr"]["position"]["z"] = obj_center_m2[2]
        obj["psr"]["rotation"]["x"] = 0
        obj["psr"]["rotation"]["y"] = 0
        obj["psr"]["rotation"]["z"] = math.atan2(heading_vector_inm2[1], heading_vector_inm2[0])
        label_info[i] = obj
    return label_info


def hesai_to_m2(data_path, save_path):
    json_paths = get_json_list(data_path)
    T = load_ex()
    for json_path in track(json_paths):
        json_name = os.path.basename(json_path)
        # print(json_name)
        frame_label_info = load_json(json_path)
        if len(frame_label_info) != 0:
            frame_label_update = handle_obj(frame_label_info, T)
            save_update_path = os.path.join(save_path, json_name)
            with open(save_update_path, "w") as f:
                json.dump(frame_label_update, f)


if __name__ == "__main__":
    data_path = "/mnt/data/lidar_detection/test_datasets/m2test_0118/annotation_all"
    save_path = "/mnt/data/lidar_detection/test_datasets/m2test_0118/annotation_all_m2"
    # params_path = "config/roborsm2_novatel_extrinsics.yaml"
    params_path = "/home/wuchuanpan/PycharmProjects/experiment/tools/m2_data_convert/roborsm2_novatel_extrinsics.yaml"
    hesai_to_m2(data_path, save_path)
