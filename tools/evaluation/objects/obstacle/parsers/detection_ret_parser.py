import json
import os

import pandas as pd

from objects.obstacle.parsers.attribute_tool import Attr, ts_round, ObstacleEnum, init_polygon, offset_z
from utils.bbox_ops import get_3d_corners, get_quaternion


def parser(data_path):
    """
    For evaluation of both onboard and offboard lidar detection in the format of list of json files.
    Args:
        data_path: Absolute path of folder contains json files.
    Returns:
        DataFrame saves each single obj as one row.
    """
    extract_data = {Attr.ts: [],
                    Attr.frame_seq: [],
                    Attr.score: [],
                    Attr.lidar_pos_x: [],
                    Attr.lidar_pos_y: [],
                    Attr.lidar_pos_z: [],
                    Attr.length: [],
                    Attr.width: [],
                    Attr.height: [],
                    Attr.lidar_yaw: [],
                    Attr.quaternion: [],
                    Attr.corners_3d: [],
                    Attr.corners_2d: [],
                    Attr.type_id: [],
                    Attr.subtype_id: [],
                    Attr.category: [],
                    Attr.is_key_obj: []}
    for frame_seq, file_name in enumerate(sorted(os.listdir(data_path))):
        with open(os.path.join(data_path, file_name), 'r') as f:
            frame_data = json.load(f)
        ts = os.path.splitext(file_name)[0]
        try:
            ts = float(ts)
        except ValueError:
            ts = ts
        for obj in frame_data:
            if "obj_score" not in obj:
                continue
            center = obj["psr"]["position"]
            lidar_x, lidar_y, lidar_z = center["x"], center["y"], center["z"]
            size = obj["psr"]["scale"]
            length, width, height = size["x"], size["y"], size["z"]
            yaw = obj["psr"]["rotation"]["z"]

            extract_data[Attr.ts].append(ts)
            extract_data[Attr.frame_seq].append(frame_seq)
            extract_data[Attr.score].append(obj["obj_score"])
            extract_data[Attr.lidar_pos_x].append(lidar_x)
            extract_data[Attr.lidar_pos_y].append(lidar_y)
            extract_data[Attr.lidar_pos_z].append(lidar_z)
            extract_data[Attr.length].append(length)
            extract_data[Attr.width].append(width)
            extract_data[Attr.height].append(height)
            extract_data[Attr.lidar_yaw].append(yaw)
            quaternion = get_quaternion(yaw)
            extract_data[Attr.quaternion].append(quaternion)
            corners_3d = get_3d_corners(lidar_x, lidar_y, lidar_z, length, width, height, quaternion.rotation_matrix)
            corners_2d = corners_3d[:4, :2]
            extract_data[Attr.corners_3d].append(corners_3d)
            extract_data[Attr.corners_2d].append(corners_2d)
            extract_data[Attr.type_id].append(obj["obj_type"])
            extract_data[Attr.subtype_id].append(obj["obj_sub_type"] if "obj_sub_type" in obj else 0)
            if "category" in obj:
                category = ObstacleEnum.super_class_map[obj["category"]]
            else:
                category = ObstacleEnum.fusion_subtype_enum.get(obj["obj_sub_type"])
                category = ObstacleEnum.super_class_map.get(category)
            extract_data[Attr.category].append(category)
            extract_data[Attr.is_key_obj].append(False)
    pd_data = pd.DataFrame(extract_data)
    pd_data = init_polygon(pd_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    return pd_data, pd_data.index.unique().tolist()
