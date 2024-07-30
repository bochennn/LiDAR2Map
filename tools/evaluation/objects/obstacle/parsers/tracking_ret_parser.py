import json
import os

import pandas as pd
from tqdm import tqdm

from ....log_mgr import logger
from ....utils.bbox_ops import get_3d_corners, get_quaternion
from .attribute_tool import Attr, ObstacleEnum, init_polygon, ts_round

pd.options.display.float_format = '{:.0f}'.format


def parser(data_path):
    """
        For evaluation of onboard obstacle fusion task and tracking task in the format of list of json files.
        Args:
            data_path: Absolute path of json files which contains "tracking result" of obstacles.
        Returns:
            DataFrame saves each single obj as one row.
        """
    extract_data = {Attr.ts: [],
                    Attr.frame_seq: [],
                    Attr.measure_ts: [],
                    Attr.score: [],
                    Attr.object_id: [],
                    Attr.utm_pos_x: [],
                    Attr.utm_pos_y: [],
                    Attr.utm_pos_z: [],
                    Attr.lidar_pos_x: [],
                    Attr.lidar_pos_y: [],
                    Attr.lidar_pos_z: [],
                    Attr.length: [],
                    Attr.width: [],
                    Attr.height: [],
                    Attr.lidar_yaw: [],
                    Attr.utm_yaw: [],
                    Attr.quaternion: [],
                    Attr.corners_3d: [],
                    Attr.corners_2d: [],
                    Attr.type_id: [],
                    Attr.subtype_id: [],
                    Attr.category: [],
                    Attr.utm_abs_vel_x: [],
                    Attr.utm_abs_vel_y: [],
                    Attr.utm_abs_vel_z: [],
                    Attr.abs_acc_x: [],
                    Attr.abs_acc_y: [],
                    Attr.abs_acc_z: [],
                    Attr.lidar_abs_vel_x: [],
                    Attr.lidar_abs_vel_y: []}
    logger.info("start to load pred files, data_path: {}".format(data_path))
    for frame_seq, file_name in tqdm(enumerate(sorted(os.listdir(data_path)))):
        with open(os.path.join(data_path, file_name), 'r') as f:
            frame_data = json.load(f)
        ts = os.path.splitext(file_name)[0]
        try:
            ts = float(ts)
        except ValueError:
            ts = ts
        for obj in frame_data:
            if obj.get("is_cluster"):
                print("cluster obj found")
                continue
            center = obj["psr"]["position"]
            lidar_x, lidar_y, lidar_z = center["x"], center["y"], center["z"]
            size = obj["psr"]["scale"]
            length, width, height = size["x"], size["y"], size["z"]
            yaw = obj["psr"]["rotation"]["z"]
            lidar_velocity = obj["lidar_velocity"]
            lidar_vx, lidar_vy, lidar_vz = lidar_velocity["x"], lidar_velocity["y"], lidar_velocity["z"]
            utm_center = obj["utm_position"]
            utm_x, utm_y, utm_z = utm_center["x"], utm_center["y"], utm_center["z"]
            utm_velocity = obj["utm_velocity"]
            utm_vx, utm_vy, utm_vz = utm_velocity["x"], utm_velocity["y"], utm_velocity["z"]
            utm_acc = obj["utm_acceleration"] if "utm_acceleration" in obj else {"x": 0, "y": 0, "z": 0}
            utm_ax, utm_ay, utm_az = utm_acc["x"], utm_acc["y"], utm_acc["z"]
            utm_yaw = obj["utm_yaw"]

            extract_data[Attr.ts].append(ts)
            extract_data[Attr.frame_seq].append(frame_seq)
            extract_data[Attr.measure_ts].append(ts_round(obj["measure_timestamp"]))
            extract_data[Attr.score].append(obj["obj_score"])
            extract_data[Attr.object_id].append(obj["obj_id"])
            extract_data[Attr.utm_pos_x].append(utm_x)
            extract_data[Attr.utm_pos_y].append(utm_y)
            extract_data[Attr.utm_pos_z].append(utm_z)
            extract_data[Attr.lidar_pos_x].append(lidar_x)
            extract_data[Attr.lidar_pos_y].append(lidar_y)
            extract_data[Attr.lidar_pos_z].append(lidar_z)
            extract_data[Attr.length].append(length)
            extract_data[Attr.width].append(width)
            extract_data[Attr.height].append(height)
            extract_data[Attr.lidar_yaw].append(yaw)
            extract_data[Attr.utm_yaw].append(utm_yaw)
            quaternion = get_quaternion(yaw)
            extract_data[Attr.quaternion].append(quaternion)
            corners_3d = get_3d_corners(lidar_x, lidar_y, lidar_z, length, width, height,
                                        quaternion.rotation_matrix)
            corners_2d = corners_3d[:4, :2]
            extract_data[Attr.corners_3d].append(corners_3d)
            extract_data[Attr.corners_2d].append(corners_2d)
            extract_data[Attr.type_id].append(obj["obj_type"])
            extract_data[Attr.subtype_id].append(obj["obj_sub_type"])
            category = ObstacleEnum.fusion_subtype_enum.get(obj["obj_sub_type"])
            category = ObstacleEnum.super_class_map.get(category)
            extract_data[Attr.category].append(category)
            extract_data[Attr.utm_abs_vel_x].append(utm_vx)
            extract_data[Attr.utm_abs_vel_y].append(utm_vy)
            extract_data[Attr.utm_abs_vel_z].append(utm_vz)
            extract_data[Attr.abs_acc_x].append(utm_ax)
            extract_data[Attr.abs_acc_y].append(utm_ay)
            extract_data[Attr.abs_acc_z].append(utm_az)
            extract_data[Attr.lidar_abs_vel_x].append(lidar_vx)
            extract_data[Attr.lidar_abs_vel_y].append(lidar_vy)
    pd_data = pd.DataFrame(extract_data)
    pd_data = init_polygon(pd_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    pd_data[Attr.adapt_frame_seq] = pd_data.groupby([pd_data.index]).ngroup()
    return pd_data, pd_data.index.unique().tolist()
