import numpy as np
import pandas as pd
from cyber_record.record import Record

from objects.obstacle.parsers.attribute_tool import Attr, ts_round
from utils.bbox_ops import get_3d_corners, get_quaternion


def parser_for_raw(data_path):
    """
    For evaluation of raw radar (output of radar driver)
    Args:
        data_path: Absolute path of txt file which contains the raw radar outputs.
    Returns:
        DataFrame saves each single obj as one row.
    """
    extract_data = {Attr.ts: [],
                    Attr.score: [],
                    Attr.object_id: [],
                    Attr.utm_pos_x: [],
                    Attr.utm_pos_y: [],
                    Attr.utm_abs_vel_x: [],
                    Attr.utm_abs_vel_y: [],
                    Attr.lidar_pos_x: [],
                    Attr.lidar_pos_y: [],
                    Attr.lidar_abs_vel_x: [],
                    Attr.lidar_abs_vel_y: [],
                    Attr.corners_3d: []}
    with open(data_path, 'r') as obj_f:
        for frame_seq, obj_info in enumerate(obj_f.readlines()):
            obj_info = obj_info.strip()
            if len(obj_info) > 0:
                obj_list = np.fromstring(obj_info, sep=" ", dtype=np.float64).reshape(-1, 14)
                for obj in obj_list:
                    ts, obj_ts, obj_id, score, rel_x, rel_y, rel_vel_x, rel_vel_y, raw_yaw, utm_x, \
                        utm_y, abs_vel_x, abs_vel_y, yaw = obj
                    extract_data[Attr.ts].append(ts_round(ts))
                    extract_data[Attr.score].append(score)
                    extract_data[Attr.object_id].append(obj_id)
                    extract_data[Attr.utm_pos_x].append(utm_x)
                    extract_data[Attr.utm_pos_y].append(utm_y)
                    extract_data[Attr.utm_abs_vel_x].append(abs_vel_x)
                    extract_data[Attr.utm_abs_vel_y].append(abs_vel_y)
                    extract_data[Attr.lidar_pos_x].append(rel_x)
                    extract_data[Attr.lidar_pos_y].append(rel_y)
                    extract_data[Attr.lidar_abs_vel_x].append(rel_vel_x)
                    extract_data[Attr.lidar_abs_vel_y].append(rel_vel_y)
                    quaternion = get_quaternion(0)
                    corners_3d = get_3d_corners(rel_x, rel_y, 0, 1, 1, 1,
                                                quaternion.rotation_matrix)
                    extract_data[Attr.corners_3d].append(corners_3d)
    pd_data = pd.DataFrame(extract_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    return pd_data, pd_data.index.unique().tolist()


def parser_for_processed(data_path):
    """
        For evaluation of processed radar outputs
        Args:
            data_path: Absolute path of txt file which contains the radar perception outputs.
        Returns:
            DataFrame saves each single obj as one row.
        """
    extract_data = {Attr.ts: [],
                    Attr.score: [],
                    Attr.object_id: [],
                    Attr.utm_pos_x: [],
                    Attr.utm_pos_y: [],
                    Attr.utm_abs_vel_x: [],
                    Attr.utm_abs_vel_y: []}
    with open(data_path, 'r') as obj_f:
        for frame_seq, obj_info in enumerate(obj_f.readlines()):
            obj_info = obj_info.strip()
            if len(obj_info) > 0:
                obj_list = np.fromstring(obj_info, sep=" ", dtype=np.float64).reshape(-1, 8)
                for obj in obj_list:
                    ts, obj_id, score, yaw, utm_x, utm_y, utm_abs_vel_x, utm_abs_vel_y = obj
                    extract_data[Attr.ts].append(ts_round(ts)-0.1)
                    extract_data[Attr.score].append(score)
                    extract_data[Attr.object_id].append(obj_id)
                    extract_data[Attr.utm_pos_x].append(utm_x)
                    extract_data[Attr.utm_pos_y].append(utm_y)
                    extract_data[Attr.utm_abs_vel_x].append(utm_abs_vel_x)
                    extract_data[Attr.utm_abs_vel_y].append(utm_abs_vel_y)
    pd_data = pd.DataFrame(extract_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    return pd_data, pd_data.index.unique().tolist()


def parse_record(data_path, topic="/apollo/sensor/radar/main"):
    extract_data = {Attr.ts: [],
                    Attr.score: [],
                    Attr.object_id: [],
                    Attr.lidar_pos_x: [],
                    Attr.lidar_pos_y: [],
                    Attr.length: [],
                    Attr.width: [],
                    Attr.corners_3d: []}
    with Record(data_path, 'r') as record:
        for frame_seq, (topic, message, t) in enumerate(record.read_messages(topic)):
            for obj in message.contiobs:
                ts = message.header.timestamp_sec
                extract_data[Attr.ts].append(ts_round(ts))
                extract_data[Attr.score].append(obj.probexist)
                extract_data[Attr.object_id].append(obj.obstacle_id)
                extract_data[Attr.lidar_pos_x].append(obj.longitude_dist)
                extract_data[Attr.lidar_pos_y].append(obj.lateral_dist)
                extract_data[Attr.length].append(obj.length)
                extract_data[Attr.width].append(obj.width)
                quaternion = get_quaternion(0)
                corners_3d = get_3d_corners(obj.longitude_dist, obj.lateral_dist, 0, 1, 1, 1, quaternion.rotation_matrix)
                extract_data[Attr.corners_3d].append(corners_3d)
    pd_data = pd.DataFrame(extract_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    return pd_data, pd_data.index.unique().tolist()
