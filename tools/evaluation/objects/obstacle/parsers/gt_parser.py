import json
import os

import pandas as pd
from tqdm import tqdm

from utils.transform import quaternion_to_euler
from utils.bbox_ops import get_3d_corners, get_quaternion
from utils.transform import rotation_matrix

from .attribute_tool import Attr, ObstacleEnum, offset_z, ts_round, init_polygon
from ....log_mgr import logger


def is_obj_valid(obj_info):
    w, l, h = obj_info["size"]
    return w > 0 and l > 0 and obj_info["num_lidar_pts"] >= 5
    # return w > 0 and l > 0


def parse_json(data_path):
    raw_data = json.load(open(data_path, 'r'))
    extract_data = {Attr.ts: [],
                    Attr.object_id: [],
                    Attr.frame_seq: [],
                    Attr.lidar_pos_x: [],
                    Attr.lidar_pos_y: [],
                    Attr.lidar_pos_z: [],
                    Attr.length: [],
                    Attr.width: [],
                    Attr.height: [],
                    Attr.lidar_yaw: [],
                    Attr.corners_3d: [],
                    Attr.corners_2d: [],
                    Attr.category: [],
                    Attr.num_lidar_pts: [],
                    Attr.uuid: []}
    for frame_seq, frame_data in enumerate(sorted(raw_data, key=lambda x: x["collected_at"])):
        ts = frame_data["collected_at"]
        if len(ts) == 13:
            ts = float(ts) / 1e3
        objs_info = frame_data['annotated_info']['3d_object_detection_info']['3d_object_detection_anns_info']
        for obj_info in objs_info:
            if not is_obj_valid(obj_info):
                continue
            yaw, _, _ = quaternion_to_euler(*obj_info["obj_rotation"])
            rel_x, rel_y, rel_z = obj_info["obj_center_pos"]
            width, length, height = obj_info["size"]
            rel_z = offset_z(rel_z, height)
            extract_data[Attr.ts].append(ts_round(ts))
            extract_data[Attr.object_id].append(obj_info["object_id"])
            extract_data[Attr.frame_seq].append(frame_seq)
            extract_data[Attr.lidar_pos_x].append(rel_x)
            extract_data[Attr.lidar_pos_y].append(rel_y)
            extract_data[Attr.lidar_pos_z].append(rel_z)
            extract_data[Attr.length].append(length)
            extract_data[Attr.width].append(width)
            extract_data[Attr.height].append(height)
            extract_data[Attr.lidar_yaw].append(yaw)
            extract_data[Attr.num_lidar_pts].append(obj_info["num_lidar_pts"])
            bbox_3d = get_3d_corners(rel_x, rel_y, rel_z, length, width, height, rotation_matrix(yaw))
            bbox_2d = bbox_3d[:4, :2]
            extract_data[Attr.corners_3d].append(bbox_3d)
            extract_data[Attr.corners_2d].append(bbox_2d)
            category = ObstacleEnum.super_class_map.get(obj_info["category"])
            extract_data[Attr.category].append(category)
            extract_data[Attr.uuid].append(obj_info["uuid"])
    pd_data = pd.DataFrame(extract_data)
    pd_data = init_polygon(pd_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    pd_data[Attr.adapt_frame_seq] = pd_data.groupby([pd_data.index]).ngroup()
    return pd_data, pd_data.index.unique().tolist()


def parse_zdrive_anno(data_path):
    extract_data = {Attr.ts: [],
                    Attr.object_id: [],
                    Attr.frame_seq: [],
                    Attr.lidar_pos_x: [],
                    Attr.lidar_pos_y: [],
                    Attr.lidar_pos_z: [],
                    Attr.length: [],
                    Attr.width: [],
                    Attr.height: [],
                    Attr.lidar_yaw: [],
                    Attr.lidar_rel_vel_x: [],
                    Attr.lidar_rel_vel_y: [],
                    Attr.corners_3d: [],
                    Attr.corners_2d: [],
                    Attr.category: [],
                    Attr.num_points: [],
                    Attr.num_lidar_pts: [],
                    Attr.clip_id: [],
                    Attr.frame_id: []}
    with open(data_path, 'r') as f:
        anno_data = json.load(f)
    clip_id = anno_data["bag_id"]
    for frame_seq, frame_data in enumerate(anno_data["frames"]):
        frame_id = frame_data["frame_id"]
        ts = frame_data["lidar_collect"]
        ts = ts_round(float(ts) / 1e6)

        objs = frame_data["annotated_info"]["3d_box_annotated_info"]["annotated_info"]["3d_object_detection_info"][
            "3d_object_detection_anns_info"] if "3d_box_annotated_info" in frame_data["annotated_info"] \
            else frame_data["annotated_info"]["m2_3d_city_object_detection_annotated_info"]["annotated_info"][
            "3d_object_detection_info"]["3d_object_detection_anns_info"]
        for obj in objs:
            length, width, height = obj["size"]
            lidar_x, lidar_y, lidar_z = obj["obj_center_pos"]
            lidar_qx, lidar_qy, lidar_qz, lidar_qw = obj["obj_rotation"]
            if obj.get("rel_velocity") is None:
                lidar_rel_vel_x, lidar_rel_vel_y, lidar_rel_vel_z = 0, 0, 0
            else:
                lidar_rel_vel_x, lidar_rel_vel_y, lidar_rel_vel_z = obj["rel_velocity"]
            yaw, _, _ = quaternion_to_euler(lidar_qw, lidar_qx, lidar_qy, lidar_qz)
            bbox_3d = get_3d_corners(lidar_x, lidar_y, lidar_z, length, width, height, rotation_matrix(yaw))
            bbox_2d = bbox_3d[:4, :2]
            category = ObstacleEnum.super_class_map.get(obj["category"])
            if category is None:
                logger.warning("unsupport category: {}".format(obj["category"]))
                logger.warning("file_path: {}, frame_id: {}".format(data_path, frame_id))
                continue
            extract_data[Attr.ts].append(ts)
            track_id = obj["object_id"] if "object_id" in obj else obj["track_id"]
            extract_data[Attr.object_id].append(track_id)
            extract_data[Attr.frame_seq].append(frame_seq)
            extract_data[Attr.lidar_pos_x].append(lidar_x)
            extract_data[Attr.lidar_pos_y].append(lidar_y)
            extract_data[Attr.lidar_pos_z].append(lidar_z)
            extract_data[Attr.length].append(length)
            extract_data[Attr.width].append(width)
            extract_data[Attr.height].append(height)
            extract_data[Attr.lidar_yaw].append(yaw)
            extract_data[Attr.lidar_rel_vel_x].append(lidar_rel_vel_x)
            extract_data[Attr.lidar_rel_vel_y].append(lidar_rel_vel_y)
            extract_data[Attr.corners_3d].append(bbox_3d)
            extract_data[Attr.corners_2d].append(bbox_2d)
            extract_data[Attr.category].append(category)
            extract_data[Attr.num_points].append(obj["num_lidar_pts"])
            extract_data[Attr.clip_id].append(clip_id)
            extract_data[Attr.frame_id].append(frame_id)
            extract_data[Attr.num_lidar_pts].append(obj["num_lidar_pts"])
    pd_data = pd.DataFrame(extract_data)
    pd_data = init_polygon(pd_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    pd_data[Attr.adapt_frame_seq] = pd_data.groupby([pd_data.index]).ngroup()
    return pd_data, pd_data.index.unique().tolist()


def parse_visual_json(data_path):
    extract_data = {Attr.ts: [],
                    Attr.object_id: [],
                    Attr.frame_seq: [],
                    Attr.lidar_pos_x: [],
                    Attr.lidar_pos_y: [],
                    Attr.lidar_pos_z: [],
                    Attr.length: [],
                    Attr.width: [],
                    Attr.height: [],
                    Attr.lidar_yaw: [],
                    Attr.corners_3d: [],
                    Attr.corners_2d: [],
                    Attr.num_lidar_pts: [],
                    Attr.category: []}
    logger.info("start to load gt files, data_path: {}".format(data_path))
    for frame_seq, file_name in tqdm(enumerate(sorted(os.listdir(data_path)))):
        with open(os.path.join(data_path, file_name), 'r') as f:
            frame_data = json.load(f)
        ts = os.path.splitext(file_name)[0]
        try:
            ts = float(ts)
        except ValueError:
            ts = ts
        for obj in frame_data:
            center = obj["psr"]["position"]
            lidar_x, lidar_y, lidar_z = center["x"], center["y"], center["z"]
            size = obj["psr"]["scale"]
            length, width, height = size["x"], size["y"], size["z"]
            yaw = obj["psr"]["rotation"]["z"]

            bbox_3d = get_3d_corners(lidar_x, lidar_y, lidar_z, length, width, height, rotation_matrix(yaw))
            bbox_2d = bbox_3d[:4, :2]

            category = obj["obj_sub_type"] if "obj_sub_type" in obj else obj["obj_type"]
            if isinstance(category, int):
                category = ObstacleEnum.fusion_subtype_enum[category]
            category = ObstacleEnum.super_class_map[category]
            extract_data[Attr.ts].append(ts)
            extract_data[Attr.object_id].append(obj["obj_id"])
            extract_data[Attr.frame_seq].append(frame_seq)
            extract_data[Attr.lidar_pos_x].append(lidar_x)
            extract_data[Attr.lidar_pos_y].append(lidar_y)
            extract_data[Attr.lidar_pos_z].append(lidar_z)
            extract_data[Attr.length].append(length)
            extract_data[Attr.width].append(width)
            extract_data[Attr.height].append(height)
            extract_data[Attr.lidar_yaw].append(yaw)
            extract_data[Attr.corners_3d].append(bbox_3d)
            extract_data[Attr.corners_2d].append(bbox_2d)
            num_lidar_pts = obj["num_lidar_pts"] if "num_lidar_pts" in obj else -1
            extract_data[Attr.num_lidar_pts].append(num_lidar_pts)
            extract_data[Attr.category].append(category)

    pd_data = pd.DataFrame(extract_data)
    pd_data = init_polygon(pd_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    pd_data[Attr.adapt_frame_seq] = pd_data.groupby([pd_data.index]).ngroup()
    return pd_data, pd_data.index.unique().tolist()


def parse_obejct_list(data_list):
    extract_data = {Attr.ts: [],
                    Attr.object_id: [],
                    Attr.frame_seq: [],
                    Attr.lidar_pos_x: [],
                    Attr.lidar_pos_y: [],
                    Attr.lidar_pos_z: [],
                    Attr.length: [],
                    Attr.width: [],
                    Attr.height: [],
                    Attr.lidar_yaw: [],
                    Attr.corners_3d: [],
                    Attr.corners_2d: [],
                    Attr.num_lidar_pts: [],
                    Attr.category: []}
    logger.info("start to load gt list of {} frames".format(len(data_list)))
    for frame_seq, ts in tqdm(enumerate(data_list)):
        frame_data = data_list[ts]['gt'] # is a dict
        try:
            ts = float(ts)
        except ValueError:
            ts = ts
        for obj in frame_data:
            center = obj["psr"]["position"]
            lidar_x, lidar_y, lidar_z = center["x"], center["y"], center["z"]
            size = obj["psr"]["scale"]
            length, width, height = size["x"], size["y"], size["z"]
            yaw = obj["psr"]["rotation"]["z"]

            bbox_3d = get_3d_corners(lidar_x, lidar_y, lidar_z, length, width, height, rotation_matrix(yaw))
            bbox_2d = bbox_3d[:4, :2]

            category = obj["obj_sub_type"] if "obj_sub_type" in obj else obj["obj_type"]
            if isinstance(category, int):
                category = ObstacleEnum.fusion_subtype_enum[category]
            category = ObstacleEnum.super_class_map[category]
            extract_data[Attr.ts].append(ts)
            extract_data[Attr.object_id].append(obj["obj_id"])
            extract_data[Attr.frame_seq].append(frame_seq)
            extract_data[Attr.lidar_pos_x].append(lidar_x)
            extract_data[Attr.lidar_pos_y].append(lidar_y)
            extract_data[Attr.lidar_pos_z].append(lidar_z)
            extract_data[Attr.length].append(length)
            extract_data[Attr.width].append(width)
            extract_data[Attr.height].append(height)
            extract_data[Attr.lidar_yaw].append(yaw)
            extract_data[Attr.corners_3d].append(bbox_3d)
            extract_data[Attr.corners_2d].append(bbox_2d)
            num_lidar_pts = obj["num_lidar_pts"] if "num_lidar_pts" in obj else -1
            extract_data[Attr.num_lidar_pts].append(num_lidar_pts)
            extract_data[Attr.category].append(category)

    pd_data = pd.DataFrame(extract_data)
    pd_data = init_polygon(pd_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    pd_data[Attr.adapt_frame_seq] = pd_data.groupby([pd_data.index]).ngroup()
    return pd_data, pd_data.index.unique().tolist()
