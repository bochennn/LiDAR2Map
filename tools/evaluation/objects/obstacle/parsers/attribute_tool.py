from shapely import polygons

from log_mgr import logger


def ts_round(ts):
    if isinstance(ts, str):
        ts = float(ts)
    return round(ts, 6)


lidar_height = 1.98
lidar_height2 = 2.03


# def offset_z(z, height):
#     return z - lidar_height + height / 2

def offset_z(z, height):
    return z + lidar_height2 - height / 2

def init_polygon(pd_data):
    pd_data[Attr.polygon] = None
    mask = pd_data[Attr.corners_2d].notna()
    try:
        pd_data.loc[mask, "polygon"] = polygons(pd_data.loc[mask, Attr.corners_2d].to_list())
    except Exception as e:
        logger.error(e)
    return pd_data


class ObstacleEnum:
    lidar_detection_type_enum = {0: 'car',
                                 1: 'pickup_truck',
                                 2: 'truck',
                                 3: 'construction_vehicle',
                                 4: 'bus',
                                 5: 'tricycle',
                                 6: 'motorcycle',
                                 7: 'bicycle',
                                 8: 'person',
                                 9: 'traffic_cone',
                                 10: 'unknown'}

    fusion_type_enum = {0: "Unknown",
                        1: "Unknown_movable",
                        2: "Unknown_unmovable",
                        3: "Pedestrian",
                        4: "Bicycle",
                        5: "Vehicle",
                        6: "Max_object_type"}

    fusion_subtype_enum = {0: "Unknown",
                           1: "Unknown_movable",
                           2: "Unknown_unmovable",
                           3: "Car",
                           4: "Van",
                           5: "Truck",
                           6: "Bus",
                           7: "Cyclist",
                           8: "Motorcyclist",
                           9: "Tricyclist",
                           10: "Pedestrian",
                           11: "Cone",
                           12: "Max_object_type"}

    invert_fusion_subtype_enum = {value: key for key, value in fusion_subtype_enum.items()}
    invert_fusion_subtype_enum["Person"] = 10

    super_class_map = {'Car': 'Car', 'car': 'Car', 'vehicle': 'Car', 'Vehicle': 'Car',

                       'Bus': 'Bus', 'bus': 'Bus', 'Van': 'Bus',

                       'Cyclist': 'Cyclist', 'cyclist': 'Cyclist', 'tricycle': 'Cyclist', 'motorcycle': 'Cyclist',
                       "Motorcycle": "Cyclist",

                       'bicycle': 'Cyclist', "Tricyclist": "Cyclist", "Motorcyclist": "Cyclist", 'Bicycle': 'Cyclist',
                       "three_wheel_car": "Cyclist",

                       'Person': 'Person', 'person': 'Person', 'Pedestrian': 'Person', 'pedestrian': 'Person',

                       "Cone": "Cone", 'cone': 'Cone', 'traffic_cone': 'Cone', 'trafficcone': 'Cone', "barrier": "Cone",
                       "stationary_obstacle": "Cone",

                       'Truck': 'Truck', 'truck': 'Truck', 'pickup_truck': 'Truck', 'construction_vehicle': 'Truck',
                       'trailer': 'Truck', "Max_object_type": "Max_object_type",

                       "Unknown": "Unknown", 'unknown': 'Unknown', 'animal': 'Unknown', 'Unknown_movable': 'Unknown',
                       'Unknown_unmovable': 'Unknown', 'unconfirm': 'Unknown',
                       "traffic_warning": "Unknown", "other": "Unknown"
                       # 'Unknown_unmovable': 'Unknown', 'unconfirm': 'Unknown', 'barrier': 'Unknown'
                       }


class Attr:
    ts = "ts"
    sensor_name = "sensor_name"
    measure_ts = "measure_ts"
    fusion_ts = "fusion_ts"
    header_ts = "header_ts"
    outer_ts = "outer_ts"
    frame_seq = "frame_seq"
    adapt_frame_seq = "adapt_frame_seq"
    sample_id = "sample_id"
    score = "score"
    object_id = "object_id"
    lidar_pos_x = "rel_x"
    lidar_pos_y = "rel_y"
    lidar_pos_z = "rel_z"
    utm_pos_x = "utm_x"
    utm_pos_y = "utm_y"
    utm_pos_z = "utm_z"
    length = "length"
    width = "width"
    height = "height"
    utm_yaw = "utm_yaw"
    lidar_yaw = "lidar_yaw"
    pitch = "pitch"
    roll = "roll"
    quaternion = "quaternion"
    utm_abs_vel_x = "utm_abs_vel_x"
    utm_abs_vel_y = "utm_abs_vel_y"
    utm_abs_vel_z = "utm_abs_vel_z"
    lidar_abs_vel_x = "lidar_abs_vel_x"
    lidar_abs_vel_y = "lidar_abs_vel_y"
    lidar_abs_vel_z = "lidar_abs_vel_z"
    utm_rel_vel_x = "utm_rel_vel_x"
    utm_rel_vel_y = "utm_rel_vel_y"
    utm_rel_vel_z = "utm_rel_vel_z"
    lidar_rel_vel_x = "lidar_rel_vel_x"
    lidar_rel_vel_y = "lidar_rel_vel_y"
    lidar_rel_vel_z = "lidar_rel_vel_z"
    abs_acc_x = "abs_acc_x"
    abs_acc_y = "abs_acc_y"
    abs_acc_z = "abs_acc_z"
    rel_acc_x = "rel_acc_x"
    rel_acc_y = "rel_acc_y"
    rel_acc_z = "rel_acc_z"
    type_id = "type_id"
    subtype_id = "subtype_id"
    category = "category"
    track_id = "track_id"
    bbox_raw = "bbox_raw"
    corners_3d = "bbox_3d"
    corners_2d = "bbox_2d"
    polygon = "polygon"
    aligned_polygon = "aligned_polygon"
    is_current_lane = "is_current_lane"
    is_key_obj = "is_key_obj"
    num_points = "num_points"
    clip_id = "clip_id"
    frame_id = "frame_id"
    uuid = "uuid"
    num_lidar_pts = "num_lidar_pts"

    img_xmin = "img_xmin"
    img_xmax = "img_xmax"
    img_ymin = "img_ymin"
    img_ymax = "img_ymax"
    img_width = "img_width"
    img_height = "img_height"
    corners_img = "corners_img"
    bbox_img = "img_bbox"

