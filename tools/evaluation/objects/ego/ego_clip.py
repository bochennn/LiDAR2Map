import pandas as pd
# from cyber_record.record import Record

from utils.transform import quaternion_to_euler, ll2utm
from objects.base_objs.base_clip_obj import ClipBase
from objects.ego.ego_instant_obj import EgoInstantObj
from objects.ego.attribute_name import Attr


class EgoClip(ClipBase):
    DEFAULT_SIZE = {"length": 5,
                    "width": 2,
                    "height": 1.7}

    def __init__(self, config, data_path=None):
        self.data_path = data_path if data_path is not None else config["data"]["ego"]["data_path"]
        self.frame_objs = dict()
        self.topic = "/apollo/localization/pose"
        super().__init__()

    def parse_csv(self):
        x_offset = 5
        headers = ['utc', 'ego_lat', 'ego_lon', 'pose_quality', 'pose_alt', 'pose_ellipsoid',
                   'ins_quality', 'vel_north', 'vel_east', 'ego_vel', 'ego_yaw', 'ego_pitch',
                   'ego_roll', 'gnss_hori_pos_err', 'target_distance', 'rdm_quality', 'target_lat', 'target_lon',
                   'target_vel_x', 'target_vel_y', 'target_vel_z', 'rel_x', 'rel_y', 'rel_z', 'rel_vel_x',
                   'rel_vel_y', 'rel_vel_z']
        pd_data = pd.read_csv(self.data_path, skiprows=[0, 1], names=headers, encoding="GBK")
        pd_data.rename(columns={"target_lat": "lat",
                                "target_lon": "lon",
                                "target_vel_x": "vel_x",
                                "target_vel_y": "vel_y",
                                "target_vel_z": "vel_z"},
                       inplace=True)

        utm_x, utm_y = ll2utm(pd_data["lat"].tolist(), pd_data["lon"].tolist())
        ego_utm_x, ego_utm_y = ll2utm(pd_data["ego_lat"].tolist(), pd_data["ego_lon"].tolist())
        pd_data["rel_x"] = pd_data["rel_x"] + x_offset
        pd_data["rel_y"] = 0 - pd_data["rel_y"]
        pd_data["ego_utm_x"] = ego_utm_x
        pd_data["ego_utm_y"] = ego_utm_y
        pd_data["utm_x"] = utm_x
        pd_data["utm_y"] = utm_y
        pd_data["cal_rel_x"] = pd_data["utm_x"] - pd_data["ego_utm_x"]
        pd_data["cal_rel_y"] = pd_data["utm_y"] - pd_data["ego_utm_y"]

        pd_data["ts"] = pd_data["utc"] + 8*60*60
        pd_data.set_index(["ts"], inplace=True)
        pd_data.sort_index(inplace=True)
        return pd_data, pd_data.index.unique().tolist()

    def parse_cyber_record(self):
        extract_data = {Attr.ts: [],
                        Attr.header_ts: [],
                        Attr.utm_pos_x: [],
                        Attr.utm_pos_y: [],
                        Attr.utm_pos_z: [],
                        Attr.utm_yaw: [],
                        Attr.utm_pitch: [],
                        Attr.utm_roll: [],
                        Attr.utm_abs_vel_x: [],
                        Attr.utm_abs_vel_y: [],
                        Attr.utm_abs_vel_z: [],
                        Attr.category: [],
                        Attr.track_id: []}
        with Record(self.data_path, 'r') as record:
            for frame_seq, (topic, message, t) in enumerate(record.read_messages(self.topic)):
                utm_yaw, utm_pitch, utm_roll = quaternion_to_euler(message.pose.orientation.qw,
                                                                   message.pose.orientation.qx,
                                                                   message.pose.orientation.qy,
                                                                   message.pose.orientation.qz, )
                extract_data[Attr.ts].append(self.ts_round(message.measurement_time))
                extract_data[Attr.header_ts].append(self.ts_round(message.header.timestamp_sec))
                extract_data[Attr.utm_pos_x].append(message.pose.position.x)
                extract_data[Attr.utm_pos_y].append(message.pose.position.y)
                extract_data[Attr.utm_pos_z].append(message.pose.position.z)
                # extract_data[Attr.utm_yaw].append(utm_yaw)
                extract_data[Attr.utm_yaw].append(message.pose.heading)
                extract_data[Attr.utm_pitch].append(utm_pitch)
                extract_data[Attr.utm_roll].append(utm_roll)
                extract_data[Attr.utm_abs_vel_x].append(message.pose.linear_velocity.x)
                extract_data[Attr.utm_abs_vel_y].append(message.pose.linear_velocity.y)
                extract_data[Attr.utm_abs_vel_z].append(message.pose.linear_velocity.z)
                extract_data[Attr.category].append("Car")
                extract_data[Attr.track_id].append("rtk_gt")
        pd_data = pd.DataFrame(extract_data)
        pd_data.set_index(["ts"], inplace=True)
        pd_data.sort_index(inplace=True)
        return pd_data, pd_data.index.unique().tolist()

    def parse(self):
        if self.data_path.endswith("csv"):
            return self.parse_csv()
        else:
            return self.parse_cyber_record()

    def get_frame_obj_by_ts(self, ts):
        return EgoInstantObj(self.data.loc[ts], ts)
