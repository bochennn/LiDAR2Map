from collections import OrderedDict

import numpy as np
import pandas as pd

from ...utils.transform import transform_matrix
from .. import SensorName, sensor_name_map
from ..base_objs.base_clip_obj import ClipBase
from .attribute_name import Attr
from .tf_instant_obj import TfInstantObj

# from cyber_record.record import Record



class TfClip(ClipBase):
    def __init__(self, data_path):
        self.data_path = data_path
        self.frame_objs = dict()
        self.topic = "/tf"
        self.static_topic = "/tf_static"
        super().__init__()
        self.extrinsic = self.parse_static()
        self.sensor_name = SensorName

    def parse_static(self):
        extract_data = OrderedDict()
        with Record(self.data_path, 'r') as record:
            for frame_seq, (topic, message, t) in enumerate(record.read_messages(self.static_topic)):
                for trans in message.transforms:
                    if trans.child_frame_id not in sensor_name_map:
                        continue

                    dst_sensor = sensor_name_map[trans.header.frame_id]
                    src_sensor = sensor_name_map[trans.child_frame_id]
                    transform = trans.transform
                    rotation = {"qw": transform.rotation.qw,
                                "qx": transform.rotation.qx,
                                "qy": transform.rotation.qy,
                                "qz": transform.rotation.qz}
                    translation = {"x": transform.translation.x,
                                   "y": transform.translation.y,
                                   "z": transform.translation.z}
                    extract_data[src_sensor, dst_sensor] = {"rotation": rotation,
                                                            "translation": translation}
        # file_list = ["/home/wuchuanpan/Projects/mars_0822/mars/modules/calibration/data/e03/E03309/lidar_params/hesai64_novatel_extrinsics.yaml"]
        # import yaml
        # for file in file_list:
        #     data = yaml.safe_load(open(file, 'r'))
        #     dst_sensor = sensor_name_map[data["header"]["frame_id"]]
        #     src_sensor = sensor_name_map[data["child_frame_id"]]
        #     transform = data["transform"]
        #     rotation = {"qw": transform["rotation"]["w"],
        #                 "qx": transform["rotation"]["x"],
        #                 "qy": transform["rotation"]["y"],
        #                 "qz": transform["rotation"]["z"]}
        #     translation = {"x": transform["translation"]["x"],
        #                    "y": transform["translation"]["y"],
        #                    "z": transform["translation"]["z"]}
        #     extract_data[src_sensor, dst_sensor] = {"rotation": rotation,
        #                                             "translation": translation}
        return extract_data

    def sensor_to_local(self, sensor_name):
        transform = None
        if (sensor_name, SensorName.local) in self.extrinsic:
            raw = self.extrinsic.get((sensor_name, SensorName.local))
            transform = transform_matrix(raw["rotation"], raw["translation"])
        elif (sensor_name, SensorName.lidar_main) in self.extrinsic:
            raw = self.extrinsic.get((sensor_name, SensorName.lidar_main))
            src_to_lidar = transform_matrix(raw["rotation"], raw["translation"])
            lidar_to_imu = self.sensor_to_sensor(SensorName.lidar_main, SensorName.IMU)
            imu_to_local = self.sensor_to_local(SensorName.lidar_main)
            transform = imu_to_local.dot(lidar_to_imu.dot(src_to_lidar))
        elif (sensor_name, SensorName.IMU) in self.extrinsic:
            raw = self.extrinsic.get((sensor_name, SensorName.IMU))
            transform_to_imu = transform_matrix(raw["rotation"], raw["translation"])
            transform_to_local = self.sensor_to_local(SensorName.IMU)
            transform = transform_to_local.dot(transform_to_imu)
        return transform

    def local_to_sensor(self, sensor_name):
        import yaml
        from utils.transform import transform_matrix_
        yaml_file_path = "/mnt/data/lidar_detection/test_datasets/m2test_updated_0201/m2test/clips/clip_1701999739400/extrinsics/lidar2imu/lidar2imu.yaml"
        with open(yaml_file_path, 'r') as f:
            data = yaml.safe_load(f)
        rotation = data["transform"]["rotation"]
        translation = data["transform"]["translation"]
        lidar2imu = transform_matrix_([rotation["w"], rotation["x"], rotation["y"], rotation["z"]],
                                     [translation["x"], translation["y"], translation["z"]])
        return np.linalg.inv(lidar2imu)

        return np.linalg.inv(self.sensor_to_local(sensor_name))

    def sensor_to_sensor(self, src, dst):
        src_to_local = self.sensor_to_local(src)
        local_to_dst = self.local_to_sensor(dst)
        return local_to_dst.dot(src_to_local)

    def parse(self):
        return self.parse_record()

    def parse_record(self):
        extract_data = {Attr.header_ts: [],
                        Attr.rotation_qw: [],
                        Attr.rotation_qx: [],
                        Attr.rotation_qy: [],
                        Attr.rotation_qz: [],
                        Attr.translation_x: [],
                        Attr.translation_y: [],
                        Attr.translation_z: []}
        recorded_ts = dict()
        message_count = 0
        with Record(self.data_path, 'r') as record:
            for frame_seq, (topic, message, t) in enumerate(record.read_messages(self.topic)):
                message_count += 1
                message = message.transforms[0]
                transform = message.transform
                if message.header.timestamp_sec in recorded_ts:
                    continue
                recorded_ts[message.header.timestamp_sec] = True
                extract_data[Attr.header_ts].append(self.ts_round(message.header.timestamp_sec))
                extract_data[Attr.rotation_qw].append(transform.rotation.qw)
                extract_data[Attr.rotation_qx].append(transform.rotation.qx)
                extract_data[Attr.rotation_qy].append(transform.rotation.qy)
                extract_data[Attr.rotation_qz].append(transform.rotation.qz)
                extract_data[Attr.translation_x].append(transform.translation.x)
                extract_data[Attr.translation_y].append(transform.translation.y)
                extract_data[Attr.translation_z].append(transform.translation.z)
        pd_data = pd.DataFrame(extract_data)
        pd_data.set_index([Attr.header_ts], inplace=True)
        pd_data.sort_index(inplace=True)
        print("{} duplicate message found".format(message_count - len(recorded_ts)))
        return pd_data, pd_data.index.unique().tolist()

    def get_frame_obj_by_ts(self, ts):
        return TfInstantObj(self.data.loc[ts], ts)
