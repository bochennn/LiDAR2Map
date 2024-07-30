import os
from collections import OrderedDict

import numpy as np
from objects.base_objs.base_clip_obj import ClipBase
from cyber_record.record import Record
import open3d as o3d
from utils.pointcloud_ops import load_bin, numpy_to_bin, load_pcd, numpy_to_pcd


class ObstacleClipPointCloud(ClipBase):
    def __init__(self, data_path, out_path=None, topic="/apollo/sensor/hesai64/PointCloud2"):
        self.data_path = data_path
        self.out_path = out_path
        self.topic = topic
        self.frame_objs = dict()
        super().__init__()

    def parse_bins(self):
        clip_data = OrderedDict()
        ts_list = []
        for name in sorted(os.listdir(self.data_path)):
            if "=" in name:
                ts = name.split("=")[2]
            elif "__" in name:
                ts = name.split("__")[-1].split(".")[0]
            elif "_" in name:
                ts = name.split("_")[1]
            else:
                ts = name.split(".")[0]

            if len(ts) == 19:
                ts = float(ts) / 1e9
            elif len(ts) == 16:
                ts = float(ts) / 1e6
            elif len(ts) == 13:
                ts = float(ts) / 1e3
            ts = self.ts_round(ts)
            file_path = os.path.join(self.data_path, name)
            clip_data[ts] = file_path
            ts_list.append(ts)
        return clip_data, ts_list

    def parse(self):
        if os.path.isdir(self.data_path):
            return self.parse_bins()
        else:
            return self.parse_record()

    def init_out_bin_path(self):
        out_bin_path = os.path.join(self.out_path, "bins")
        os.makedirs(out_bin_path, exist_ok=True)
        self.out_path = out_bin_path

    def parse_record(self):
        self.init_out_bin_path()
        clip_data = OrderedDict()
        ts_list = []
        with Record(self.data_path, 'r') as record:
            for topic, message, _ in record.read_messages(self.topic):
                ts = round(message.measurement_time, 6)
                pc_data = np.array([[point.x, point.y, point.z, point.intensity] for point in message.point], dtype=np.float32)
                out_file = os.path.join(self.out_path, "{}.bin".format(ts))
                # out_file = os.path.join(self.out_path, "{}.pcd".format(ts))
                # numpy_to_pcd(pc_data, out_file)
                pc_data.tofile(out_file)
                clip_data[ts] = out_file
                ts_list.append(ts)
        return clip_data, ts_list

    def get_frame_obj_by_ts(self, ts, color=(0.38, 0.376, 0.365)):
        pcd = o3d.geometry.PointCloud()
        file_path = self.data[ts]
        if file_path.endswith("bin"):
            points = load_bin(file_path)
        elif file_path.endswith("pcd"):
            points = load_pcd(file_path)
        else:
            raise NotImplementedError("unsupported format of {}".format(file_path))
        pcd.points = o3d.utility.Vector3dVector(points)
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector([color] * len(points))
        return pcd
