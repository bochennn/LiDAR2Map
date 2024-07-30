import os
from collections import defaultdict

import numpy as np
import pandas as pd

from ....objects.obstacle.parsers.attribute_tool import Attr
from ....tools.datasets_tool.data_distribution_statistic import data_statistics
from ....utils.pointcloud_ops import load_bin, points_crop


class ObstacleSampleDistribution:
    def __init__(self, match_pair_list, out_path, raw_lidar_path):
        self.match_pair_list = match_pair_list
        self.out_path = out_path
        self.raw_lidar_path = raw_lidar_path

    def complete_lidar_points_info(self, instant_list):
        instant_record = defaultdict(list)
        for instant in instant_list:
            ts_str = "{:.6f}".format(instant.get_ts())
            instant_record[ts_str].append(instant)
        for ts_str, instants in instant_record.items():
            lidar_file = os.path.join(self.raw_lidar_path, "{}.bin".format(ts_str))
            for instant in instants:
                lidar_points = load_bin(lidar_file)
                corners = instant.get_corners_3d()
                points_inside = points_crop(lidar_points,
                                            [np.min(corners[:, 0]), np.max(corners[:, 0])],
                                            [np.min(corners[:, 1]), np.max(corners[:, 1])],
                                            [np.min(corners[:, 2]), np.max(corners[:, 2])],
                                            )
                instant.data[Attr.num_lidar_pts] = len(points_inside)
        return instant_list

    def data_distribution_on_instants(self, instant_list, out_path):
        if not hasattr(instant_list[0], Attr.num_lidar_pts):
            instant_list = self.complete_lidar_points_info(instant_list)
        df_data = pd.concat([instant.data for instant in instant_list], axis=1).T
        data_statistics(df_data, out_path)

    def start(self):
        tp_instant_list = [match.pred_instant for match in self.match_pair_list if match.is_tp()]
        tp_out_path = os.path.join(self.out_path, "data_distribution", "tp")
        os.makedirs(tp_out_path, exist_ok=True)
        self.data_distribution_on_instants(tp_instant_list, tp_out_path)

        fp_instant_list = [match.pred_instant for match in self.match_pair_list if match.is_fp()]
        fp_out_path = os.path.join(self.out_path, "data_distribution", "fp")
        os.makedirs(fp_out_path, exist_ok=True)
        self.data_distribution_on_instants(fp_instant_list, fp_out_path)

        fn_instant_list = [match.gt_instant for match in self.match_pair_list if match.is_fn()]
        fn_out_path = os.path.join(self.out_path, "data_distribution", "fn")
        os.makedirs(fn_out_path, exist_ok=True)
        self.data_distribution_on_instants(fn_instant_list, fn_out_path)
