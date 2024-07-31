import os
import pickle
from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ....utils.plot import palette, get_ax2
from ....utils.result_formatter import print_result
from ....log_mgr import logger


class ObstacleEvalHeading:
    def __init__(self, match_pair_list, out_path):
        self.match_pair_list = match_pair_list
        self.out_path = out_path
        self.pc_img_path = self.get_pc_img_path()
        self.delta_prefix = "delta_"
        self.plt_ax = None

    def get_pc_img_path(self):
        pc_img_path = os.path.join(self.out_path, "pc_imgs")
        os.makedirs(pc_img_path, exist_ok=True)
        return pc_img_path

    @staticmethod
    def acquire_heading(instant):
        utm_yaw_radian = instant.get_utm_yaw() \
            if hasattr(instant, "get_utm_yaw") and instant.get_utm_yaw() is not None else None
        utm_yaw_degree = instant.get_utm_yaw_degree() \
            if hasattr(instant, "get_utm_yaw_degree") and instant.get_utm_yaw_degree() is not None else None
        velodyne_yaw_radian = instant.get_lidar_yaw() \
            if hasattr(instant, "get_lidar_yaw") and instant.get_lidar_yaw() is not None else None
        velodyne_yaw_degree = instant.get_lidar_yaw_degree() \
            if hasattr(instant, "get_lidar_yaw_degree") and instant.get_lidar_yaw_degree() is not None else None
        return {"utm_yaw_radian": utm_yaw_radian,
                "utm_yaw_degree": utm_yaw_degree,
                "velodyne_yaw_radian": velodyne_yaw_radian,
                "velodyne_yaw_degree": velodyne_yaw_degree}

    @staticmethod
    def cal_delta(gt_info, pred_info, prefix):
        for_update = dict()
        for attr_name, value in pred_info.items():
            gt_value = gt_info.get(attr_name)
            delta_name = prefix + attr_name
            delta_value = None
            if value is not None and gt_value is not None:
                delta_value = abs(gt_value - value)
                if delta_value > 180:
                    delta_value = 360 - delta_value
            for_update.update({delta_name: delta_value})
        pred_info.update(for_update)
        return pred_info

    @staticmethod
    def add_meta_info(info, type_name, frame_seq, ts):
        info.update({"type": type_name,
                     "frame_seq": frame_seq,
                     "ts": ts})
        return info

    def collect_data(self):
        info_collect = defaultdict(list)
        for frame_seq, match_pair in enumerate(self.match_pair_list):
            gt_ts = match_pair.gt_instant.get_ts()
            gt_info = self.acquire_heading(match_pair.gt_instant)
            gt_info = self.add_meta_info(gt_info, "gt", frame_seq, gt_ts)
            info_collect["gt"].append(gt_info)
            for idx, (type_name, instant, _) in enumerate(match_pair.pred_instants):
                pred_info = self.acquire_heading(instant)
                pred_info = self.cal_delta(gt_info, pred_info, self.delta_prefix)
                pred_info = self.add_meta_info(pred_info, type_name, frame_seq, instant.get_ts())
                info_collect[type_name].append(pred_info)
        table_collect = {name: pd.DataFrame(info) for name, info in info_collect.items()}
        table_for_evaluate = pd.concat(list(table_collect.values()))
        table_collect_for_record = []
        for name, table in table_collect.items():
            table = table.drop(columns=["type"])
            table = table.set_index(["frame_seq"])
            table = table.add_prefix(name + "_")
            table_collect_for_record.append(table)
        table_for_record = pd.concat(table_collect_for_record, axis=1)
        return table_for_evaluate, table_for_record

    def get_ax2(self):
        if self.plt_ax is None:
            self.plt_ax = get_ax2()
        return self.plt_ax

    def evaluate_result(self, table_for_evaluate):
        target_attrs = ["utm_yaw_degree", "velodyne_yaw_degree"]
        for attr_name in target_attrs:
            delta_name = self.delta_prefix + attr_name
            sub_table = table_for_evaluate[table_for_evaluate[attr_name].notna()]
            if sub_table.empty:
                continue
            sub_statistic = []
            for type_name in sub_table["type"].unique().tolist():
                if type_name == "gt":
                    continue
                sub_type_table = sub_table[sub_table["type"] == type_name]
                mean_delta = round(sub_type_table[delta_name].mean(), 6)
                max_delta = round(sub_type_table[delta_name].max(), 6)
                min_delta = round(sub_type_table[delta_name].min(), 6)
                sub_statistic.append({"type": type_name,
                                      "yaw_degree" + "_mean": mean_delta,
                                      "yaw_degree" + "_max": max_delta,
                                      "yaw_degree" + "_min": min_delta})
            if not sub_statistic:
                logger.warning("{} not found".format(attr_name))
                continue

            statistic_table = pd.DataFrame(sub_statistic)
            statistic_table.set_index(["type"], inplace=True)
            print_result(statistic_table, attr_name)
            fig, ax1, ax2 = self.get_ax2()
            fig.suptitle(attr_name)
            ax1.set_title(attr_name)
            ax2.set_title(delta_name)
            palette_instant = [palette.get(name) for name in sub_table["type"].unique().tolist()]
            sns.lineplot(data=sub_table, x="frame_seq", y=attr_name, hue="type", ax=ax1, palette=palette_instant)
            delta_sub_table = sub_table[sub_table[delta_name].notna()]
            palette_instant = [palette.get(name) for name in delta_sub_table["type"].unique().tolist()]
            sns.lineplot(data=delta_sub_table, x="frame_seq", y=delta_name, hue="type", size="type", ax=ax2,
                         palette=palette_instant)
            plt.savefig(os.path.join(self.out_path, "{}.png".format(attr_name)))
            pickle.dump(fig, open(os.path.join(self.out_path, "{}.pkl".format(attr_name)), 'wb'))
            ax1.clear()
            ax2.clear()

    def save_raw_data(self, table_for_record):
        table_for_record.to_csv(os.path.join(self.out_path, "heading_raw_data.csv"))

    def close_fig(self):
        if self.plt_ax is not None:
            plt.close(self.plt_ax[0])

    def run(self):
        table_for_evaluate, table_for_record = self.collect_data()
        self.save_raw_data(table_for_record)
        self.evaluate_result(table_for_evaluate)
        self.close_fig()
