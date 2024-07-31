import os
import pickle
from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ....utils.plot import palette, get_ax4
from ....utils.result_formatter import print_result


class ObstacleEvalLocation:
    def __init__(self, match_pair_list, out_path):
        self.match_pair_list = match_pair_list
        self.out_path = out_path
        self.delta_prefix = "delta_"

    @staticmethod
    def acquire_location(instant):
        velodyne_location_x = instant.get_lidar_pos_x() \
            if hasattr(instant, "get_lidar_pos_x") and instant.get_lidar_pos_x() is not None else None
        velodyne_location_y = instant.get_lidar_pos_y() \
            if hasattr(instant, "get_lidar_pos_y") and instant.get_lidar_pos_y() is not None else None
        utm_location_x = instant.get_utm_pos_x() \
            if hasattr(instant, "get_utm_pos_x") and instant.get_utm_pos_x() is not None else None
        utm_location_y = instant.get_utm_pos_y() \
            if hasattr(instant, "get_utm_pos_y") and instant.get_utm_pos_y() is not None else None
        return {"velodyne_location_x": velodyne_location_x,
                "velodyne_location_y": velodyne_location_y,
                "utm_location_x": utm_location_x,
                "utm_location_y": utm_location_y}

    @staticmethod
    def cal_delta(gt_info, pred_info, prefix):
        for_update = dict()
        for attr_name, value in pred_info.items():
            gt_value = gt_info.get(attr_name)
            delta_name = prefix + attr_name
            delta_value = None
            if value is not None and gt_value is not None:
                delta_value = abs(gt_value - value)
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
            gt_info = self.acquire_location(match_pair.gt_instant)
            gt_info = self.add_meta_info(gt_info, "gt", frame_seq, match_pair.gt_instant.get_ts())
            info_collect["gt"].append(gt_info)
            for type_name, instant, _ in match_pair.pred_instants:
                pred_info = self.acquire_location(instant)
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

    def evaluate_result(self, table_for_evaluate):
        target_attrs = [["velodyne_location_x", "velodyne_location_y"],
                        ["utm_location_x", "utm_location_y"]]
        for x_name, y_name in target_attrs:
            attr_name = x_name.split("_x")[0]
            delta_x_name = self.delta_prefix + x_name
            delta_y_name = self.delta_prefix + y_name
            sub_table = table_for_evaluate[table_for_evaluate[x_name].notna()]
            sub_statistic = []
            for type_name in sub_table["type"].unique().tolist():
                if type_name == "gt":
                    continue
                sub_type_table = sub_table[sub_table["type"] == type_name]
                mean_delta_x = round(sub_type_table[delta_x_name].mean(), 6)
                mean_delta_y = round(sub_type_table[delta_y_name].mean(), 6)
                max_delta_x = round(sub_type_table[delta_x_name].max(), 6)
                max_delta_y = round(sub_type_table[delta_y_name].max(), 6)
                min_delta_x = round(sub_type_table[delta_x_name].min(), 6)
                min_delta_y = round(sub_type_table[delta_y_name].min(), 6)
                sub_statistic.append({"type": type_name,
                                      "x_mean": mean_delta_x,
                                      "y_mean": mean_delta_y,
                                      "x_max": max_delta_x,
                                      "y_max": max_delta_y,
                                      "x_min": min_delta_x,
                                      "y_min": min_delta_y})
            statistic_table = pd.DataFrame(sub_statistic)
            statistic_table.set_index(["type"], inplace=True)
            print_result(statistic_table, attr_name)

            fig, ax1, ax2, ax3, ax4 = get_ax4()
            fig.suptitle(attr_name)
            ax1.set_title(x_name)
            ax2.set_title(y_name)
            ax3.set_title(delta_x_name)
            ax4.set_title(delta_y_name)
            palette_instant = [palette.get(name) for name in sub_table["type"].unique().tolist()]
            sns.lineplot(data=sub_table, x="frame_seq", y=x_name, hue="type", ax=ax1, palette=palette_instant)
            sns.lineplot(data=sub_table, x="frame_seq", y=y_name, hue="type", size="type", ax=ax2,
                         palette=palette_instant)
            delta_sub_table = sub_table[sub_table[delta_x_name].notna()]
            palette_instant = [palette.get(name) for name in delta_sub_table["type"].unique().tolist()]
            sns.lineplot(data=delta_sub_table, x="frame_seq", y=delta_x_name, hue="type", ax=ax3, palette=palette_instant)
            sns.lineplot(data=delta_sub_table, x="frame_seq", y=delta_y_name, hue="type", ax=ax4, palette=palette_instant)
            plt.savefig(os.path.join(self.out_path, "{}.png".format(attr_name)))
            pickle.dump(fig, open(os.path.join(self.out_path, "{}.pkl".format(attr_name)), 'wb'))
            plt.close(fig)

    def save_raw_data(self, table_for_record):
        table_for_record.to_csv(os.path.join(self.out_path, "location_raw_data.csv"), index=False)

    def run(self):
        table_for_evaluate, table_for_record = self.collect_data()
        self.save_raw_data(table_for_record)
        self.evaluate_result(table_for_evaluate)
