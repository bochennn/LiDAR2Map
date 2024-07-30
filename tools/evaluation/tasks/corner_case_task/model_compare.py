import os
import pickle

from collections import defaultdict
from log_mgr import logger


class ModelCompare:
    def __init__(self, model_A_info, model_B_info, out_root_path):
        self.model_A_name = model_A_info["model_name"]
        self.model_A_data_path = model_A_info["data_path"]
        self.model_B_name = model_B_info["model_name"]
        self.model_B_data_path = model_B_info["data_path"]
        self.out_path = out_root_path
        self.model_A_out_path = self.init_model_A_path()
        self.model_B_out_path = self.init_model_B_path()
        self.model_A_match_pair_list = self.get_model_A_match_pair_list()
        self.model_B_match_pair_list = self.get_model_B_match_pair_list()

    def init_model_A_path(self):
        model_A_path = os.path.join(self.out_path, self.model_A_name)
        os.makedirs(model_A_path, exist_ok=True)
        return model_A_path

    def init_model_B_path(self):
        model_B_path = os.path.join(self.out_path, self.model_B_name)
        os.makedirs(model_B_path, exist_ok=True)
        return model_B_path

    def get_model_A_match_pair_list(self):
        with open(self.model_A_data_path, 'rb') as f:
            return pickle.load(f)

    def get_model_B_match_pair_list(self):
        with open(self.model_B_data_path, 'rb') as f:
            return pickle.load(f)

    def collect_ret_status(self, record, match_pair_list, model_name):
        for match in match_pair_list:
            if not match.gt_valid():
                continue
            if match.gt_instant.get_lidar_pos_x() < 0:
                continue
            ts = match.gt_instant.get_ts()
            uuid = match.gt_instant.get_uuid()
            if match.is_tp():
                status = "tp"
            elif match.is_fp():
                status = "fp"
            else:
                status = "fn"
            record[uuid][model_name] = (status, match.pred_instant)
            if "ts" not in record[uuid]:
                record[uuid]["ts"] = ts
            if "gt" not in record[uuid]:
                record[uuid]["gt"] = match.gt_instant
        return record

    def start(self):
        record = defaultdict(dict)
        record = self.collect_ret_status(record, self.model_A_match_pair_list, self.model_A_name)
        record = self.collect_ret_status(record, self.model_B_match_pair_list, self.model_B_name)
        A_better_gt = defaultdict(list)
        A_better_model_A = defaultdict(list)
        A_better_model_B = defaultdict(list)
        B_better_gt = defaultdict(list)
        B_better_model_A = defaultdict(list)
        B_better_model_B = defaultdict(list)

        A_better_num = 0
        B_better_num = 0
        for uuid, obj_compare_info in record.items():
            if self.model_A_name not in obj_compare_info or self.model_B_name not in obj_compare_info:
                continue
            model_A_info = obj_compare_info[self.model_A_name]
            model_B_info = obj_compare_info[self.model_B_name]
            ts = obj_compare_info["ts"]
            gt_instant = obj_compare_info["gt"]
            if model_A_info[0] == "tp" and model_B_info != "tp":
                A_better_num += 1
                A_better_gt[ts].append(gt_instant)
                A_better_model_A[ts].append(model_A_info[1])
                if model_B_info[1] is not None:
                    A_better_model_B[ts].append(model_B_info[1])

            if model_B_info[0] == "tp" and model_A_info != "tp":
                B_better_num += 1
                B_better_gt[ts].append(gt_instant)
                B_better_model_B[ts].append(model_B_info[1])
                if model_A_info[1] is not None:
                    B_better_model_A[ts].append(model_A_info[1])
        logger.info("{} better than {} num: {}".format(self.model_A_name, self.model_B_name, A_better_num))
        logger.info("{} better than {} num: {}".format(self.model_B_name, self.model_A_name, B_better_num))


if __name__ == "__main__":
    second_model_path = "/mnt/data/lidar_detection/1016_ret/failed_case/failed_case.pkl"
    regnet_model_path = "/mnt/data/lidar_detection/regnet_ret/failed_case/failed_case.pkl"
    out_path = "/mnt/data/lidar_detection/model_compare/second_regnet"
    ModelCompare({"model_name": "SECOND", "data_path": second_model_path},
                 {"model_name": "RegNet", "data_path": regnet_model_path},
                 out_path).start()