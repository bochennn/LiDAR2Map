import os
from collections import defaultdict

import simplejson as json

from ....log_mgr import logger


class ObstacleFailedVisual:
    def __init__(self, match_pair_list, out_path, raw_lidar_path=None, target="failed"):
        self.match_pair_list = match_pair_list
        self.out_path = os.path.join(out_path, target)
        self.raw_lidar_path = raw_lidar_path
        self.target = target
        self.lidar_out_path = self.init_lidar_out_path()
        self.gt_out_path = self.init_gt_out_path()
        self.fp_out_path = self.init_fp_out_path()
        self.fn_out_path = self.init_fn_out_path()
        self.tp_out_path = self.init_tp_out_path()

    @staticmethod
    def init_target_path(target_path):
        os.makedirs(target_path, exist_ok=True)
        return target_path

    def init_lidar_out_path(self):
        return self.init_target_path(os.path.join(self.out_path, "lidar"))

    def init_gt_out_path(self):
        return self.init_target_path(os.path.join(self.out_path, "label", "gt"))

    def init_fp_out_path(self):
        return self.init_target_path(os.path.join(self.out_path, "label", "fp"))

    def init_fn_out_path(self):
        return self.init_target_path(os.path.join(self.out_path, "label", "fn"))

    def init_tp_out_path(self):
        return self.init_target_path(os.path.join(self.out_path, "label", "tp"))

    @staticmethod
    def dump_json(obj_record, out_path):
        for ts, objs in obj_record.items():
            with open(os.path.join(out_path, "{}.json".format(ts)), 'w') as f:
                json.dump(objs, f)

    def link_lidar_file(self, ts_list):
        if self.raw_lidar_path is None:
            logger.info("path of lidar pointcloud file not been provided, lidar frames will not be created")
            return
        for ts in ts_list:
            raw_lidar_file = os.path.join(self.raw_lidar_path, "{}.bin".format(ts))
            target_lidar_file = os.path.join(self.lidar_out_path, "{}.bin".format(ts))
            if not os.path.exists(raw_lidar_file):
                raw_lidar_file = os.path.join(self.raw_lidar_path, "{}.pcd".format(ts))
                target_lidar_file = os.path.join(self.lidar_out_path, "{}.pcd".format(ts))
            raw_lidar_file = os.path.realpath(raw_lidar_file)
            os.symlink(raw_lidar_file, target_lidar_file)
        logger.info("{} symbol link of lidar frame created".format(len(ts_list)))

    def get_eval_ret_record(self):
        gt_record = defaultdict(list)
        fp_record = defaultdict(list)
        fn_record = defaultdict(list)
        tp_record = defaultdict(list)
        for match in self.match_pair_list:
            ts = match.gt_ts if match.gt_ts is not None else match.get_ts()
            if not match.is_tp():
                if match.gt_valid():
                    fn_record[ts].append(match.gt_instant.get_visual_json_obj())
                if match.pred_valid():
                    fp_record[ts].append(
                        match.pred_instant.get_visual_json_obj("fp, iou: {:.2f}".format(match.match_factor)))
            else:
                tp_record[ts].append(match.pred_instant.get_visual_json_obj("tp, iou: {:.2f}".format(match.match_factor)))
                gt_record[ts].append(match.gt_instant.get_visual_json_obj())
        if self.target == "failed":
            return [(fp_record, self.fp_out_path),
                    (fn_record, self.fn_out_path)]
        elif self.target == "all":
            return [(fp_record, self.fp_out_path),
                    (fn_record, self.fn_out_path),
                    (gt_record, self.gt_out_path),
                    (tp_record, self.tp_out_path)]

    def start(self):
        ret_record_list = self.get_eval_ret_record()
        unique_ts_list = []
        for record, out_path in ret_record_list:
            self.dump_json(record, out_path)
            if len(record) > 0:
                unique_ts_list.extend(list(record.keys()))
        unique_ts_list = list(set(unique_ts_list))
        self.link_lidar_file(unique_ts_list)
