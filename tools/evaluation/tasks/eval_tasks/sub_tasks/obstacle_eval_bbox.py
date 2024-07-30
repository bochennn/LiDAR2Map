import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from utils.multiprocess import multiprocess_execute
from utils.result_formatter import print_result
from utils import timeit
from log_mgr import logger


class ObstacleEvalBbox:
    def __init__(self, match_pair_list, category, distance, threshold_map, out_path, test_name=None, process_num=1):
        self.match_pair_list = match_pair_list
        self.category = category
        self.distance = distance
        self.threshold_map = threshold_map
        self.out_path = out_path
        self.test_name = test_name
        self.processes = process_num

    @staticmethod
    def cal_lat_lon_error(ori_match_pair_list, target_category):
        match_pair_list = [match_pair for match_pair in ori_match_pair_list if match_pair.is_tp() and
                           match_pair.get_gt_category() == target_category]
        lat_error_list = [match_pair.get_lat_error() for match_pair in match_pair_list]
        lon_error_list = [match_pair.get_lon_error() for match_pair in match_pair_list]
        lat_mean = np.mean(lat_error_list) if lat_error_list else 0.0
        lat_max = np.max(lat_error_list) if lat_error_list else 0.0
        lon_mean = np.mean(lon_error_list) if lon_error_list else 0.0
        lon_max = np.max(lon_error_list) if lon_error_list else 0.0
        return lat_mean, lat_max, lon_mean, lon_max

    @staticmethod
    def cal_heading_degree_error(ori_match_pair_list, target_category):
        match_pair_list = [match_pair for match_pair in ori_match_pair_list if match_pair.is_tp() and
                           match_pair.get_gt_category() == target_category]
        heading_error_list = [match_pair.get_yaw_degree_delta() for match_pair in match_pair_list]
        heading_error_list = [error for error in heading_error_list if error < 150]
        h_error_mean = np.mean(heading_error_list) if heading_error_list else 0.0
        h_error_max = np.max(heading_error_list) if heading_error_list else 0.0
        return h_error_mean, h_error_max

    @staticmethod
    def cal_ap(match_pair_list, gt_num, interp_num=51):
        scores = np.array([match.get_score() for match in match_pair_list])
        match_ret = np.array([match.is_tp() for match in match_pair_list])
        yaw_similarity_ret = np.array([match.get_yaw_similarity() for match in match_pair_list])
        tph_ret = np.array([match.get_tph() for match in match_pair_list])

        score_order_idx = np.argsort(-scores)
        match_ret = match_ret[score_order_idx]
        yaw_similarity_ret = yaw_similarity_ret[score_order_idx]
        tph_ret = tph_ret[score_order_idx]

        tp = np.cumsum(match_ret)
        fp = np.cumsum(~match_ret)
        similarity_value = np.cumsum(yaw_similarity_ret)
        tph_value = np.cumsum(tph_ret)

        precision = tp / (tp + fp)
        aos = similarity_value / (tp + fp)
        tph = tph_value / (tph_value + fp)

        precision = np.maximum.accumulate(precision[::-1])[::-1]
        aos = np.maximum.accumulate(aos[::-1])[::-1]
        tph = np.maximum.accumulate(tph[::-1])[::-1]
        recall = tp / gt_num
        recall_thresholds = np.linspace(0.0, 1.0, interp_num, endpoint=True)
        interp_idx = np.searchsorted(recall, recall_thresholds, side="left")

        def select_func(x):
            return np.array([x[idx] if idx < len(x) else 0 for idx in interp_idx])

        interp_pr = select_func(precision)
        interp_aos = select_func(aos)
        interp_tph = select_func(tph)
        return interp_pr.mean(), interp_aos.mean(), interp_tph.mean(), tp[-1], fp[-1]

    def cal_all_metric(self, category, distance_name):
        distance_value = self.distance.get(distance_name)
        match_pair_list = [match for match in self.match_pair_list if match.within_distance_region(distance_value)]
        gt_num = sum(1 for match in match_pair_list if match.is_gt_target_category(category))
        pred_matches = [match for match in match_pair_list if match.is_pred_target_category(category)]
        iou_fp_num, category_fp_num, iou_cate_fp_num, unmatch_fp = 0, 0, 0, 0
        if gt_num > 0 and len(pred_matches) > 0:
            ap50, aos50, tph50, tp_num, fp_num = self.cal_ap(pred_matches, gt_num)
            iou_fp_num = sum(1 for match in pred_matches if match.is_iou_fp_only())
            category_fp_num = sum(1 for match in pred_matches if match.is_category_fp_only())
            iou_cate_fp_num = sum(1 for match in pred_matches if match.is_iou_category_fp())
            unmatch_fp = sum(1 for match in pred_matches if match.is_extra_fp())
        else:
            ap50, aos50, tph50, tp_num = 0, 0, 0, 0
            fp_num = sum(1 for match_pair in match_pair_list if match_pair.is_fp() and
                         match_pair.is_pred_target_category(category))
        lat_error_mean, lat_error_max, lon_error_mean, lon_error_max = self.cal_lat_lon_error(match_pair_list,
                                                                                              category)
        heading_error_mean, heading_error_max = self.cal_heading_degree_error(match_pair_list, category)
        result = OrderedDict({"category": category,
                              "distance": distance_name,
                              "AP@50": round(ap50 * 100, 2),
                              "AOS@50": round(aos50 * 100, 2),
                              "TPH@50": round(tph50 * 100, 2),
                              "lat_error_mean": round(lat_error_mean, 2),
                              "lat_error_max": round(lat_error_max, 2),
                              "lon_error_mean": round(lon_error_mean, 2),
                              "lon_error_max": round(lon_error_max, 2),
                              "heading_error_mean": round(heading_error_mean, 2),
                              "heading_error_max": round(heading_error_max, 2),
                              "tp": tp_num,
                              "fp": fp_num,
                              "iou_fp": iou_fp_num,
                              "cate_fp": category_fp_num,
                              "iou_cate_fp": iou_cate_fp_num,
                              "unmatch_fp": unmatch_fp,
                              "fn": gt_num - tp_num,
                              "precision": round(tp_num / (tp_num + fp_num + 1e-6) * 100, 2),
                              "recall": round(tp_num / (gt_num + 1e-6) * 100, 2)})
        return result

    def executor(self, dimension_list, child_conn=None):
        ret = []
        for category, distance in dimension_list:
            ret.append(self.cal_all_metric(category, distance))
        if child_conn is not None:
            child_conn.send(ret)
        else:
            return ret

    def obstacle_result_to_excel(self, result_table_list):
        os.makedirs(self.out_path, exist_ok=True)
        out_file_path = os.path.join(self.out_path,
                                     "{}.xlsx".format(self.test_name) if self.test_name is not None else "perception.xlsx")
        with pd.ExcelWriter(out_file_path) as writer:
            for table, table_name in result_table_list:
                table.to_excel(writer, sheet_name=table_name)
        logger.info("results saved in {}".format(out_file_path))

    def obstacle_result_process(self, results):
        result_table = pd.DataFrame(results)
        dimension_name = ["category", "distance"]
        dimension_name += ["iou_fp", "cate_fp", "iou_cate_fp", "unmatch_fp"]
        result_str = ""
        result_table_list = []
        for metric_name in result_table.columns:
            if metric_name not in dimension_name:
                metric_table = result_table.pivot(index="category", columns="distance", values=metric_name)
                metric_table.sort_index(inplace=True)
                if metric_name in ["AOS@50", "TPH@50", "heading_error_mean", "heading_error_max"]:
                    if "Cone" in metric_table.index:
                        metric_table.drop(labels="Cone", inplace=True)
                if metric_name in ["AP@50", "AOS@50", "TPH@50", "precision", "recall"]:
                    metric_table.loc["mean"] = metric_table.mean().round(2)
                elif metric_name in ["tp", "fp", "fn"]:
                    metric_table.loc["total"] = metric_table.sum()
                sub_ret_str = print_result(metric_table, metric_name)
                result_str = result_str + sub_ret_str + "\n"
                result_table_list.append((metric_table, metric_name))
        logger.info("\n{}".format(result_str))
        if self.out_path is not None:
            self.obstacle_result_to_excel(result_table_list)
        else:
            logger.warning("out_path not provided, KPI output will not be saved")

    def run(self):
        arg_list = []
        for category in self.category:
            for distance in self.distance:
                arg_list.append((category, distance))
        eval_result = multiprocess_execute(self.executor, arg_list, processes=self.processes)
        self.obstacle_result_process(eval_result)
        return eval_result
