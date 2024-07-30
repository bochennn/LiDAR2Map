from collections import OrderedDict, defaultdict
import os
import glob

import cv2
import numpy as np
import pandas as pd
# from tabulate import tabulate

from objects.trafficlight.trafficlight_clip_gt import TrafficlightClipGt
from objects.trafficlight.trafficlight_clip_pred import TrafficlightClipPred
from objects.trafficlight.trafficlight_match_obj import TrafficlightMatchObj as MatchObj
# from objects.trafficlight.parsers.attribute_tool import TrafficLightEnum
from tasks.eval_tasks.eval_base import EvalBase
from utils.index_match import token_match, ts_match
from utils.result_formatter import print_result
from log_mgr import logger


def cal_metric_tl(match_pair_list, gt_num, interp_num=101):
    scores = np.array([match.get_score() for match in match_pair_list])
    match_ret = np.array([match.is_tp() for match in match_pair_list])

    score_order_idx = np.argsort(-scores)
    match_ret = match_ret[score_order_idx]

    tp = np.cumsum(match_ret)
    fp = np.cumsum(~match_ret)

    precision = tp / (tp + fp)

    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = tp / gt_num
    recall_thresholds = np.linspace(0.0, 1.0, interp_num, endpoint=True)
    interp_idx = np.searchsorted(recall, recall_thresholds, side="left")

    def select_func(x):
        return np.array([x[idx] if idx < len(x) else 0 for idx in interp_idx])
    interp_pr = select_func(precision)
    return interp_pr.mean(), tp[-1], fp[-1]


class TrafficlightEval(EvalBase):
    def __init__(self, config):
        super().__init__(config, TrafficlightClipGt(config), TrafficlightClipPred(config), MatchObj)
        self.category = config["category"]
        self.distance = config["distance"]
        self.img_path = config.get("img_path")
        self.img_data = self.get_img_data()
        self.match_obj = MatchObj
        self.match_pair_list = []

    @staticmethod
    def get_iou(instant1, instant2):
        polygon1 = instant1.get_polygon()
        polygon2 = instant2.get_polygon()
        intersection = polygon1.intersection(polygon2)
        return intersection.area / (polygon1.area + polygon2.area - intersection.area)

    def get_img_data(self):
        img_data = dict()
        if self.img_path is not None:
            all_img_file = glob.glob(os.path.join(self.img_path, "**", "*.jpg"), recursive=True)
            for img_file in all_img_file:
                # token = os.path.basename(img_file).split("_")[0]
                token = os.path.splitext(os.path.basename(img_file))[0]
                # token = round(float(token), 6)
                img_data[token] = img_file
        return img_data

    def show_img(self, match_pair_list):
        tp_record = defaultdict(list)
        fp_record = defaultdict(list)
        fn_record = defaultdict(list)
        unique_ts_list = set()
        for match in match_pair_list:
            token = match.pred_instant.get_ts() if match.pred_valid() else match.gt_instant.get_ts()
            unique_ts_list.add(token)
            if match.is_tp():
                tp_record[token].append(match.pred_instant)
            if match.is_fp():
                fp_record[token].append((match.pred_instant, match.iou))
            if match.is_fn():
                fn_record[token].append(match.gt_instant)
        for token in list(unique_ts_list):
            img_file = self.img_data[token]
            img = cv2.imread(img_file)
            tp_num = 0
            fp_num = 0
            fn_num = 0
            for tp_instant in tp_record[token]:
                tp_num += 1
                p1, p2 = tp_instant.get_2_corners()
                p1 = (int(p1[0]), int(p1[1]))
                p2 = (int(p2[0]), int(p2[1]))
                cv2.rectangle(img, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
            for fp_instant, iou in fp_record[token]:
                fp_num += 1
                p1, p2 = fp_instant.get_2_corners()
                p1 = (int(p1[0]), int(p1[1]))
                p2 = (int(p2[0]), int(p2[1]))
                cv2.rectangle(img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, "iou: {:.2f}".format(iou), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            for fn_instant in fn_record[token]:
                fn_num += 1
                p1, p2 = fn_instant.get_2_corners()
                p1 = (int(p1[0]), int(p1[1]))
                p2 = (int(p2[0]), int(p2[1]))
                cv2.rectangle(img, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)
            if fn_num == 0 and fp_num == 0:
                continue
            else:
                logger.info(token)
                title = token
                cv2.imwrite(os.path.join("/home/wuchuanpan/PycharmProjects/experiment/data/20240126/trafficlight/det_failed", "{}.jpg".format(token)),
                            img)
                # cv2.imshow(title, img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    def eval_det(self, matched_ts_list, show=False):
        match_pair_list = []
        for gt_ts, pred_ts in matched_ts_list:
            gt_frame = self.gt_obj.get_frame_obj_by_ts(gt_ts)
            pred_frame = self.pred_obj.get_frame_obj_by_ts(pred_ts)
            gt_instants = gt_frame.get_instant_objects()
            pred_instants = pred_frame.get_det_instant_objects()
            matched_index_list = []
            for gt_instant in gt_instants:
                iou_list = [self.get_iou(gt_instant, pred_instant) for pred_instant in pred_instants]
                matched_index = np.argmax(iou_list)
                iou = iou_list[matched_index]
                match_pair_list.append(self.match_obj(gt_instant,
                                                      pred_instants[matched_index] if iou > 0 else None,
                                                      iou,
                                                      self.category))
                if iou > 0:
                    matched_index_list.append(matched_index)
                # if iou <= 0:
                #     match_pair_list.append(self.match_obj(None, pred_instants[0], iou, self.category))
            remain_pred_instants = [instant for idx, instant in enumerate(pred_instants) if idx not in matched_index_list]
            for pred_instant in remain_pred_instants:
                match_pair_list.append(self.match_obj(None, pred_instant, 0, self.category))

        failed_token_list = list(set(self.gt_obj.get_ts_list()) - set(self.pred_obj.get_ts_list()))
        for token in failed_token_list:
            gt_frame = self.gt_obj.get_frame_obj_by_ts(token)
            for instant in gt_frame.get_instant_objects():
                match_pair_list.append(self.match_obj(instant, None, 0, self.category))

        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        for match in match_pair_list:
            if match.is_tp():
                shape = match.gt_instant.get_shape()
                tp[shape] += 1
            if match.is_fp():
                shape = match.pred_instant.get_shape()
                fp[shape] += 1
            if match.is_fn():
                shape = match.gt_instant.get_shape()
                fn[shape] += 1
        if show:
            self.show_img(match_pair_list)
        record = []
        for shape in ["vertical_light", "horizontal_light", "single_light"]:
            tp_num = tp[shape]
            fp_num = fp[shape]
            fn_num = fn[shape]
            record.append({"shape": shape,
                           "precision": round(tp_num / (tp_num + fp_num + 1e-6) * 100, 2),
                           "recall": round(tp_num / (tp_num + fn_num + 1e-6) * 100, 2),
                           "tp": tp_num,
                           "fp": fp_num,
                           "fn": fn_num})
        table = pd.DataFrame(record)
        table.set_index(["shape"], inplace=True)
        print_result(table, "trafficlight detection")

    def eval_classify(self, matched_ts_list, show=False):
        match_pair_list = []
        for gt_ts, pred_ts in matched_ts_list:
            gt_frame = self.gt_obj.get_frame_obj_by_ts(gt_ts)
            pred_frame = self.pred_obj.get_frame_obj_by_ts(pred_ts)
            gt_instants = gt_frame.get_instant_objects()
            pred_instants = pred_frame.get_classify_instant_objects()
            for gt_instant in gt_instants:
                iou_list = [self.get_iou(gt_instant, pred_instant) for pred_instant in pred_instants]
                matched_index = np.argmax(iou_list)
                iou = iou_list[matched_index]
                match_pair_list.append(self.match_obj(gt_instant,
                                                      pred_instants[matched_index] if iou > 0 else None,
                                                      iou,
                                                      self.category))
        logger.info("number of match pair: {}".format(len(match_pair_list)))
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        for match in match_pair_list:
            if match.is_color_tp():
                color = match.gt_instant.get_color()
                tp[color] += 1
            if match.is_color_fp():
                color = match.pred_instant.get_color()
                fp[color] += 1
                if show:
                    self.show_img(match, match.pred_instant.get_ts(), "color_fp")
            if match.is_color_fn():
                color = match.gt_instant.get_color()
                fn[color] += 1
                if show:
                    self.show_img(match, match.gt_instant.get_ts(), "color_fn")
        record = []
        for color in ["Unknown", "Red", "Yellow", "Green", "Black"]:
            tp_num = tp[color]
            fp_num = fp[color]
            fn_num = fn[color]
            record.append({"color": color,
                           "precision": round(tp_num / (tp_num + fp_num + 1e-6) * 100, 2),
                           "recall": round(tp_num / (tp_num + fn_num + 1e-6) * 100, 2),
                           "tp": tp_num,
                           "fp": fp_num,
                           "fn": fn_num})
        table = pd.DataFrame(record)
        table.set_index(["color"], inplace=True)
        print_result(table, "trafficlight classify")

    def eval_classify_tmp(self):
        import json
        def get_color_map(color_label_root_path):
            color_map = dict()
            colors = ["unknown", "red", "yellow", "green", "black"]
            for root, dirs, files in os.walk(color_label_root_path):
                for file in files:
                    if file.endswith(".jpg"):
                        token = os.path.splitext(file)[0]
                        color_id = os.path.basename(root)
                        color_map[token] = colors[int(color_id)]
            return color_map
        gt_data_path = "/mnt/data/trafficlight/test_datasets/20240124_front_wide/classify"
        pred_data_path = "/home/wuchuanpan/Projects/trafficlight_projects/trafficlight_0126/trafficlight/classfiy/resnet_18/runs/exp/classify_ret.json"
        gt_color_map = get_color_map(gt_data_path)
        with open(pred_data_path, 'r') as f:
            pred_color_map = json.load(f)
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        failed_record = dict()
        for token, color in gt_color_map.items():
            pred_color = pred_color_map[token]
            if color == pred_color:
                tp[color] += 1
            else:
                fp[color] += 1
                fn[color] += 1
                failed_record[token] = {"gt": color,
                                        "pred": pred_color}
        record = []
        for color in ["unknown", "red", "yellow", "green", "black"]:
            tp_num = tp[color]
            fp_num = fp[color]
            fn_num = fn[color]
            record.append({"color": color,
                           "precision": round(tp_num / (tp_num + fp_num + 1e-6) * 100, 2),
                           "recall": round(tp_num / (tp_num + fn_num + 1e-6) * 100, 2),
                           "tp": tp_num,
                           "fp": fp_num,
                           "fn": fn_num})
        table = pd.DataFrame(record)
        table.set_index(["color"], inplace=True)
        print_result(table, "trafficlight classify")
        print(failed_record)


    def eval_e2e(self, matched_ts_list, show=False):
        match_pair_list = []
        for gt_ts, pred_ts in matched_ts_list:
            gt_frame = self.gt_obj.get_frame_obj_by_ts(gt_ts)
            pred_frame = self.pred_obj.get_frame_obj_by_ts(pred_ts)
            gt_instants = gt_frame.get_instant_objects()
            pred_instants = pred_frame.get_e2e_instant_objects()
            for gt_instant in gt_instants:
                iou_list = [self.get_iou(gt_instant, pred_instant) for pred_instant in pred_instants]
                matched_index = np.argmax(iou_list)
                iou = iou_list[matched_index]
                match_pair_list.append(self.match_obj(gt_instant,
                                                      pred_instants[matched_index] if iou > 0 else None,
                                                      iou,
                                                      self.category))
        logger.info("number of match pair: {}".format(len(match_pair_list)))
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        for match in match_pair_list:
            # if match.gt_instant.get_color() == "unknown":
            #     self.show_img(match, match.gt_instant.get_ts(), "unknown")
            if match.is_color_tp():
                color = match.gt_instant.get_color()
                tp[color] += 1
            if match.is_color_fp():
                color = match.pred_instant.get_color()
                fp[color] += 1
                if show:
                    self.show_img(match, match.pred_instant.get_ts(), "e2e_fp")
            if match.is_color_fn():
                color = match.gt_instant.get_color()
                fn[color] += 1
                if show:
                    self.show_img(match, match.gt_instant.get_ts(), "e2e_fn")
        record = []
        for color in ["Unknown", "Red", "Yellow", "Green", "Black"]:
            tp_num = tp[color]
            fp_num = fp[color]
            fn_num = fn[color]
            record.append({"color": color,
                           "precision": round(tp_num / (tp_num + fp_num + 1e-6) * 100, 2),
                           "recall": round(tp_num / (tp_num + fn_num + 1e-6) * 100, 2),
                           "tp": tp_num,
                           "fp": fp_num,
                           "fn": fn_num})
        table = pd.DataFrame(record)
        table.set_index(["color"], inplace=True)
        print_result(table, "trafficlight e2e")

    def run(self):
        matched_ts_list = token_match([self.gt_obj.get_ts_list(), self.pred_obj.get_ts_list()], verbose=True)
        # matched_ts_list = ts_match([self.gt_obj.get_ts_list(), self.pred_obj.get_ts_list()], verbose=True)
        self.eval_det(matched_ts_list, show=True)
        self.eval_classify_tmp()
        # self.eval_classify(matched_ts_list, show=False)
        # self.eval_e2e(matched_ts_list, show=True)

        # match_info_list = self.match()
        # self.match_pair_list = self.create_match_pairs(match_info_list)
        # eval_results = self.eval_trafficlight()
        # self.result_process(eval_results)
        # self.eval_trafficlight_color()
