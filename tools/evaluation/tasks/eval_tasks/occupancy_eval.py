import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from tasks.eval_tasks.eval_base import EvalBase

ignore_label = 0
unique_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_map = {17: 'empty',
             0: 'noise',
             16: 'vegetation',
             15: 'manmade',
             14: 'terrain',
             13: 'sidewalk',
             12: 'other_flat',
             11: 'driveable_surface',
             10: 'truck',
             9: 'trailer',
             6: 'motorcycle',
             5: 'construction_vehicle',
             4: 'car',
             3: 'bus',
             2: 'bicycle',
             8: 'traffic_cone',
             1: 'barrier',
             7: 'pedestrian'}


class OccupancyEval(EvalBase):
    def __init__(self, config, gt_obj, pred_obj):
        super().__init__(config, gt_obj, pred_obj, match_obj=None, match_method=None)

    @staticmethod
    def load_voxel(file_path):
        with open(file_path, 'rb') as f:
            voxel = pickle.load(f).flatten()
        return voxel

    @staticmethod
    def eval(gt_file_list, pred_file_list):
        pd.set_option('display.max_columns', None)
        gt = defaultdict(int)
        pred = defaultdict(int)
        tp = defaultdict(int)
        for gt_file, pred_file in zip(gt_file_list, pred_file_list):
            gt_data = OccupancyEval.load_voxel(gt_file)
            pred_data = OccupancyEval.load_voxel(pred_file)
            pred_data = pred_data[gt_data != ignore_label]
            gt_data = gt_data[gt_data != ignore_label]
            for label in unique_label:
                gt[label] += np.sum(gt_data == label)
                pred[label] += np.sum(pred_data == label)
                tp[label] += np.sum((gt_data == label) & (pred_data == label))
        ret = {label_map[label]: 0 for label in unique_label}
        ret_table = pd.DataFrame([ret])
        iou_accu = []
        for label in unique_label:
            iou = tp[label] / (gt[label] + pred[label] - tp[label]) * 100
            if iou != 0:
                iou_accu.append(iou)
            ret_table[label_map[label]] = iou
            print("{}, iou: {:.2f}%".format(label_map[label], iou))
        print("miou: {:.2f}%".format(np.mean(iou_accu)))
        ret_table["miou"] = np.mean(iou_accu)
        print(ret_table)


if __name__ == "__main__":
    gt_root_path = "/home/wuchuanpan/Projects/occupancy-for-nuscenes/project/data/gt_voxel"
    pred_root_path = "/home/wuchuanpan/Projects/occupancy-for-nuscenes/project/data/predict_output"
    gt_file_list = [os.path.join(gt_root_path, file_name) for file_name in sorted(os.listdir(gt_root_path))]
    pred_file_list = [os.path.join(pred_root_path, file_name) for file_name in sorted(os.listdir(pred_root_path))]
    OccupancyEval.eval(gt_file_list, pred_file_list)
