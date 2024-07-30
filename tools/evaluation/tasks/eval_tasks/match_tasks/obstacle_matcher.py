import warnings
from itertools import zip_longest

import numpy as np
from scipy.optimize import linear_sum_assignment
from geopandas import GeoSeries
from shapely import points
from shapely.geometry import Point

from utils.index_match import ts_match as ts_match_func
from utils.index_match import token_match
from utils.multiprocess import multiprocess_execute
from utils import timeit
from log_mgr import logger


def get_iou_matrix(gt_polygons, pred_polygons):
    iou_matrix = []
    pred_series = GeoSeries(pred_polygons)
    pred_ares = pred_series.area.to_numpy()
    with warnings.catch_warnings(record=True) as w:
        for gt_polygon in gt_polygons:
            inter_areas = pred_series.intersection(gt_polygon).area.to_numpy()
            ious = inter_areas / (gt_polygon.area + pred_ares - inter_areas)
            iou_matrix.append(ious)
        if len(w) > 0:
            "TODO: add record"
            pass
    return np.array(iou_matrix)


def get_iop_matrix(gt_polygons, pred_polygons):
    iop_matrix = []
    pred_series = GeoSeries(pred_polygons)
    pred_ares = pred_series.area.to_numpy()
    with warnings.catch_warnings(record=True) as w:
        for gt_polygon in gt_polygons:
            inter_areas = pred_series.intersection(gt_polygon).area.to_numpy()
            iops = inter_areas / pred_ares
            iop_matrix.append(iops)
        if len(w) > 0:
            "TODO: add record"
            pass
    return np.array(iop_matrix)


def get_LET_iou_matrix(gt_instants, pred_instants, lon_affine_weighted=True):
    iou_matrix = []
    with warnings.catch_warnings(record=True) as w:
        for gt_instant in gt_instants:
            gt_polygon = gt_instant.get_polygon()
            gt_center = gt_instant.get_center_3d()
            pred_aligned_polygons = [pred_instant.get_aligned_polygon(gt_center) for pred_instant in pred_instants]
            pred_series = GeoSeries(pred_aligned_polygons)
            pred_ares = pred_series.area.to_numpy()
            inter_areas = pred_series.intersection(gt_polygon).area.to_numpy()
            ious = inter_areas / (gt_polygon.area + pred_ares - inter_areas)

            ori_pred_series = GeoSeries([pred_instant.get_polygon() for pred_instant in pred_instants])
            ori_pred_ares = ori_pred_series.area.to_numpy()
            ori_inter_area = ori_pred_series.intersection(gt_polygon).area.to_numpy()
            ori_ious = ori_inter_area / (gt_polygon.area + ori_pred_ares - ori_inter_area)
            if lon_affine_weighted:
                lon_affine_list = [pred_instant.get_longitudinal_affine(gt_center) for pred_instant in pred_instants]
                ious = [iou * lon_affine if lon_affine > 0 else ori_iou
                        for iou, ori_iou, lon_affine in zip(ious, ori_ious, lon_affine_list)]
            iou_matrix.append(ious)
        if len(w) > 0:
            "TODO: add record"
            pass
    return np.array(iou_matrix)


def get_dist_matrix(gt_points, pred_points):
    dist_matrix = []
    p_points = points(pred_points)
    for gt_point in gt_points:
        g_point = Point(gt_point)
        dist_matrix.append(g_point.distance(p_points))
    return np.array(dist_matrix)


def get_munkres_match(iou_matrix):
    cost_matrix = np.ones_like(iou_matrix) - iou_matrix
    match_index_list = linear_sum_assignment(cost_matrix)
    return match_index_list


class FrameMatchStatus:
    normal = "normal"
    fn = "fn"
    fp = "fp"


def create_frame_match_pairs(gt_frame, pred_frame, match_obj,
                             iou_record, match_uuid_list,
                             filter_bad=True):
    match_pair_list = []
    gt_ts = gt_frame.get_ts()
    gt_assigned_idx = dict()
    pred_assigned_idx = dict()
    gt_uuid_record = {instant.get_uuid(): instant for instant in gt_frame.get_instant_objects()}
    pred_uuid_record = {instant.get_uuid(): instant for instant in pred_frame.get_instant_objects()}
    for gt_uuid, pred_uuid in match_uuid_list:
        iou = iou_record[gt_uuid, pred_uuid]
        if iou < 0.1 and filter_bad:
            continue
        gt_assigned_idx[gt_uuid] = True
        pred_assigned_idx[pred_uuid] = True
        match_pair_list.append(match_obj(gt_uuid_record[gt_uuid],
                                         pred_uuid_record[pred_uuid],
                                         iou,
                                         gt_ts=gt_ts))
    fn_match = [match_obj(gt_instant, None, 0, gt_ts=gt_ts)
                for gt_uuid, gt_instant in gt_uuid_record.items() if gt_uuid not in gt_assigned_idx]
    fp_match = [match_obj(None, pred_instant, 0, gt_ts=gt_ts)
                for pred_uuid, pred_instant in pred_uuid_record.items() if pred_uuid not in pred_assigned_idx]
    match_pair_list.extend(fn_match)
    match_pair_list.extend(fp_match)
    return match_pair_list


class Matcher:
    def __init__(self, gt_obj, pred_obj, match_method="iou", category=None, process_num=1):
        self.gt_obj = gt_obj
        self.pred_obj = pred_obj
        self.match_method = match_method
        self.category = category
        self.process_num = process_num

    @staticmethod
    def get_ts_match(target_ts_lists):
        try:
            return ts_match_func(target_ts_lists, verbose=True)
        except ValueError:
            return token_match(target_ts_lists, verbose=True)

    @staticmethod
    def order_index_to_uuid(gt_instants, pred_instants, match_index_list):
        match_uuid_list = []
        for gt_idx, pred_idx in zip(*match_index_list):
            match_uuid_list.append((gt_instants[gt_idx].get_uuid(), pred_instants[pred_idx].get_uuid()))
        return match_uuid_list

    @staticmethod
    def index_matrix_to_uuid(gt_instants, pred_instants, matrix):
        uuid_record = dict()
        for gt_idx in range(matrix.shape[0]):
            for pred_idx in range(matrix.shape[1]):
                uuid_record[(gt_instants[gt_idx].get_uuid(), pred_instants[pred_idx].get_uuid())] = matrix[gt_idx, pred_idx]
        return uuid_record

    def create_match_pairs(self, match_info_list, match_obj):
        match_pair_list = []
        for frame_match_info in match_info_list:
            match_pair_list.extend(self.create_frame_match_pairs(frame_match_info, match_obj))
        return match_pair_list

    def create_frame_match_pairs(self, frame_match_info, match_obj, filter_bad=True):
        match_pair_list = []
        gt_assigned_idx = dict()
        pred_assigned_idx = dict()
        gt_frame = None
        pred_frame = None
        gt_uuid_record = None
        pred_uuid_record = None
        for match_status, (gt_ts, pred_ts), match_factor_record, match_uuid_list in frame_match_info:
            if gt_frame is None and pred_frame is None:
                gt_frame = self.gt_obj.get_frame_obj_by_ts(gt_ts)
                pred_frame = self.pred_obj.get_frame_obj_by_ts(pred_ts)
                gt_uuid_record = {instant.get_uuid(): instant for instant in gt_frame.get_instant_objects()}
                pred_uuid_record = {instant.get_uuid(): instant for instant in pred_frame.get_instant_objects()}
            if match_status != FrameMatchStatus.normal:
                continue
            for gt_uuid, pred_uuid in match_uuid_list:
                match_factor = match_factor_record[(gt_uuid, pred_uuid)]
                if filter_bad:
                    if (self.match_method == "iou" and match_factor < 0.1) or \
                            (self.match_method == "distance") and (match_factor > 4):
                        continue
                gt_assigned_idx[gt_uuid] = True
                pred_assigned_idx[pred_uuid] = True
                match_pair_list.append(match_obj(gt_uuid_record[gt_uuid],
                                                 pred_uuid_record[pred_uuid],
                                                 match_factor))
        fn_match = [match_obj(gt_instant, None, 0)
                    for gt_uuid, gt_instant in gt_uuid_record.items() if gt_uuid not in gt_assigned_idx]
        fp_match = [match_obj(None, pred_instant, 0)
                    for pred_uuid, pred_instant in pred_uuid_record.items() if
                    pred_uuid not in pred_assigned_idx]
        match_pair_list.extend(fn_match)
        match_pair_list.extend(fp_match)
        return match_pair_list

    def init_objects_by_ts(self, matched_ts_list):
        for gt_ts, pred_ts in matched_ts_list:
            self.gt_obj.get_frame_obj_by_ts(gt_ts).get_instant_objects()
            self.pred_obj.get_frame_obj_by_ts(pred_ts).get_instant_objects()

    def get_frame_match_info(self, matched_ts):
        if self.match_method == "iou":
            return self.match_frame_by_iou(matched_ts)
        elif self.match_method == "iop":
            return self.match_frame_by_iop(matched_ts)
        elif self.match_method == "distance":
            return self.match_frame_by_distance(matched_ts)
        elif self.match_method == "LET_iou":
            return self.match_frame_by_LET_iou(matched_ts)
        else:
            raise NotImplementedError("match method of {} was not implemented".format(self.match_method))

    def match_frame_by_iop(self, matched_ts):
        match_info_list = []
        gt_ts, pred_ts = matched_ts
        gt_frame = self.gt_obj.get_frame_obj_by_ts(gt_ts)
        pred_frame = self.pred_obj.get_frame_obj_by_ts(pred_ts)
        if not gt_frame.is_empty():
        # if not gt_frame.is_empty() and not pred_frame.is_empty():
            gt_instants = [instant for instant in gt_frame.get_instant_objects()
                           if instant.get_category() in self.category]
            pred_instants = [instant for instant in pred_frame.get_instant_objects()]
            iop_matrix = get_iop_matrix([instant.get_polygon() for instant in gt_instants],
                                        [instant.get_polygon() for instant in pred_instants])
            match_idx_gt = []
            match_idx_pred = []
            for pred_idx, gt_row_iop in enumerate(iop_matrix.transpose()):
                gt_idx = np.argmax(gt_row_iop)
                if gt_row_iop[gt_idx] > 0:
                    match_idx_gt.append(gt_idx)
                    match_idx_pred.append(pred_idx)
            match_index_list = (match_idx_gt, match_idx_pred)
            match_uuid_list = self.order_index_to_uuid(gt_instants, pred_instants, match_index_list)
            uuid_iop_record = self.index_matrix_to_uuid(gt_instants, pred_instants, iop_matrix)
            match_info_list.append([FrameMatchStatus.normal, matched_ts, uuid_iop_record, match_uuid_list])
        elif not gt_frame.is_empty():
            match_uuid_list = [(instant.get_uuid(), None) for instant in gt_frame.get_instant_objects()]
            match_info_list.append([FrameMatchStatus.fn, matched_ts, [], match_uuid_list])
        elif not pred_frame.is_empty():
            # entire frame fp
            match_uuid_list = [(None, instant.get_uuid()) for instant in pred_frame.get_instant_objects()]
            match_info_list.append([FrameMatchStatus.fp, matched_ts, [], match_uuid_list])
        return match_info_list

    def match_frame_by_LET_iou(self, matched_ts):
        match_info_list = []
        gt_ts, pred_ts = matched_ts
        gt_frame = self.gt_obj.get_frame_obj_by_ts(gt_ts)
        pred_frame = self.pred_obj.get_frame_obj_by_ts(pred_ts)
        if not gt_frame.is_empty() and not pred_frame.is_empty():
            gt_instants = [instant for instant in gt_frame.get_instant_objects()
                           if instant.get_category() in self.category]
            pred_instants = [instant for instant in pred_frame.get_instant_objects()
                             if instant.get_category() in self.category]
            iou_matrix = get_LET_iou_matrix(gt_instants, pred_instants)
            if len(iou_matrix) > 0:
                match_index_list = get_munkres_match(iou_matrix)
                match_uuid_list = self.order_index_to_uuid(gt_instants, pred_instants, match_index_list)
                uuid_iou_record = self.index_matrix_to_uuid(gt_instants, pred_instants, iou_matrix)
                match_info_list.append([FrameMatchStatus.normal, matched_ts, uuid_iou_record, match_uuid_list])
        elif not gt_frame.is_empty():
            # entire frame fn
            match_uuid_list = [(instant.get_uuid(), None) for instant in gt_frame.get_instant_objects()]
            match_info_list.append([FrameMatchStatus.fn, matched_ts, [], match_uuid_list])
        elif not pred_frame.is_empty():
            # entire frame fp
            match_uuid_list = [(None, instant.get_uuid()) for instant in pred_frame.get_instant_objects()]
            match_info_list.append([FrameMatchStatus.fp, matched_ts, [], match_uuid_list])
        return match_info_list

    def match_frame_by_iou(self, matched_ts):
        match_info_list = []
        gt_ts, pred_ts = matched_ts
        if gt_ts == 1690943523.400138:
            print("-")
        gt_frame = self.gt_obj.get_frame_obj_by_ts(gt_ts)
        pred_frame = self.pred_obj.get_frame_obj_by_ts(pred_ts)
        if not gt_frame.is_empty() and not pred_frame.is_empty():
            gt_instants = [instant for instant in gt_frame.get_instant_objects()
                           if instant.get_category() in self.category]
            pred_instants = [instant for instant in pred_frame.get_instant_objects()
                             if instant.get_category() in self.category]
            iou_matrix = get_iou_matrix([instant.get_polygon() for instant in gt_instants],
                                        [instant.get_polygon() for instant in pred_instants])
            if len(iou_matrix) > 0:
                match_index_list = get_munkres_match(iou_matrix)
                # use uuid to record instant level match pair info
                match_uuid_list = self.order_index_to_uuid(gt_instants, pred_instants, match_index_list)
                # use (gt_uuid, pred_uuid)  as key to record value of match_retio
                uuid_iou_record = self.index_matrix_to_uuid(gt_instants, pred_instants, iou_matrix)
                match_info_list.append([FrameMatchStatus.normal, matched_ts, uuid_iou_record, match_uuid_list])
        elif not gt_frame.is_empty():
            # entire frame fn
            match_uuid_list = [(instant.get_uuid(), None) for instant in gt_frame.get_instant_objects()]
            match_info_list.append([FrameMatchStatus.fn, matched_ts, [], match_uuid_list])
        elif not pred_frame.is_empty():
            # entire frame fp
            match_uuid_list = [(None, instant.get_uuid()) for instant in pred_frame.get_instant_objects()]
            match_info_list.append([FrameMatchStatus.fp, matched_ts, [], match_uuid_list])
        return match_info_list

    def match_frame_by_distance(self, matched_ts):
        if self.category is None:
            raise AttributeError("list of target category is required for match by distance")
        match_info_list = []
        gt_ts, pred_ts = matched_ts
        gt_frame = self.gt_obj.get_frame_obj_by_ts(gt_ts)
        pred_frame = self.pred_obj.get_frame_obj_by_ts(pred_ts)
        if not gt_frame.is_empty() and not pred_frame.is_empty():

            import os
            import simplejson as json
            gt_out_objs = [instant.get_visual_json_obj() for instant in gt_frame.get_instant_objects()]
            pred_out_objs = [instant.get_visual_json_obj() for instant in pred_frame.get_instant_objects()]
            with open(os.path.join("/mnt/data/visual_perception/clip_155/visual_perception/label/gt", "{:.6f}.json".format(gt_ts)),
                      'w') as f:
                json.dump(gt_out_objs, f)
            with open(os.path.join("/mnt/data/visual_perception/clip_155/visual_perception/label/detection",
                                   "{:.6f}.json".format(gt_ts)), 'w') as f:
                json.dump(pred_out_objs, f)
            src_lidar_path = os.path.join("/mnt/data/lidar_detection/test_datasets/20230717_clip155/pcd",
                                          "{:.6f}.pcd".format(gt_ts))
            dst_lidar_path = os.path.join("/mnt/data/visual_perception/clip_155/visual_perception/lidar",
                                          "{:.6f}.pcd".format(gt_ts))
            os.symlink(src_lidar_path, dst_lidar_path)

            for target_category in self.category:
                gt_instants = gt_frame.get_instant_objects_by_category(target_category)
                pred_instants = pred_frame.get_instant_objects_by_category(target_category)
                if len(gt_instants) > 0 and len(pred_instants) > 0:
                    dist_matrix = get_dist_matrix([instant.get_center_2d() for instant in gt_instants],
                                                  [instant.get_center_2d() for instant in pred_instants])
                    # cate_match_index_list = linear_sum_assignment(dist_matrix)
                    cate_match_index_list = [list(range(len(dist_matrix))), np.argmin(dist_matrix, axis=1)]
                    dist_record = self.index_matrix_to_uuid(gt_instants, pred_instants, dist_matrix)
                    cate_match_uuid_list = self.order_index_to_uuid(gt_instants, pred_instants, cate_match_index_list)
                    match_info_list.append([FrameMatchStatus.normal, matched_ts, dist_record, cate_match_uuid_list])
                elif len(gt_instants) > 0:
                    cate_match_uuid_list = [(instant.get_uuid(), None) for instant in gt_instants]
                    match_info_list.append([FrameMatchStatus.fn, matched_ts, [], cate_match_uuid_list])
                elif len(pred_instants) > 0:
                    cate_match_uuid_list = [(None, instant.get_uuid()) for instant in pred_instants]
                    match_info_list.append([FrameMatchStatus.fp, matched_ts, [], cate_match_uuid_list])
        return match_info_list

    def get_match_info(self, matched_ts_list, child_conn=None):
        match_info_list = []
        for matched_ts in matched_ts_list:
            frame_match_info = self.get_frame_match_info(matched_ts)
            match_info_list.append(frame_match_info)
        if child_conn is not None:
            child_conn.send(match_info_list)
        else:
            return match_info_list

    def get_match_info_parallel(self, matched_ts_list, processes=8):
        return multiprocess_execute(self.get_match_info, matched_ts_list, processes)

    @timeit
    def match(self, threshold=200):
        """
        match by iou or distance on each frame
        Args:
            threshold: use multiprocess if the input frame number exceed the threshold value

        Returns:
            [[frame 1 match info], [frame 2 match info], ...], where frame match info is [[match info], [match info]]
        """
        matched_ts_list = self.get_ts_match([self.gt_obj.get_ts_list(), self.pred_obj.get_ts_list()])
        self.init_objects_by_ts(matched_ts_list)
        if self.process_num > 1:
            match_info_list = self.get_match_info_parallel(matched_ts_list, processes=self.process_num)
        else:
            match_info_list = self.get_match_info(matched_ts_list)
        match_info_list = [frame_match for frame_match in match_info_list if len(frame_match) > 0]
        return match_info_list
