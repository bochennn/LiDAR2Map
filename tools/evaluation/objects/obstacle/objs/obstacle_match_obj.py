import math

import open3d as o3d
from shapely.geometry import Polygon, Point

from objects.base_objs.base_match_obj import BaseMatch


class ObstacleMatchObj(BaseMatch):
    def __init__(self,
                 gt_instant,
                 pred_instant,
                 match_factor,
                 gt_ts=None,
                 threshold_map=None,
                 match_method="iou",
                 global_range=None):
        super().__init__(gt_instant, pred_instant, match_factor)
        self.gt_instant = gt_instant
        self.pred_instant = pred_instant
        self.match_factor = match_factor
        self.gt_ts = gt_ts
        self.threshold_map = threshold_map
        self.match_method = match_method
        self.global_range = global_range
        self.iou = self.match_factor if self.match_method in ["iou", "LET_iou"] else None
        self.iop = self.match_factor if self.match_method == 'iop' else None
        self.distance = self.match_factor if self.match_method == "distance" else None
        self.tp_flag = None
        self.fp_flag = None
        self.fn_flag = None
        self.distance = None
        self.lon_error = None
        self.lat_error = None

    def is_pred_target_category(self, target_category):
        return self.get_pred_category() == target_category

    def is_gt_target_category(self, target_category):
        return self.get_gt_category() == target_category

    def within_distance_region(self, distance):
        if self.global_range is not None and not self.global_range.contains(Point(self.get_center_rel())):
            return False
        if isinstance(distance, Polygon):
            return distance.contains(Point(self.get_center_rel()))
        elif len(distance) == 2:  # lower and upper distance
            return distance[0] < self.get_distance() < min(120, distance[1])
        elif len(distance) == 0:  # no distance constraint
            return True

    def child_attr_wrapper(self, target_child, attr_access_func):
        instant = self.gt_instant if target_child == "gt" else self.pred_instant
        if instant is not None:
            return attr_access_func(instant)
        else:
            return None

    def get_ts(self):
        return self.gt_instant.get_ts() if self.gt_valid() else self.pred_instant.get_ts()

    def get_gt_track_id(self):
        return self.child_attr_wrapper("gt", lambda x: x.get_track_id())

    def get_pred_track_id(self):
        return self.child_attr_wrapper("pred", lambda x: x.get_track_id())

    def get_category(self):
        if self.gt_valid():
            return self.get_gt_category()
        else:
            return self.get_pred_category()

    def get_gt_category(self):
        return self.child_attr_wrapper("gt", lambda x: x.get_category())

    def get_pred_category(self):
        return self.child_attr_wrapper("pred", lambda x: x.get_category())

    def get_distance(self):
        if self.distance is None:
            self.distance = self.gt_instant.get_distance_to_ego() if self.gt_valid() \
                else self.pred_instant.get_distance_to_ego()
        return self.distance

    def get_distance_error(self):
        if not self.match_valid():
            return None
        else:
            return abs(self.gt_instant.get_distance_to_ego() - self.pred_instant.get_distance_to_ego())

    def get_lidar_pos_x(self):
        return self.gt_instant.get_lidar_pos_x() if self.gt_valid() else self.pred_instant.get_lidar_pos_x()

    def get_lidar_pos_y(self):
        return self.gt_instant.get_lidar_pos_y() if self.gt_valid() else self.pred_instant.get_lidar_pos_y()

    def get_center_rel(self):
        if self.gt_valid():
            return self.gt_instant.get_center_2d_lidar()
        elif self.pred_instant:
            return self.pred_instant.get_center_2d_lidar()

    def get_score(self):
        score = 0.0
        if self.pred_valid():
            score = self.pred_instant.get_score()
        return score

    def get_iou_threshold(self):
        if self.gt_valid():
            return self.threshold_map["iou"][self.get_gt_category()]
        else:
            return None

    def get_lat_error_thrs(self):
        if self.gt_valid():
            return self.threshold_map["distance"][self.get_gt_category()]["lat"]
        else:
            return None

    def get_lon_error_thrs(self):
        if self.gt_valid():
            return self.threshold_map["distance"][self.get_gt_category()]["lon"]
        else:
            return None

    def get_iop_threshold(self):
        return self.threshold_map["iop_threshold"][self.get_gt_category()]

    def get_cluster_dist_threshold(self):
        if not self.match_valid():
            return None
        else:
            return max(self.threshold_map["min_dist_thrs"],
                       self.gt_instant.get_distance_to_ego() * self.threshold_map["dist_percentage"])

    def is_iop_match(self):
        print("iop: {}, thrs: {}".format(self.iop, self.get_iop_threshold()))
        return self.iop >= self.get_iop_threshold()

    def is_cluster_dist_match(self):
        return True
        # print("distance error: {}, thrs: {}".format(self.get_distance_error(), self.get_cluster_dist_threshold()))
        return self.get_distance_error() < self.get_cluster_dist_threshold()

    def is_iou_match(self):
        return self.iou >= self.get_iou_threshold()

    def is_dist_match(self):
        return self.get_lat_error() < self.get_lat_error_thrs() \
            and self.get_lon_error() < self.get_lon_error_thrs()

    def is_category_match(self):
        return self.match_valid() and (self.gt_instant.get_category() == self.pred_instant.get_category())

    def is_tp(self):
        if self.tp_flag is None:
            if self.match_method == "iou":
                self.tp_flag = self.match_valid() and self.is_iou_match() and self.is_category_match()
            elif self.match_method == "iop":
                self.tp_flag = self.match_valid() and self.is_iop_match() and self.is_cluster_dist_match()
            elif self.match_method == "distance":
                self.tp_flag = self.match_valid() and self.is_dist_match() and self.is_category_match()
            elif self.match_method == "LET_iou":
                self.tp_flag = self.match_valid() and self.is_iou_match() and self.is_category_match()
        return self.tp_flag

    def is_fp(self):
        if self.fp_flag is None:
            self.fp_flag = not self.is_tp() and self.pred_valid()
        return self.fp_flag

    def is_iou_fp_only(self):
        return self.match_valid() and not self.is_iou_match() and self.is_category_match()

    def is_category_fp_only(self):
        return self.match_valid() and not self.is_category_match() and self.is_iou_match()

    def is_iou_category_fp(self):
        return self.match_valid() and not self.is_iou_match() and not self.is_category_match()

    def is_extra_fp(self):
        return not self.gt_valid()

    def is_fn(self):
        if self.fn_flag is None:
            self.fn_flag = not self.is_tp() and self.gt_valid()
        return self.fn_flag

    def get_yaw_delta(self):
        if self.match_valid():
            return self.gt_instant.get_lidar_yaw() - self.pred_instant.get_lidar_yaw()
        else:
            return None

    def get_yaw_degree_delta(self):
        if self.match_valid():
            delta = self.gt_instant.get_lidar_yaw_degree() - self.pred_instant.get_lidar_yaw_degree()
            delta = 360 - delta if delta > 180 else delta
            return abs(delta)
        else:
            return None

    def get_lat_error(self):
        if self.match_valid():
            return abs(self.gt_instant.get_lidar_pos_y() - self.pred_instant.get_lidar_pos_y())
        else:
            return None

    def get_lon_error(self):
        if self.match_valid():
            return abs(self.gt_instant.get_lidar_pos_x() - self.pred_instant.get_lidar_pos_x())
        else:
            return None

    def get_yaw_similarity(self):
        yaw_delta = self.get_yaw_delta()
        similarity = (1.0 + math.cos(yaw_delta)) / 2.0 if yaw_delta is not None and self.is_tp() else .0
        return similarity

    def get_tph(self):
        yaw_delta = self.get_yaw_delta()
        tph = 1 - abs(yaw_delta % math.pi) / math.pi if yaw_delta is not None and self.is_tp() else .0
        return tph
