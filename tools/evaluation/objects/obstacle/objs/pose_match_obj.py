from ....utils.distance import points_distance
from .obstacle_match_obj import ObstacleMatchObj


class PoseMatchObj:
    def __init__(self, gt_instant, pred_instants):
        self.gt_instant = gt_instant
        self.pred_instants = pred_instants
        self.dist_threshold = 3
        self.vel_threshold = 3

    @staticmethod
    def get_delta_velocity(gt_instant, pred_instant):
        delta = None
        if gt_instant.get_utm_abs_vel_x() is not None and pred_instant.get_utm_abs_vel_x() is not None:
            delta = points_distance([gt_instant.get_utm_abs_vel_x(), gt_instant.get_utm_abs_vel_y()],
                                    [pred_instant.get_utm_abs_vel_x(), pred_instant.get_utm_abs_vel_y()])
        elif gt_instant.get_lidar_abs_vel_x() is not None and pred_instant.get_lidar_abs_vel_x() is not None:
            delta = points_distance([gt_instant.get_lidar_abs_vel_x(), gt_instant.get_lidar_abs_vel_y()],
                                    [pred_instant.get_lidar_abs_vel_x(), pred_instant.get_lidar_abs_vel_y()])
        elif pred_instant.get_lidar_rel_vel_x() is not None:
            delta = points_distance([0, 0],
                                    [pred_instant.get_lidar_rel_vel_x(), pred_instant.get_lidar_rel_vel_y()])
        return delta

    def create_obstacle_match(self, target_pred_name):
        pred_instant = None
        iou = 0
        for name, instant, distance in self.pred_instants:

            if name == target_pred_name and \
                    distance < self.dist_threshold and \
                    self.get_delta_velocity(self.gt_instant, instant) < self.vel_threshold:
                pred_instant = instant
                iou = 1
        return ObstacleMatchObj(self.gt_instant, pred_instant, iou=iou)
