import math
import sys

import numpy as np

from utils.transform import rotation_matrix


class MotionMeasurement:
    def __init__(self, use_adaptive=True):
        self.use_adaptive = use_adaptive

    @staticmethod
    def projection_vector(projected_vector, project_vector):
        # to be verify if calculation is correct and equal to Calculate2DXYProjectVector
        project_vector /= np.linalg.norm(projected_vector)
        scalar_projection = np.dot(projected_vector, project_vector)
        return scalar_projection * project_vector

    @staticmethod
    def cal_bbox_size_center_2d(corners_3d, yaw):
        """
        calculate center and size based on 8 points in corners_3d and yaw in radians
        Args:
            corners_3d: 8 points of rotated bbox
            yaw: radians along z-axis

        Returns:
            size and center of bbox along given direction of yaw
        """
        minimum_edge_length = 0.01
        rotation = rotation_matrix(yaw)
        min_pt = np.array([sys.float_info.max, sys.float_info.max, sys.float_info.max])
        max_pt = np.array([-sys.float_info.max, sys.float_info.max, sys.float_info.max])
        for point in corners_3d:
            loc_point = rotation.dot(np.array(point))

            min_pt[0] = min(min_pt[0], loc_point[0])
            min_pt[1] = min(min_pt[1], loc_point[1])
            min_pt[2] = min(min_pt[2], loc_point[2])

            max_pt[0] = max(max_pt[0], loc_point[0])
            max_pt[1] = max(max_pt[1], loc_point[1])
            max_pt[2] = max(max_pt[2], loc_point[2])
        size = max_pt - min_pt
        coeff = (max_pt + min_pt) / 2
        coeff[2] = min_pt[2]
        center = rotation.dot(coeff)
        size[0] = max(minimum_edge_length, size[0])
        size[1] = max(minimum_edge_length, size[1])
        size[2] = max(minimum_edge_length, size[2])
        return center, size

    @staticmethod
    def measure_anchor_point_velocity(last_obj, crt_obj, time_diff):
        anchor_point_velocity = (crt_obj.get_barycenter_utm() - last_obj.get_barycenter_utm()) / time_diff
        return anchor_point_velocity

    def measure_boxcenter_velocity(self, last_obj, crt_obj, time_diff):
        """
        calculate velocity by last bbox and bbox generated from current corners and last direction
        """
        cal_center, cal_size = self.cal_bbox_size_center_2d(crt_obj.get_corners3d_utm(),
                                                            last_obj.get_utm_yaw())
        box_center_vel_with_old_dir = cal_center - last_obj.get_center_utm() / time_diff
        box_center_vel_with_old_dir[2] = 0.0
        project_dir = crt_obj.get_center_utm() - last_obj.get_center_utm()
        if box_center_vel_with_old_dir.dot(project_dir) <= 0:
            box_center_vel_with_old_dir = np.array([0.0, 0.0, 0.0])
        return box_center_vel_with_old_dir

    def measure_bbox_corner_velocity(self, last_obj, crt_obj, time_diff):
        cal_center, cal_size = self.cal_bbox_size_center_2d(crt_obj.get_corners3d_utm(),
                                                            last_obj.get_utm_yaw())
        old_dir = np.array([np.cos(last_obj.get_utm_yaw()),
                            np.sin(last_obj.get_utm_yaw()),
                            0])
        ortho_old_dir = np.array([old_dir[1], old_dir[0], 0])
        old_box_corners = np.array(
            [
                last_obj.get_center_utm() + old_dir * last_obj.get_length() * 0.5 + ortho_old_dir * last_obj.get_width() * 0.5,
                last_obj.get_center_utm() - old_dir * last_obj.get_length() * 0.5 + ortho_old_dir * last_obj.get_width() * 0.5,
                last_obj.get_center_utm() + old_dir * last_obj.get_length() * 0.5 - ortho_old_dir * last_obj.get_width() * 0.5,
                last_obj.get_center_utm() - old_dir * last_obj.get_length() * 0.5 - ortho_old_dir * last_obj.get_width() * 0.5,
            ]
        )
        new_box_corners = np.array(
            [
                cal_center + old_dir * cal_size[0] * 0.5 + ortho_old_dir * cal_size[1] * 0.5,
                cal_center - old_dir * cal_size[0] * 0.5 + ortho_old_dir * cal_size[1] * 0.5,
                cal_center + old_dir * cal_size[0] * 0.5 - ortho_old_dir * cal_size[1] * 0.5,
                cal_center - old_dir * cal_size[0] * 0.5 - ortho_old_dir * cal_size[1] * 0.5,
            ]
        )

        ref_location = crt_obj.get_lidar_to_world().translation
        new_corner_distance_list = np.linalg.norm(new_box_corners - ref_location, axis=1)
        nearest_corner_idx = np.argmin(new_corner_distance_list)
        nearest_old_box_corner = old_box_corners[nearest_corner_idx]
        nearest_new_box_corner = new_box_corners[nearest_corner_idx]

        measured_nearest_corner_velocity = (nearest_new_box_corner - nearest_old_box_corner) / time_diff

        length_change = abs(last_obj.get_length() - crt_obj.get_length()) / last_obj.get_length()
        width_change = abs(last_obj.get_width() - crt_obj.get_width()) / last_obj.get_width()
        max_change_thresh = 0.1
        if length_change < max_change_thresh and width_change < max_change_thresh:
            project_dir = crt_obj.get_center() - last_obj.get_center()
        else:
            project_dir = crt_obj.get_anchor_point() - last_obj.get_anchor_point

        measured_corners_velocity = []
        for idx, (new_corner, old_corner) in enumerate(zip(new_box_corners, old_box_corners)):
            corner_velocity = (new_corner - old_corner) / time_diff
            corner_velocity_on_project_dir = self.projection_vector(corner_velocity, project_dir)
            if np.dot(corner_velocity_on_project_dir, project_dir) <= 0:
                corner_velocity_on_project_dir = np.array([0, 0, 0])
            measured_corners_velocity.append(corner_velocity_on_project_dir)
        return measured_nearest_corner_velocity, measured_corners_velocity

    @staticmethod
    def measurement_selection(last_obj, crt_obj):
        corner_velocity_gain_norms = np.linalg.norm(crt_obj.measured_corners_velocity() - last_obj.belief_velocity(),
                                                    axis=1)
        corner_velocity_gain = min(corner_velocity_gain_norms)
        anchor_point_velocity_gain = np.linalg.norm(crt_obj.measured_barycenter_velocity() - last_obj.belief_velocity())
        boxcenter_velocity_gain = crt_obj.measured_center_velocity - last_obj.belief_velocity()
        velocity_gain_record = (
            (corner_velocity_gain, crt_obj.measured_center_velocity()),
            (anchor_point_velocity_gain, crt_obj.measured_barycenter_velocity()),
            (boxcenter_velocity_gain, crt_obj.measured_center_velocity)
        )
        crt_obj.selected_measured_velocity = min(velocity_gain_record, key=lambda x: x[0])[1]

    def measurement_quality_estimation(self, last_obj, crt_obj):
        # change to last_obj.get_points_inside_bbox()
        last_points_num = len(last_obj.get_corners_3d())
        crt_points_num = len(last_obj.get_corners_3d())
        quality_based_on_point_diff_ratio = \
            1 - abs(last_points_num - crt_points_num) / max(last_points_num, crt_points_num)
        quality_based_on_association_score = pow(1 - crt_obj.get_association_score(), 2)
        crt_obj.update_quality = min(quality_based_on_point_diff_ratio, quality_based_on_association_score)
        if not self.use_adaptive:
            crt_obj.update_quality = 1

    def compute_motion_measurement(self, last_obj, crt_obj):
        time_diff = crt_obj.get_ts() - last_obj.get_ts()
        self.measure_anchor_point_velocity(last_obj, crt_obj, time_diff)
        self.measure_boxcenter_velocity(last_obj, crt_obj, time_diff)
        self.measure_bbox_corner_velocity(last_obj, crt_obj, time_diff)
        self.measurement_selection(last_obj, crt_obj)



