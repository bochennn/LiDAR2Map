import math
from decimal import Decimal

import cv2
import numpy as np
import open3d as o3d
# import simplejson as json
from shapely.geometry import Polygon

from ....log_mgr import logger
from ....utils.bbox_ops import get_3d_corners, get_quaternion
from ....utils.plot import text_o3d
from ....utils.transform import rotation_matrix, scalar_transform, vector_transform
from ...base_objs.base_instant_obj import InstantBase
from ..parsers.attribute_tool import Attr, ObstacleEnum

kEpsilon = 1e-6


class ObstacleInstantObj(InstantBase):
    def __init__(self, data, ts):
        super().__init__(data, ts)
        self.lidar_yaw_degree = None

    def get_bbox_3d(self):
        return [self.get_lidar_pos_x(),
                self.get_lidar_pos_y(),
                self.get_lidar_pos_z(),
                self.get_length(),
                self.get_width(),
                self.get_height(),
                self.get_lidar_yaw()]

    def get_length(self) -> float:
        return self.getattr(Attr.length)

    def get_width(self) -> float:
        return self.getattr(Attr.width)

    def get_height(self) -> float:
        return self.getattr(Attr.height)

    def get_lidar_yaw(self) -> float:
        return self.getattr(Attr.lidar_yaw)

    def get_utm_yaw(self):
        return self.getattr(Attr.utm_yaw)

    def get_obj_size(self):
        return [self.get_length(), self.get_width(), self.get_height()]

    def get_obj_center(self):
        return [self.get_lidar_pos_x(), self.get_lidar_pos_y(), self.get_lidar_pos_z()]

    def get_center_2d(self):
        return [self.get_lidar_pos_x(), self.get_lidar_pos_y()]

    def get_center_3d(self):
        return [self.get_lidar_pos_x(), self.get_lidar_pos_y(), self.get_lidar_pos_z()]

    def get_center_2d_utm(self):
        return [self.get_utm_pos_x(), self.get_utm_pos_y()]

    def get_center_3d_utm(self):
        return [self.get_utm_pos_x(), self.get_utm_pos_y(), self.get_utm_pos_z()]

    def get_lidar_barycenter(self):
        bary_center = np.zeros((3,))
        corners = self.get_corners_3d()
        for point in corners:
            bary_center += point
        return bary_center / len(corners)

    def get_utm_barycenter(self):
        bary_center = np.zeros((3,))
        corners = self.get_utm_corners_3d()
        for point in corners:
            bary_center += point
        return bary_center / len(corners)

    def get_center_2d_lidar(self):
        return [self.get_lidar_pos_x(), self.get_lidar_pos_y()]

    def get_lidar_pos_x(self) -> float:
        return self.getattr(Attr.lidar_pos_x)

    def get_lidar_pos_y(self) -> float:
        return self.getattr(Attr.lidar_pos_y)

    def get_lidar_pos_z(self) -> float:
        return self.getattr(Attr.lidar_pos_z)

    def get_utm_pos_x(self) -> float:
        return self.getattr(Attr.utm_pos_x)

    def get_utm_pos_y(self) -> float:
        return self.getattr(Attr.utm_pos_y)

    def get_utm_pos_z(self) -> float:
        return self.getattr(Attr.utm_pos_z)

    def get_corners_3d(self):
        return self.getattr(Attr.corners_3d)

    def get_corners_2d(self):
        return self.getattr(Attr.corners_2d)

    def get_surfaces(self):
        corners = self.get_corners_3d()
        face_x_p = corners[[0, 1, 4, 5]]
        face_x_n = corners[[2, 3, 6, 7]]
        face_y_p = corners[[0, 3, 4, 7]]
        face_y_n = corners[[1, 2, 5, 6]]
        face_z_p = corners[[0, 1, 2, 3]]
        face_z_n = corners[[4, 5, 6, 7]]
        return face_x_p, face_x_n, face_y_p, face_y_n, face_z_p, face_z_n

    def get_utm_corners_3d(self):
        required_attrs = [Attr.utm_yaw, Attr.utm_pos_x, Attr.utm_pos_y, Attr.utm_pos_z]
        for attr in required_attrs:
            if not self.hasattr(attr) or self.getattr(attr) is None:
                raise AttributeError("{} is required to calculate utm corners".format(attr))
        rot = rotation_matrix(self.get_utm_yaw())
        return get_3d_corners(self.get_utm_pos_x(),
                              self.get_utm_pos_y(),
                              self.get_utm_pos_z(),
                              self.get_length(),
                              self.get_width(),
                              self.get_height(),
                              rot)

    def get_utm_corners_2d(self):
        return self.get_utm_corners_3d()[:4, :2]

    def get_distance_to_ego(self):
        return round(np.linalg.norm(self.get_center_2d_lidar()), 2)

    def get_polygon(self) -> Polygon:
        if not self.hasattr(Attr.polygon) or self.getattr(Attr.polygon) is None:
            return Polygon(self.get_corners_2d())
        else:
            return self.getattr(Attr.polygon)

    def get_aligned_polygon(self, target_center):
        target_x = target_center[0]
        target_y = target_center[1]
        quaternion = get_quaternion(self.get_lidar_yaw())
        self_x, self_y, self_z = self.get_center_3d()

        align_multiplier = (target_x * self_x + target_y * self_y) / max((self_x ** 2 + self_y ** 2), kEpsilon)
        aligned_x = self_x * align_multiplier
        aligned_y = self_y * align_multiplier

        corners_3d = get_3d_corners(aligned_x,
                                    aligned_y,
                                    self_z,
                                    self.get_length(),
                                    self.get_width(),
                                    self.get_height(),
                                    quaternion.rotation_matrix)
        corners_2d = corners_3d[:4, :2]
        return Polygon(corners_2d)

    def get_longitudinal_affine(self, target_center):
        def clamp(value, low, high):
            return max(low, min(value, high))

        longitudinal_tolerance_percentage = 0.1
        min_longitudinal_tolerance_meter = 0.5

        target_x = target_center[0]
        target_y = target_center[1]
        self_x, self_y, self_z = self.get_center_3d()
        target_dot_self = target_x * self_x + target_y * self_y
        target_range = max(np.linalg.norm([target_x, target_y]), kEpsilon)
        self_range = max(np.linalg.norm([self_x, self_y]), kEpsilon)
        cos_theta = target_dot_self / target_range / self_range
        max_range_tolerance_meter = max(target_range * longitudinal_tolerance_percentage,
                                        min_longitudinal_tolerance_meter)
        range_error = abs((self_range * cos_theta - target_range) / max_range_tolerance_meter)
        return 0 if range_error >= 1 else 1
        # return clamp(1 - range_error, 0.0, 1.0)

    def get_score(self) -> float:
        return self.getattr(Attr.score)

    def get_category(self) -> str:
        return self.getattr(Attr.category)

    def get_type_id(self):
        return self.getattr(Attr.type_id)

    def get_sub_type_id(self):
        return self.getattr(Attr.subtype_id)

    def get_obj_id(self) -> int:
        return self.getattr(Attr.object_id)

    def get_track_id(self) -> str:
        return self.getattr(Attr.track_id)

    def get_distance(self) -> float:
        return math.sqrt(pow(self.get_lidar_pos_x(), 2) + pow(self.get_lidar_pos_y(), 2))

    def get_lidar_yaw_degree(self) -> float:
        if self.lidar_yaw_degree is None:
            self.lidar_yaw_degree = math.degrees(self.get_lidar_yaw()) if self.get_lidar_yaw() is not None else None
        return self.lidar_yaw_degree

    def get_utm_yaw_degree(self):
        return math.degrees(self.get_utm_yaw()) if self.get_utm_yaw() is not None else None

    def get_frame_seq(self) -> int:
        return self.getattr(Attr.frame_seq)

    def get_num_points(self):
        return self.getattr(Attr.num_points)

    # only valid when data source is zdrive format annotation
    def get_clip_id(self):
        return self.getattr(Attr.clip_id)

    # only valid when data source is zdrive format annotation
    def get_frame_id(self):
        return self.getattr(Attr.frame_id)

    # 3x3 dimension ndarray
    def get_rotation_matrix(self) -> np.ndarray:
        return rotation_matrix(self.get_lidar_yaw())

    def get_num_lidar_pts(self):
        return self.getattr(Attr.num_lidar_pts)

    def get_visual_json_obj_with_velocity(self):
        ts_str = "{:6f}".format(self.get_ts())
        visual_obj = {"measure_timestamp": Decimal(ts_str),
                      "track_id": self.get_obj_id(),
                      "obj_id": self.get_obj_id(),
                      "obj_type": self.get_type_id(),
                      "obj_sub_type": ObstacleEnum.invert_fusion_subtype_enum[self.get_category()],
                      "category": self.get_category(),
                      "obj_score": self.get_score() if self.get_score() is not None else 1,
                      "lidar_velocity": {
                          "x": self.getattr("refined_lidar_abs_vel_x"),
                          "y": self.getattr("refined_lidar_abs_vel_y"),
                          "z": 0.0
                      },
                      "psr": {"position": {"x": self.get_lidar_pos_x(),
                                           "y": self.get_lidar_pos_y(),
                                           "z": self.get_lidar_pos_z()},
                              "scale": {"x": self.get_length(),
                                        "y": self.get_width(),
                                        "z": self.get_height()},
                              "rotation": {"x": 0,
                                           "y": 0,
                                           "z": self.get_lidar_yaw()}},
                      "utm_position": {"x": self.get_utm_pos_x(),
                                       "y": self.get_utm_pos_y(),
                                       "z": self.get_utm_pos_z()},
                      "utm_yaw": self.get_utm_yaw(),
                      "utm_velocity": {"x": self.getattr("refined_utm_abs_vel_x"),
                                       "y": self.getattr("refined_utm_abs_vel_y"),
                                       "z": 0.0},
                      "forward_utm_velocity": {"x": self.getattr("forward_utm_abs_vel_x"),
                                               "y": self.getattr("forward_utm_abs_vel_y"),
                                               "z": 0.0},
                      "backward_utm_velocity": {"x": self.getattr("backward_utm_abs_vel_x"),
                                                "y": self.getattr("backward_utm_abs_vel_y"),
                                                "z": 0.0}}
        return visual_obj

    def get_visual_json_obj(self, obj_attr=None):
        ts_str = "{}".format(self.get_ts())
        try:
            visual_obj = {
                          "track_id": self.get_track_id(),
                          "obj_id": self.get_obj_id(),
                          "obj_type": self.get_type_id(),
                          "obj_sub_type": ObstacleEnum.invert_fusion_subtype_enum[self.get_category()],
                          "category": self.get_category(),
                          "obj_score": self.get_score() if self.get_score() is not None else 1,
                          "psr": {"position": {"x": self.get_lidar_pos_x(),
                                               "y": self.get_lidar_pos_y(),
                                               "z": self.get_lidar_pos_z()},
                                  "scale": {"x": self.get_length(),
                                            "y": self.get_width(),
                                            "z": self.get_height()},
                                  "rotation": {"x": 0,
                                               "y": 0,
                                               "z": self.get_lidar_yaw()}}}
            if self.hasattr(Attr.utm_pos_x):
                visual_obj.update(
                    {
                        "utm_position": {
                            "x": self.get_utm_pos_x(),
                            "y": self.get_utm_pos_y(),
                            "z": self.get_utm_pos_z()
                        },
                        "utm_yaw": self.get_utm_yaw(),
                    }
                )
            if self.hasattr(Attr.utm_abs_vel_x):
                visual_obj.update(
                    {
                        "utm_velocity": {
                            "x": self.get_utm_abs_vel_x(),
                            "y": self.get_utm_abs_vel_y(),
                            "z": 0
                        }
                    }
                )
        except Exception as e:
            logger.error(e)
            import traceback
            logger.error(traceback.format_exc())
            logger.error(self.data)

        if self.hasattr(Attr.num_lidar_pts):
            visual_obj.update({"num_lidar_pts": self.get_num_lidar_pts()})
        if obj_attr is not None:
            visual_obj["obj_attr"] = obj_attr
        return visual_obj

    def get_o3d_patch(self, color=None) -> o3d.geometry.OrientedBoundingBox:
        color = color if color is not None else [1, 0, 0]
        o3d_points = o3d.utility.Vector3dVector(self.get_corners_3d())
        try:
            o3d_patch = o3d.geometry.OrientedBoundingBox.create_from_points(o3d_points)
            o3d_patch.color = color
            return o3d_patch
        except Exception as e:
            logger.error("track_id: {}, error: {}".format(self.get_track_id(), e))
            import traceback
            logger.error(traceback.format_exc())

    def get_o3d_heading_patch(self, color=None):
        color = color if color is not None else [1, 0, 0]
        line = np.array([[0, 0, 0],
                         [self.get_length(), 0, 0]]).T
        line = self.get_rotation_matrix().dot(line).T
        line[:, 0] += self.get_lidar_pos_x()
        line[:, 1] += self.get_lidar_pos_y()
        line[:, 2] += self.get_lidar_pos_z()
        o3d_patch = o3d.geometry.LineSet()
        o3d_patch.points = o3d.utility.Vector3dVector(line)
        o3d_patch.lines = o3d.utility.Vector2iVector([[0, 1]])
        o3d_patch.colors = o3d.utility.Vector3dVector([color])
        return o3d_patch

    def get_o3d_text_patch(self):
        text = "track_id: {}, cate: {}, yaw: {}".format(self.get_obj_id(),
                                                        self.get_category(),
                                                        round(self.get_lidar_yaw_degree(), 2))
        # text = "track_id: {}, cate: {}".format(self.get_obj_id(),
        #                                        self.get_category())
        # text = " velocity: {}".format(round(np.linalg.norm([self.get_utm_abs_vel_x(),
        #                                                self.get_utm_abs_vel_y()]), 2))
        text = "score: {}, cate: {}".format(round(self.get_score(), 2), self.get_category())
        return text_o3d(text, self.get_corners_3d()[0])

    def get_o3d_points_volume(self):
        corners = np.array(self.get_corners_3d(), dtype=np.float64)
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Y"
        vol.axis_min = np.min(corners[:, 1])
        vol.axis_max = np.max(corners[:, 1])
        corners[:, 1] = 0
        vol.bounding_polygon = o3d.utility.Vector3dVector(corners)
        return vol

    def get_calculate_utm_velocity(self):
        if not self.hasattr("calculate_velocity"):
            return np.zeros((3, ))
        return self.getattr("calculate_velocity")

    def get_utm_abs_vel_x(self) -> float:
        return self.getattr(Attr.utm_abs_vel_x)

    def get_utm_abs_vel_y(self) -> float:
        return self.getattr(Attr.utm_abs_vel_y)

    def get_utm_acc_x(self):
        return self.getattr(Attr.abs_acc_x)

    def get_utm_acc_y(self):
        return self.getattr(Attr.abs_acc_y)

    def get_lidar_abs_vel_x(self):
        return self.getattr(Attr.lidar_abs_vel_x)

    def get_lidar_abs_vel_y(self):
        return self.getattr(Attr.lidar_abs_vel_y)

    def get_lidar_linear_vel(self):
        return np.linalg.norm([self.get_lidar_abs_vel_x(),
                               self.get_lidar_abs_vel_y()])

    def get_utm_linear_vel(self):
        return np.linalg.norm([self.get_utm_abs_vel_x(),
                               self.get_utm_abs_vel_y()])

    def get_utm_rel_vel_x(self):
        return self.getattr(Attr.utm_rel_vel_x)

    def get_utm_rel_vel_y(self):
        return self.getattr(Attr.utm_rel_vel_y)

    def get_utm_rel_vel_z(self):
        return self.getattr(Attr.utm_rel_vel_z)

    def get_lidar_rel_vel_x(self):
        return self.getattr(Attr.lidar_rel_vel_x)

    def get_lidar_rel_vel_y(self):
        return self.getattr(Attr.lidar_rel_vel_y)

    def get_lidar_rel_vel_z(self):
        return self.getattr(Attr.lidar_rel_vel_z)

    def cal_utm_position(self, lidar2world):
        if not self.hasattr(Attr.utm_pos_x):
            velodyne_point = [self.get_lidar_pos_x(),
                              self.get_lidar_pos_y(),
                              self.get_lidar_pos_z()]
            utm_point = scalar_transform(velodyne_point, lidar2world)
            self.setattr(Attr.utm_pos_x, utm_point[0])
            self.setattr(Attr.utm_pos_y, utm_point[1])
            self.setattr(Attr.utm_pos_z, utm_point[2])

    def cal_lidar_position(self, world2lidar):
        if not self.hasattr(Attr.lidar_pos_x):
            utm_point = [self.get_utm_pos_x(),
                         self.get_utm_pos_y(),
                         self.get_utm_pos_z()]
            velodyne_point = scalar_transform(utm_point, world2lidar)
            self.setattr(Attr.lidar_pos_x, velodyne_point[0])
            self.setattr(Attr.lidar_pos_y, velodyne_point[1])
            self.setattr(Attr.lidar_pos_z, velodyne_point[2])

    def cal_utm_abs_velocity(self, lidar2world):
        if not self.hasattr(Attr.utm_abs_vel_x):
            lidar_abs_velocity = [self.get_lidar_abs_vel_x(),
                                  self.get_lidar_abs_vel_y()]
            utm_abs_velocity = vector_transform(lidar_abs_velocity, lidar2world)
            self.setattr(Attr.utm_abs_vel_x, utm_abs_velocity[0])
            self.setattr(Attr.utm_abs_vel_y, utm_abs_velocity[1])
            self.setattr(Attr.utm_abs_vel_z, 0)

    def cal_lidar_abs_velocity(self, world2lidar, source_ego=None):
        if not self.hasattr(Attr.lidar_abs_vel_x) and self.hasattr(Attr.utm_abs_vel_x):
            utm_abs_velocity = [self.get_utm_abs_vel_x(),
                                self.get_utm_abs_vel_y()]
            lidar_abs_velocity = vector_transform(utm_abs_velocity, world2lidar)
            self.setattr(Attr.lidar_abs_vel_x, lidar_abs_velocity[0])
            self.setattr(Attr.lidar_abs_vel_y, lidar_abs_velocity[1])
            self.setattr(Attr.lidar_abs_vel_z, 0)
        elif self.hasattr(Attr.lidar_rel_vel_x) and source_ego is not None:
            source_lidar_abs_vel_x = source_ego.get_lidar_abs_vel_x()
            source_lidar_abs_vel_y = source_ego.get_lidar_abs_vel_y()
            lidar_abs_vel_x = self.get_lidar_rel_vel_x() + source_lidar_abs_vel_x
            lidar_abs_vel_y = self.get_lidar_rel_vel_y() + source_lidar_abs_vel_y
            self.setattr(Attr.lidar_abs_vel_x, lidar_abs_vel_x)
            self.setattr(Attr.lidar_abs_vel_y, lidar_abs_vel_y)

    def cal_utm_yaw(self, lidar2world):
        if not self.hasattr(Attr.utm_yaw):
            lidar_yaw = self.get_lidar_yaw()
            lidar_direction = [math.cos(lidar_yaw), math.sin(lidar_yaw), 0]
            utm_direction = vector_transform(lidar_direction, lidar2world)
            self.setattr(Attr.lidar_yaw, math.atan2(utm_direction[1], utm_direction[0]))

    def cal_lidar_yaw(self, world2lidar):
        if not self.hasattr(Attr.lidar_yaw):
            utm_yaw = self.get_utm_yaw()
            utm_direction = [math.cos(utm_yaw), math.sin(utm_yaw), 0]
            lidar_direction = vector_transform(utm_direction, world2lidar)
            self.setattr(Attr.lidar_yaw, math.atan2(lidar_direction[1], lidar_direction[0]))

    def cal_utm_pose(self, lidar2world):
        self.cal_utm_position(lidar2world)
        self.cal_utm_abs_velocity(lidar2world)
        self.cal_utm_yaw(lidar2world)

    def cal_lidar_pose(self, world2lidar, source_ego=None):
        self.cal_lidar_position(world2lidar)
        self.cal_lidar_abs_velocity(world2lidar, source_ego)
        self.cal_lidar_yaw(world2lidar)

    def is_key_target(self):
        return self.getattr(Attr.is_key_obj)

    def get_2d_box_patch(self, ax, color):
        box = self.get_corners_2d()
        ax.plot([p[0] for p in box], [p[1] for p in box], color=color)

    def get_img_xmin(self):
        return self.getattr(Attr.img_xmin)

    def get_img_xmax(self):
        return self.getattr(Attr.img_xmax)

    def get_img_ymin(self):
        return self.getattr(Attr.img_ymin)

    def get_img_ymax(self):
        return self.getattr(Attr.img_ymax)

    def get_img_width(self):
        return self.getattr(Attr.img_width)

    def get_img_height(self):
        return self.getattr(Attr.img_height)

    def get_img_bbox(self):
        return [self.get_img_xmin(),
                self.get_img_ymin(),
                self.get_img_xmax(),
                self.get_img_ymax()]

    def render_cv(self, img_mat, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        p1 = (self.get_img_xmin(), self.get_img_ymin())
        p2 = (self.get_img_xmax(), self.get_img_ymax())
        cv2.rectangle(img_mat, p1, p2, color, 2, cv2.LINE_AA)
        cv2.putText(img_mat, self.get_category(), p1, font, 1, color, 2)
