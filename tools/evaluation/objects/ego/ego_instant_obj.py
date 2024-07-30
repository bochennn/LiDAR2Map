import math

import open3d as o3d

from objects.base_objs.base_instant_obj import InstantBase
from objects.ego.attribute_name import Attr
from utils.transform import scalar_transform, vector_transform, rotation_matrix
from utils.bbox_ops import get_3d_corners


class EgoInstantObj(InstantBase):
    def __init__(self, data, ts):
        super().__init__(data, ts)

    def get_lat(self) -> float:
        return self.getattr(Attr.lat)

    def get_lon(self) -> float:
        return self.getattr(Attr.lon)

    def get_lidar_pos_x(self) -> float:
        return self.getattr(Attr.lidar_pos_x)

    def get_lidar_pos_y(self) -> float:
        return self.getattr(Attr.lidar_pos_y)

    def get_center_2d_lidar(self):
        return [self.get_lidar_pos_x(), self.get_lidar_pos_y()]

    def get_lidar_pos_z(self) -> float:
        return self.getattr(Attr.lidar_pos_z)

    def get_utm_pos_x(self) -> float:
        return self.getattr(Attr.utm_pos_x)

    def get_utm_pos_y(self) -> float:
        return self.getattr(Attr.utm_pos_y)

    def get_utm_pos_z(self) -> float:
        return self.getattr(Attr.utm_pos_z)

    def get_utm_abs_vel_x(self) -> float:
        return self.getattr(Attr.utm_abs_vel_x)

    def get_utm_abs_vel_y(self) -> float:
        return self.getattr(Attr.utm_abs_vel_y)

    def get_utm_abs_vel_z(self) -> float:
        return self.getattr(Attr.utm_abs_vel_z)

    def get_utm_rel_vel_x(self) -> float:
        return self.getattr(Attr.utm_rel_vel_x)

    def get_utm_rel_vel_y(self) -> float:
        return self.getattr(Attr.utm_rel_vel_y)

    def get_utm_rel_vel_z(self) -> float:
        return self.getattr(Attr.utm_rel_vel_z)

    def get_lidar_abs_vel_x(self) -> float:
        return self.getattr(Attr.lidar_abs_vel_x)

    def get_lidar_abs_vel_y(self) -> float:
        return self.getattr(Attr.lidar_abs_vel_y)

    def get_lidar_abs_vel_z(self) -> float:
        return self.getattr(Attr.lidar_abs_vel_z)

    def get_lidar_rel_vel_x(self) -> float:
        return self.getattr(Attr.lidar_rel_vel_x)

    def get_lidar_rel_vel_y(self) -> float:
        return self.getattr(Attr.lidar_rel_vel_y)

    def get_lidar_rel_vel_z(self) -> float:
        return self.getattr(Attr.lidar_rel_vel_z)

    def get_center_2d_rel(self):
        return [self.get_lidar_pos_x(), self.get_lidar_pos_y()]

    def get_center_2d_utm(self):
        return [self.get_utm_pos_x(), self.get_utm_pos_y()]

    def get_utm_yaw(self):
        return self.getattr(Attr.utm_yaw)

    def get_utm_yaw_degree(self):
        return math.degrees(self.get_utm_yaw()) if self.get_utm_yaw() is not None else None

    def get_lidar_yaw(self):
        return self.getattr(Attr.lidar_yaw)

    def get_lidar_yaw_degree(self):
        return math.degrees(self.get_lidar_yaw()) if self.get_lidar_yaw() is not None else None

    def get_pitch(self):
        return self.getattr(Attr.utm_pitch)

    def get_category(self):
        return self.getattr(Attr.category)

    def get_track_id(self):
        return self.getattr(Attr.track_id)

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

    def cal_lidar_abs_velocity(self, world2lidar):
        if not self.hasattr(Attr.lidar_abs_vel_x) and self.hasattr(Attr.utm_abs_vel_x):
            utm_abs_velocity = [self.get_utm_abs_vel_x(),
                                self.get_utm_abs_vel_y()]
            lidar_abs_velocity = vector_transform(utm_abs_velocity, world2lidar)
            self.setattr(Attr.lidar_abs_vel_x, lidar_abs_velocity[0])
            self.setattr(Attr.lidar_abs_vel_y, lidar_abs_velocity[1])
            self.setattr(Attr.lidar_abs_vel_z, 0)

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

    def cal_lidar_pose(self, world2lidar):
        self.cal_lidar_position(world2lidar)
        self.cal_lidar_abs_velocity(world2lidar)
        self.cal_lidar_yaw(world2lidar)

    def get_lidar_o3d_patch(self, color=None) -> o3d.geometry.OrientedBoundingBox:
        color = color if color is not None else [1, 0, 0]
        bbox_3d = get_3d_corners(self.get_lidar_pos_x(),
                                 self.get_lidar_pos_y(),
                                 self.get_lidar_pos_z(),
                                 1,
                                 1,
                                 1,
                                 rot_matrix=rotation_matrix(self.get_lidar_yaw()))
        o3d_points = o3d.utility.Vector3dVector(bbox_3d)
        o3d_patch = o3d.geometry.OrientedBoundingBox.create_from_points(o3d_points)
        o3d_patch.color = color
        return o3d_patch

