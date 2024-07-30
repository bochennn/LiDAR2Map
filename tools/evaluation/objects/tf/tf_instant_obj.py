import numpy as np
from pyquaternion import Quaternion

from objects.base_objs.base_instant_obj import InstantBase
from objects.tf.attribute_name import Attr


import math


def quaternion_to_euler(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return yaw, pitch, roll


def roty(t):
    c = np.cos(t)
    s = np.sin(t)

    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


class TfInstantObj(InstantBase):
    def __init__(self, data, ts):
        super().__init__(data, ts)
        self.quaternion = None

    def get_ts(self):
        return self.ts

    def get_qw(self):
        return self.getattr(Attr.rotation_qw)

    def get_qx(self):
        return self.getattr(Attr.rotation_qx)

    def get_qy(self):
        return self.getattr(Attr.rotation_qy)

    def get_qz(self):
        return self.getattr(Attr.rotation_qz)

    def get_quaternion(self):
        return [self.get_qw(), self.get_qx(), self.get_qy(), self.get_qz()]

    def get_translation_x(self):
        return self.getattr(Attr.translation_x)

    def get_translation_y(self):
        return self.getattr(Attr.translation_y)

    def get_translation_z(self):
        return self.getattr(Attr.translation_z)

    def get_translation(self):
        return np.array([self.get_translation_x(),
                         self.get_translation_y(),
                         self.get_translation_z()]).reshape((3, 1))

    def get_yaw_pitch_roll(self):
        if self.quaternion is None:
            self.quaternion = Quaternion(self.get_quaternion())
        return self.quaternion.yaw_pitch_roll

    def get_yaw(self):
        return self.get_yaw_pitch_roll()[0]

    def get_pitch(self):
        return self.get_yaw_pitch_roll()[1]

    def get_roll(self):
        return self.get_yaw_pitch_roll()[2]

    def get_rotation_matrix(self):
        if not isinstance(self.get_qw(), float):
            print(self.get_ts())
        yaw, _, _ = quaternion_to_euler(self.get_qw(), self.get_qx(), self.get_qy(), self.get_qz())
        return roty(yaw)
        #
        # if self.quaternion is None:
        #     self.quaternion = Quaternion(self.get_quaternion())
        # return self.quaternion.rotation_matrix

    def get_transform_matrix(self):
        return np.vstack([np.hstack([self.get_rotation_matrix(), self.get_translation()]),
                          [0, 0, 0, 1]])

    def loc_to_world(self):
        return self.get_transform_matrix()

    def world_to_loc(self):
        return np.linalg.inv(self.loc_to_world())
