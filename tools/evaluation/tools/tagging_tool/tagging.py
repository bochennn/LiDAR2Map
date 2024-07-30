from enum import Enum
# from objects.obstacle.objs.obstacle_clip_pred import ObstacleClipPred


class Tags(Enum):
    class Direction(Enum):
        same_as_ego = "same_as_ego"
        oncoming = "oncoming"
        left_crossing = "left_crossing"
        right_crossing = "right_crossing"

    class Dynamic:
        moving = "moving"
        stationary = "stationary"

    class LateralPosition:
        same_lane = "same_lane"


class DirectionTag:
    @staticmethod
    def direction(instant_obj):
        same_as_ego_bound = (-45, 45)
        oncoming_bound = (-135, 135)
        left_crossing_bound = (45, 135)
        right_crossing_bound = (-135, -45)
        yaw_degree = instant_obj.get_lidar_yaw_degree()
        if same_as_ego_bound[0] <= yaw_degree <= same_as_ego_bound[1]:
            return Tags.Direction.same_as_ego
        elif oncoming_bound[0] <= yaw_degree <= oncoming_bound[1]:
            return Tags.Direction.oncoming
        elif left_crossing_bound[0] < yaw_degree < left_crossing_bound[1]:
            return Tags.Direction.left_crossing
        elif right_crossing_bound[0] < yaw_degree < right_crossing_bound[1]:
            return Tags.Direction.right_crossing
        else:
            raise ValueError("value of heading in degree should within [-180, 180]")

    @staticmethod
    def dynamics(instant_obj):
        vel_threshold = 1
        obj_vel = instant_obj.get_utm_linear_vel()
        if obj_vel >= vel_threshold:
            return Tags.Dynamic.moving
        else:
            return Tags.Dynamic.stationary

    @staticmethod
    def lateral_position(instant_obj, lanes=None):
        def lateral_position_without_lane(_instant_obj):
            _lateral_position = _instant_obj.get_lidar_pos_y()
            if instant_obj.direction() in [Tags.Direction.same_as_ego, Tags.Direction.oncoming]:
                _proj_length = instant_obj.get_width()
            else:
                _proj_length = instant_obj.get_length()
            if abs(_lateral_position) < (_proj_length / 2):
                return

        if lanes is None:
            return

    @staticmethod
    def longitudinal_position(instant_obj, lanes=None):
        pass

    @staticmethod
    def visibility():
        pass

    @staticmethod
    def appearing():
        pass

    @staticmethod
    def disappearing():
        pass

    @staticmethod
    def following():
        pass

    @staticmethod
    def vehicle_lateral_activity():
        pass

    @staticmethod
    def vehicle_longitudinal_activity():
        pass

    @staticmethod
    def pedestrian_activity():
        pass

    @staticmethod
    def cyclist_lateral_activity():
        pass

    @staticmethod
    def cyclist_longitudinal_activity():
        pass



