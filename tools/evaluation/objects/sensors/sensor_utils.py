import enum


class SensorName:
    IMU = "IMU"
    lidar_main = "lidar_main"
    lidar_front = "lidar_front"
    lidar_left = "lidar_left"
    lidar_right = "lidar_right"
    local = "local"
    camera_front_main = "camera_front_main"
    camera_front_wide = "camera_front_wide"
    camera_front_left = "camera_front_left"
    camera_front_right = "camera_front_right"
    camera_rear_main = "camera_rear_main"
    camera_rear_left = "camera_rear_left"
    camera_rear_right = "camera_rear_right"
    radar_front = "radar_front"
    radar_front_left = "radar_front_left"
    radar_front_right = "radar_front_right"
    radar_rear = "radar_rear"
    radar_rear_left = "radar_rear_left"
    radar_rear_right = "radar_rear_right"


sensor_name_map = {"novatel": SensorName.IMU,
                   "velodyne64": SensorName.lidar_main,
                   "qt128_front": SensorName.lidar_front,
                   "qt128_left": SensorName.lidar_left,
                   "qt128_right": SensorName.lidar_right,
                   "localization": SensorName.local,
                   "camera_front_main": SensorName.camera_front_main,
                   "camera_front_wide": SensorName.camera_front_wide,
                   "camera_left_front": SensorName.camera_front_left,
                   "camera_right_front": SensorName.camera_front_right,
                   "camera_rear_main": SensorName.camera_rear_main,
                   "camera_left_rear": SensorName.camera_rear_left,
                   "camera_right_rear": SensorName.camera_rear_right,
                   "radar_front": SensorName.radar_front,
                   "radar_front_left": SensorName.radar_front_left,
                   "radar_front_right": SensorName.radar_front_right,
                   "radar_rear": SensorName.radar_rear,
                   "radar_rear_left": SensorName.radar_rear_left,
                   "radar_rear_right": SensorName.radar_rear_right}

