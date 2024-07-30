import os
import simplejson as json
import yaml

# import quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
from cmath import sin
from tqdm import tqdm
from pypcd import pypcd
# import open3d as o3d
# from cyber_record.record import Record
# from record_msg.parser import PointCloudParser


HEIGHT = 2.03


class PointCompensator:
    def __init__(self, clip_path, mc_time=30):
        self.clip_path = clip_path
        self.mc_time = mc_time
        self.localization_info = self.get_localization_info()
        self.lidar2imu = self.get_lidar2imu_info()

    def get_localization_info(self):
        file_path = os.path.join(self.clip_path, "localization.json")
        with open(file_path, 'r') as f:
            info = json.load(f)
        return info

    def get_lidar2imu_info(self):
        file_path = os.path.join(self.clip_path, "extrinsics", "lidar2imu", "lidar2imu.yaml")
        with open(file_path, 'r') as f:
            file = yaml.safe_load(f)
            q_x = file["transform"]["rotation"]["x"]
            q_y = file["transform"]["rotation"]["y"]
            q_z = file["transform"]["rotation"]["z"]
            q_w = file["transform"]["rotation"]["w"]
            t_x = file["transform"]["translation"]["x"]
            t_y = file["transform"]["translation"]["y"]
            t_z = file["transform"]["translation"]["z"]
            r = R.from_quat(np.array([q_x, q_y, q_z, q_w]))
            rotation = r.as_matrix()
            t = np.array([t_x, t_y, t_z])
            lidar2imu = np.zeros((4, 4))
            lidar2imu[:3, :3] = rotation
            lidar2imu[:3, 3] = t
            lidar2imu[3, 3] = 1.0
        return lidar2imu
        # hesai2velodyne = np.zeros((4, 4))
        # hesai2velodyne[0, 1] = -1
        # hesai2velodyne[1, 0] = 1
        # hesai2velodyne[2, 2] = 1
        # hesai2velodyne[3, 3] = 1
        # hesailidar2imu = lidar2imu @ hesai2velodyne
        # return hesailidar2imu

    @staticmethod
    def msg_to_points(lidar_msg):
        points_num = len(lidar_msg.point)
        np_x = np.zeros(points_num, dtype=np.float32)
        np_y = np.zeros(points_num, dtype=np.float32)
        np_z = np.zeros(points_num, dtype=np.float32)
        np_i = np.zeros(points_num)
        np_t = np.zeros(points_num, dtype=np.float64)
        for idx, point in enumerate(lidar_msg.point):
            np_x[idx] = point.x
            np_y[idx] = point.y
            np_z[idx] = point.z
            np_i[idx] = point.intensity
            np_t[idx] = point.timestamp / 1e9
        return np.array(np.transpose(np.vstack([np_x, np_y, np_z, np_i, np_t])))

    def get_pose_by_ts(self, ts):
        pose_use = None
        for pose in self.localization_info:
            pose_time = pose["timestamp"] * 1000
            if np.abs(ts - pose_time) < 5:
                qx = pose["pose"]["orientation"]["qx"]
                qy = pose["pose"]["orientation"]["qy"]
                qz = pose["pose"]["orientation"]["qz"]
                qw = pose["pose"]["orientation"]["qw"]
                r_ = R.from_quat(np.array([qx, qy, qz, qw]))
                rotation_r = r_.as_matrix()
                x = pose["pose"]["position"]["x"]
                y = pose["pose"]["position"]["y"]
                z = pose["pose"]["position"]["z"]
                T = np.eye(4)
                T[:3, :3] = rotation_r
                T[:3, 3] = np.array([x, y, z])
                T_lidar2world = self.lidar2imu @ T
                r_scipy = R.from_matrix(T_lidar2world[:3, :3])
                q = r_scipy.as_quat()
                pose_use = [ts, q[3], q[0], q[1], q[2], x, y, z]
                break
        return pose_use

    def get_pose_by_ts_new(self, ts):
        pose_use = None
        pose_ts_list = [(idx, pose["timestamp"] * 1000) for idx, pose in enumerate(self.localization_info)]
        target_idx = min(pose_ts_list, key=lambda x: abs(x[1] - ts))[0]
        pose = self.localization_info[target_idx]
        qx = pose["pose"]["orientation"]["qx"]
        qy = pose["pose"]["orientation"]["qy"]
        qz = pose["pose"]["orientation"]["qz"]
        qw = pose["pose"]["orientation"]["qw"]
        r_ = R.from_quat(np.array([qx, qy, qz, qw]))
        rotation_r = r_.as_matrix()
        x = pose["pose"]["position"]["x"]
        y = pose["pose"]["position"]["y"]
        z = pose["pose"]["position"]["z"]
        T = np.eye(4)
        T[:3, :3] = rotation_r
        T[:3, 3] = np.array([x, y, z])
        T_lidar2world = self.lidar2imu @ T
        r_scipy = R.from_matrix(T_lidar2world[:3, :3])
        q = r_scipy.as_quat()
        pose_use = [ts, q[3], q[0], q[1], q[2], x, y, z]
        return pose_use

    def point2q(self, point):
        w_ = 0.0
        if isinstance(point, np.ndarray):
            x_ = point[0]
            y_ = point[1]
            z_ = point[2]
        else:
            x_ = point.x
            y_ = point.y
            z_ = point.z
        return np.quaternion(w_, x_, y_, z_)

    def conjugate_q(self, q):  # w, x, y, z
        w_ = q.w
        x_ = -q.x
        y_ = -q.y
        z_ = -q.z
        return np.quaternion(w_, x_, y_, z_)

    def motion_comp_static(self, point_q, point_t, start_pose, mc_pose, end_pose):
        start_time = start_pose[0]
        start_q = np.quaternion(start_pose[1], start_pose[2], start_pose[3], start_pose[4])  ##w,x,y,z
        start_t = np.array(start_pose[5:])
        mid_time = mc_pose[0]
        mid_q = np.quaternion(mc_pose[1], mc_pose[2], mc_pose[3], mc_pose[4])  ##w,x,y,z
        mid_t = np.array(mc_pose[5:])
        end_time = end_pose[0]
        end_q = np.quaternion(end_pose[1], end_pose[2], end_pose[3], end_pose[4])  ##w,x,y,z
        end_t = np.array(end_pose[5:])
        obj_t = point_t
        if start_time - 5.0 <= obj_t < mid_time:
            translation = start_t - mid_t
            translation_q = self.point2q(translation)
            mid_q_conjugate = self.conjugate_q(mid_q)
            q_start2mid = mid_q_conjugate * start_q
            translation_mid_cord_ = mid_q_conjugate * translation_q
            translation_mid_cord = translation_mid_cord_ * mid_q
            q1 = q_start2mid
            q0 = np.quaternion(1.0, 0.0, 0.0, 0.0)
            d = np.dot(np.array([q0.w, q0.x, q0.y, q0.z]), np.array([q1.w, q1.x, q1.y, q1.z]).T)
            f = 1 / (mid_time - start_time)
            t_nor = (mid_time - obj_t) * f
        elif mid_time <= obj_t <= end_time + 5.0:
            translation = end_t - mid_t
            translation_q = self.point2q(translation)
            mid_q_conjugate = self.conjugate_q(mid_q)
            q_end2mid = mid_q_conjugate * end_q
            translation_mid_cord_ = mid_q_conjugate * translation_q
            translation_mid_cord = translation_mid_cord_ * mid_q
            q1 = q_end2mid
            q0 = np.quaternion(1.0, 0.0, 0.0, 0.0)
            d = np.dot(np.array([q0.w, q0.x, q0.y, q0.z]), np.array([q1.w, q1.x, q1.y, q1.z]).T)
            f = 1 / (end_time - mid_time)
            t_nor = (obj_t - mid_time) * f
        else:
            return None
        if t_nor < 0:
            return None
        # if t_nor >= 0:
        if abs(d) < 1.0 - 1.0e-8:
            theta = np.arccos(abs(d))
            sin_theta = np.sin(theta)
            c1_sign = 1 if (d > 0) else -1
            translation_nor = t_nor * np.array(
                [translation_mid_cord.x, translation_mid_cord.y, translation_mid_cord.z])
            c0 = sin((1 - t_nor) * theta) / sin_theta
            c1 = sin(t_nor * theta) / sin_theta * c1_sign
            q_nor = quaternion.as_quat_array(
                c1 * np.array([q1.w, q1.x, q1.y, q1.z]) + c0 * np.array([q0.w, q0.x, q0.y, q0.z]))
            q_nor_conjugate = self.conjugate_q(q_nor)
            point2mid_ = q_nor * point_q
            point2mid = point2mid_ * q_nor_conjugate
            point2mid = np.array([point2mid.x, point2mid.y, point2mid.z]) + translation_nor
        else:
            translation_nor = t_nor * np.array(
                [translation_mid_cord.x, translation_mid_cord.y, translation_mid_cord.z])
            point2mid = np.array([point_q.x, point_q.y, point_q.z]) + translation_nor
        return point2mid

    @staticmethod
    def lidar_msg_update(lidar_msg, mc_points):
        assert len(lidar_msg.point) == len(mc_points)
        for ori_point, mc_point in zip(lidar_msg.point, mc_points):
            ori_point.x = mc_point[0]
            ori_point.y = mc_point[1]
            ori_point.z = mc_point[2]
        return lidar_msg

    def compensate(self, lidar_msg, logger=None):
        print_func = logger.warning if logger is not None else print
        all_timestamps = [point.timestamp for point in lidar_msg.point]
        min_timestamp = min(all_timestamps) / 1e6
        max_timestamp = max(all_timestamps) / 1e6
        mc_time = min_timestamp + self.mc_time
        assert (max_timestamp - min_timestamp) < 105
        start_pose = self.get_pose_by_ts(min_timestamp)
        end_pose = self.get_pose_by_ts(max_timestamp)
        mc_pose = self.get_pose_by_ts(mc_time)
        # assert start_pose is not None and end_pose is not None and mc_pose is not None
        if start_pose is None or end_pose is None or mc_pose is None:
            print_func("msg {} skipped".format(lidar_msg.measurement_time))
            return lidar_msg
        points_pc = []
        for point in lidar_msg.point:
            point_time = point.timestamp / 1e6
            q_point = self.point2q(point)
            point_with_mc = self.motion_comp_static(q_point, point_time, start_pose, mc_pose, end_pose)
            assert point_with_mc is not None
            # point[:3] = point_with_mc
            points_pc.append(list(point_with_mc) + [point.intensity])
        lidar_msg = self.lidar_msg_update(lidar_msg, points_pc)
        return lidar_msg


def points_crop(points, x_range, y_range, z_range):
    mask = (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) & \
           (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1]) & \
           (points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1])
    return points[mask]


def points_crop_oriented(points, x_range, y_range, z_range, rot_matrix):
    rotated_points = points.dot(rot_matrix.T)
    return points_crop(rotated_points, x_range, y_range, z_range)


def load_pcd(addr, offset=(0, 0, 0)):
    pc = pypcd.PointCloud.from_path(addr)
    x = pc.pc_data['x'].reshape(-1, 1) + offset[0]
    y = pc.pc_data['y'].reshape(-1, 1) + offset[1]
    z = pc.pc_data['z'].reshape(-1, 1) + offset[2]
    arr = np.hstack([x, y, z])
    return arr


def load_bin(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    return scan.reshape(-1, 4)[:, [0, 1, 2]]


def numpy_to_bin(data_array, out_path):
    with open(out_path, 'w') as f:
        data_array.tofile(f)


def convert_xyzit_pb_to_array(points_array, data_type):
    arr = np.zeros(len(points_array), dtype=data_type)
    point_dim = len(points_array[0])
    for i, point in enumerate(points_array):
        arr[i] = (point[0], point[1], point[2], point[3], point[4]/1e9) if point_dim == 5 \
            else (point[0], point[1], point[2], point[3])
    return arr


def make_xyzit_point_cloud(points_array):
    """
    Make a pointcloud object from PointXYZIT message, as Pointcloud.proto.
    message PointXYZIT {
      optional float x = 1 [default = nan];
      optional float y = 2 [default = nan];
      optional float z = 3 [default = nan];
      optional uint32 intensity = 4 [default = 0];
      optional uint64 timestamp = 5 [default = 0];
    }
    """
    fields_map = {4: ["x", "y", "z", "intensity"],
                  5: ["x", "y", "z", "intensity", "timestamp"]}
    count_map = {4: [1, 1, 1, 1],
                 5: [1, 1, 1, 1, 1]}
    type_map = {4: ["F", "F", "F", "U"],
                5: ["F", "F", "F", "U", "F"]}
    size_map = {4: [4, 4, 4, 4],
                5: [4, 4, 4, 4, 8]}
    point_dim = len(points_array[0])
    md = {'version': .7,
          'fields': fields_map[point_dim],
          'count': count_map[point_dim],
          'width': len(points_array),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(points_array),
          'type': type_map[point_dim],
          'size': size_map[point_dim],
          'data': 'binary_compressed'}
    typenames = []
    for t, s in zip(md['type'], md['size']):
        np_type = pypcd.pcd_type_to_numpy_type[(t, s)]
        typenames.append(np_type)
    np_dtype = np.dtype(list(zip(md['fields'], typenames)))
    pc_data = convert_xyzit_pb_to_array(points_array, data_type=np_dtype)
    pc = pypcd.PointCloud(md, pc_data)
    return pc


def numpy_to_pcd(data_array, out_path):
    from record_msg import pypcd as ppcd
    pc_meta = make_xyzit_point_cloud(data_array)
    ppcd.save_point_cloud(pc_meta, out_path)


def bin_to_pcd(bin_file_path, out_path, dim=4):
    file_name = os.path.splitext(os.path.basename(bin_file_path))[0]
    out_pcd_path = os.path.join(out_path, "{}.pcd".format(file_name))
    data_type = np.float32 if dim == 4 else np.float64
    points_array = np.fromfile(bin_file_path, dtype=data_type)
    points_array = points_array.reshape(-1, dim)
    numpy_to_pcd(points_array, out_pcd_path)


def record_to_pcd(record_path, out_path, topic="/apollo/sensor/main/fusion/PointCloud2", clip_path=None):
    compensator = None
    if clip_path is not None:
        compensator = PointCompensator(clip_path)
    with Record(record_path, 'r') as record:
        parser = PointCloudParser(out_path)
        for topic, message, _ in tqdm(record.read_messages(topic)):
            measurement_ts = str(message.header.lidar_timestamp)[:16]
            measurement_ts = measurement_ts[:10] + "." + measurement_ts[10:]
            if compensator is not None:
                message = compensator.compensate(message)
            parser.parse(message, measurement_ts, mode='binary')


def record_to_bin(record_path, out_path, topic="/apollo/sensor/main/fusion/PointCloud2", for_infer=False):
    with Record(record_path, 'r') as record:
        for topic, message, _ in tqdm(record.read_messages(topic)):
            out_file = os.path.join(out_path, "{:.6f}.bin".format(int(message.measurement_time * 1e6) / 1e6))
            pc_data = np.array([[point.x, point.y, point.z, point.intensity] for point in message.point],
                               dtype=np.float32)
            if for_infer:
                pc_data[:, 2] = pc_data[:, 2] + HEIGHT
                pc_data[:, 3] = pc_data[:, 3] / 255.0
            numpy_to_bin(pc_data, out_file)


def pcd_to_bin(src_file, out_path):
    pc = pypcd.PointCloud.from_path(src_file)
    np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32) + HEIGHT
    np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32) / 255.0
    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    file_name = os.path.basename(src_file)
    file_name = os.path.splitext(file_name)[0]
    file_name = "{:.6f}".format(float(file_name.split("_")[1]) / 1e6)
    bin_path = os.path.join(out_path, "{}.bin".format(file_name))
    with open(bin_path, 'w') as f:
        points_32.tofile(f)
    return bin_path


def project_lidar_to_img(lidar_pts, cam_extrinsic, cam_intrinsic, img_w, img_h):
    x, y, z, end = 0, 1, 2, 3
    lidar_pts_raw = lidar_pts[:, :end]
    lidar_pts_pad = np.concatenate([lidar_pts_raw, np.ones_like(lidar_pts_raw[:, [x]])], axis=-1)
    lidar_pts_cam = np.matmul(lidar_pts_pad, cam_extrinsic.T)
    target_depth_mask = lidar_pts_cam[:, z] > 0
    lidar_pts_img = np.matmul(lidar_pts_cam, cam_intrinsic.T)
    lidar_pts_img[:, [x, y]] = lidar_pts_img[:, [x, y]] / lidar_pts_img[:, [z]]
    # lidar_pts_img = lidar_pts_img.astype(np.int32)[:, [x, y]]

    target_range_mask = (lidar_pts_img[:, x] >= 0) & (lidar_pts_img[:, x] < img_w) & \
                        (lidar_pts_img[:, y] >= 0) & (lidar_pts_img[:, y] < img_h)
    mask = target_depth_mask & target_range_mask
    return lidar_pts_raw[mask], lidar_pts_img[mask], mask


def clip_pcd_process(root_data_path, clip_name, dataset_pcd_path, topic):
    print("{} start".format(clip_name))
    clip_path = os.path.join(root_data_path, clip_name)
    clip_pcd_out_path = os.path.join(root_data_path, clip_name, "pcd_mc")
    os.makedirs(clip_pcd_out_path, exist_ok=True)
    record_folder = os.path.join(root_data_path, clip_name, "bag")
    record_name = os.listdir(record_folder)[0]
    record_file_path = os.path.join(record_folder, record_name)
    record_to_pcd(record_file_path, clip_pcd_out_path, topic=topic, clip_path=clip_path)
    for name in os.listdir(clip_pcd_out_path):
        if name.endswith(".pcd"):
            src_pcd_file = os.path.join(clip_pcd_out_path, name)
            out_pcd_file = os.path.join(dataset_pcd_path, name)
            os.symlink(src_pcd_file, out_pcd_file)


if __name__ == "__main__":
    # topic = "/apollo/sensor/rsm2/PointCloud2"
    # input_file_path = "/home/wuchuanpan/Projects/mars_orin/m2_test_data/20231208093519.record.00021"
    # out_path = "/mnt/data/tmp/zark_lidar_tracking/m2_test_message_verify/m2_test_pose_verify/lidar"
    # record_to_pcd(input_file_path, out_path, topic)

    root_data_path = "/mnt/data/lidar_detection/results/zark_lidar_tracking/2024_0328.00000"
    out_path = "/mnt/data/lidar_detection/results/zark_lidar_tracking/2024_0328_0/lidar"
    topic = "/apollo/sensor/rsm2/PointCloud2"
    record_to_pcd(root_data_path, out_path, topic)


