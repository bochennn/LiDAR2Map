import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from ...objects.clip_meta.clip_mgr import ClipMeta
from ...objects.obstacle.objs.obstacle_clip_pred import ObstacleClipPred
from ...utils.bbox_ops import bbox_overlap_filter, get_3d_boxes_in_cam_view, nms_bbox
from ...utils.cluster import cluster_pcd
# from inference.inference import CameraInference, LidarInference
from ...utils.pointcloud_ops import load_pcd, pcd_to_bin, project_lidar_to_img


class LongDistChecker:
    def __init__(self, config):
        self.clip_path = config["clip_path"]
        self.clip_mgr = ClipMeta(self.clip_path)
        self.out_path = self.get_out_path(config["out_path"])
        self.lidar_model_config = config["lidar_model"]
        self.camera_model_config = config["camera_model"]
        self.target_cam = "camera1"
        self.target_distance = 100
        self.target_category = ["car", "truck", "bus"]
        self.lidar_det_obj = self.get_lidar_det_obj()
        self.cam_det_obj = self.get_cam_det_obj()

    def get_out_path(self, root_out_path):
        out_path = os.path.join(root_out_path, self.clip_mgr.clip_name)
        os.makedirs(out_path, exist_ok=True)
        return out_path

    def run_lidar_model_inference(self, out_file):
        lidar_model = LidarInference(self.lidar_model_config)
        sensor_name = "lidar"
        out_pts_path = os.path.join(self.out_path, "xyzi_bin")
        os.makedirs(out_pts_path, exist_ok=True)
        all_frame_id = []
        all_pts_file = []
        for frame_id in self.clip_mgr.frame_info:
            all_frame_id.append(frame_id)
            pcd_path = self.clip_mgr.get_pcd_path(frame_id, sensor_name)
            if pcd_path.endswith(".pcd"):
                all_pts_file.append(pcd_to_bin(pcd_path, out_pts_path))
            else:
                all_pts_file.append(pcd_path)
        all_frame_result = lidar_model.inference(all_pts_file)
        clip_result = dict(zip(all_frame_id, all_frame_result))
        with open(out_file, 'w') as f:
            json.dump(clip_result, f)
        return clip_result

    def run_cam_model_inference(self, out_file):
        camera_model = CameraInference(self.camera_model_config)
        sensor_name = self.target_cam
        all_frame_id = []
        all_img_path = []
        for frame_id in self.clip_mgr.frame_info:
            all_frame_id.append(frame_id)
            all_img_path.append(self.clip_mgr.get_img_path(frame_id, sensor_name))
        all_frame_result = camera_model.inference(all_img_path)
        clip_result = dict(zip(all_frame_id, all_frame_result))
        with open(out_file, 'w') as f:
            json.dump(clip_result, f)
        return clip_result

    def get_lidar_det_obj(self):
        lidar_det_path = os.path.join(self.out_path,  "{}_3d_detection.json".format(self.clip_mgr.clip_name))
        if not os.path.exists(lidar_det_path):
            self.run_lidar_model_inference(out_file=lidar_det_path)
        return ObstacleClipPred(lidar_det_path)

    def get_cam_det_obj(self):
        cam_det_path = os.path.join(self.out_path,  "{}_2d_detection.json".format(self.clip_mgr.clip_name))
        if not os.path.exists(cam_det_path):
            self.run_cam_model_inference(out_file=cam_det_path)
        return ObstacleClipPred(cam_det_path)

    @staticmethod
    def get_lidar_pts_by_distance(lidar_pts, lidar_pts_img, target_distance):
        x = 0
        mask = lidar_pts[:, x] > target_distance
        return lidar_pts[mask], lidar_pts_img[mask]

    def get_target_lidar_pts(self, sample_name, cam_extrinsic, cam_intrinsic, img_w, img_h):
        lidar_pts = load_pcd(self.clip_mgr.get_pcd_path(sample_name))
        lidar_pts, lidar_pts_img, mask = project_lidar_to_img(lidar_pts, cam_extrinsic, cam_intrinsic, img_w, img_h)
        # lidar_pts, lidar_pts_img = self.get_lidar_pts_by_distance(lidar_pts, lidar_pts_img, self.target_distance)
        return lidar_pts, lidar_pts_img, mask

    @staticmethod
    def get_nearest_cluster(cluster_infos, bbox_center, cam_info):
        nearest_idx = 0
        if len(cluster_infos) > 1:
            cluster_centers = np.concatenate([item[1][None, ...] for item in cluster_infos], axis=0)
            _, cluster_centerpts_img, depth_mask = project_lidar_to_img(cluster_centers, *cam_info)
            l2_dist = np.linalg.norm(
                cluster_centerpts_img[:, [0, 1]] - np.array(bbox_center), ord=2, axis=0)
            nearest_idx = np.argmin(l2_dist).item()
        return nearest_idx

    def get_long_distance_objs(self, sample_id, cam_det_instants, cam_info, img_mat):
        lidar_pts, lidar_pts_img, mask = self.get_target_lidar_pts(sample_id, *cam_info)
        frame_results = {"sample_id": sample_id,
                         "pseudo_label": []}
        draw_color = (0, 255, 255)
        text_color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        pixel_offset = 10
        offsets = [0] if pixel_offset is None else [0, pixel_offset]
        x, y, z, end = 0, 1, 2, 3
        for obj_2d in cam_det_instants:
            bbox = (xmin, ymin, xmax, ymax) = obj_2d.get_img_bbox()
            center_pt = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
            category = obj_2d.get_shape()
            score = obj_2d.get_score()
            bbox_2d_mask = None
            for offset in offsets:
                bbox_2d_mask = (lidar_pts_img[:, x] > xmin - offset) & (lidar_pts_img[:, x] < xmax + offset) & \
                               (lidar_pts_img[:, y] > ymin - offset) & (lidar_pts_img[:, y] < ymax + offset)
                if np.sum(bbox_2d_mask) >= 6:
                    break
            if np.sum(bbox_2d_mask) < 1:
                continue
            target_noisy_lidar_pts = lidar_pts[bbox_2d_mask]
            target_noisy_lidar_pts_img = lidar_pts_img[bbox_2d_mask]
            cluster_result = cluster_pcd(target_noisy_lidar_pts)
            nearest_idx = self.get_nearest_cluster(cluster_result, center_pt, cam_info)
            nearest_cluster, nearest_center = cluster_result[nearest_idx]
            _, target_center_lidar_pts_img, _ = project_lidar_to_img(nearest_center[None, ...], *cam_info)
            center_x, center_y, _ = target_center_lidar_pts_img[0].tolist()
            if xmin < center_x < xmax and ymin < center_y < ymax:
                cv2.rectangle(img_mat, (xmin, ymin), (xmax, ymax), draw_color, 2, cv2.LINE_AA)
                cv2.putText(img_mat, category, (xmin, ymin), font, 1, text_color, 2)
                frame_results["pseudo_label"].append({"center_3d": nearest_center.tolist(),
                                                      "cluster_3d": nearest_cluster.tolist(),
                                                      "category": category,
                                                      "bbox_2d": bbox,
                                                      "score_2d": score})
        self.save_img(sample_id, img_mat, lidar_pts_img, cam_info)
        return frame_results, img_mat

    def save_img(self, sample_id, img_mat, lidar_pts_img, cam_info):
        plt_img = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        dpi = 100
        fig.set_dpi(dpi)
        fig.set_size_inches(cam_info[2] / dpi, cam_info[3] / dpi)
        ax.set_axis_off()
        ax.imshow(plt_img)
        ax.scatter(lidar_pts_img[:, [0]], lidar_pts_img[:, [1]], c=[lidar_pts_img[:, [2]]], cmap='rainbow_r', alpha=1,
                   s=3)
        out_img_path = os.path.join(self.out_path, "imgs")
        os.makedirs(out_img_path, exist_ok=True)
        plt.savefig(os.path.join(out_img_path, "{}.jpg".format(sample_id)), bbox_inches='tight')
        plt.close()

    def find_long_dist_objs(self, sample_id, lidar_det_frame, cam_det_frame):
        # get camera extrinsic, intrinsic, img_w, img_h
        cam_info = (extrinsic, intrinsic, img_w, img_h) = self.clip_mgr.get_cam_info(self.target_cam)
        img_mat = cv2.imread(self.clip_mgr.get_img_path(sample_id, self.target_cam))

        # get 3d box within cam view and corresponding 2d projected box
        lidar_det_instants = lidar_det_frame.get_instants_in_cam_view(*cam_info)
        lidar_det_bbox_2d_list, img_mat = lidar_det_frame.project_3d_to_2d(lidar_det_instants,
                                                                           extrinsic,
                                                                           intrinsic,
                                                                           img_mat)

        # get 2d box belong to target category(VRU), then run NMS, then keep box not overlap with 3d projected boxes
        cam_det_instants = cam_det_frame.get_instant_objects()
        # for instant in cam_det_instants:
        #     instant.render_cv(img_mat, color=(255, 0, 255))
        cam_det_instants = [instant for instant in cam_det_instants if instant.get_shape() in self.target_category]
        # cam_det_instants = nms_bbox(cam_det_instants)
        cam_det_instants = bbox_overlap_filter(lidar_det_bbox_2d_list, cam_det_instants)

        # get lidar points in the view of camera

        long_dist_objs, img_mat = self.get_long_distance_objs(sample_id, cam_det_instants, cam_info, img_mat)
        # self.save_img(sample_id, img_mat)
        return long_dist_objs

    def run(self):
        results = []
        index_list = self.lidar_det_obj.get_ts_list()
        pbar = tqdm.tqdm(index_list, total=len(index_list), desc="long distance obj searching")
        for index in pbar:
            lidar_det_frame = self.lidar_det_obj.get_frame_obj_by_ts(index)
            cam_det_frame = self.cam_det_obj.get_frame_obj_by_ts(index)
            frame_target_objs = self.find_long_dist_objs(index, lidar_det_frame, cam_det_frame)
            results.append(frame_target_objs)
        return results

    def save_results(self, results):
        out_result_file = os.path.join(self.out_path, "{}_pseudolabel.json".format(self.clip_mgr.clip_name))
        with open(out_result_file, 'w') as f:
            json.dump(results, f)

    def start(self):
        long_dist_cc_result = self.run()
        self.save_results(long_dist_cc_result)
