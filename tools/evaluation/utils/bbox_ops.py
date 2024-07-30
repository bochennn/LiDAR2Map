import numpy as np
from pyquaternion import Quaternion
import cv2

from utils.pointcloud_ops import project_lidar_to_img
from log_mgr import logger


LABEL_COLOR = {
    'car': (255, 0, 0),
    'pickuptruck': (0, 255, 0),
    'truck': (0, 0, 255),
    'constructionvehicle': (0, 0, 0),
    'bus': (255, 0, 255),
    'tricycle': (0, 125, 125),
    'motorcycle': (125, 125, 0),
    'bicycle': (125, 0, 125),
    'person': (125, 255, 125)
}


def get_quaternion(yaw, pitch=None, roll=None):
    quat = Quaternion(axis=[0, 0, 1], radians=yaw)
    if roll is not None:
        quat = quat * Quaternion(axis=[1, 0, 0], radians=roll)
    if pitch is not None:
        quat = quat * Quaternion(axis=[0, 1, 0], radians=pitch)
    return quat


def get_iou(b1, b2):
    x1, y1, x2, y2 = b1
    u1, v1, u2, v2 = b2
    tl = np.max([b1[: 2], b2[: 2]], axis=0)
    br = np.min([b1[2:], b2[2:]], axis=0)
    if br[0] < tl[0] or br[1] < tl[1]:
        intersect_area = 0
    else:
        w, h = (br - tl).tolist()
        intersect_area = w * h
    if intersect_area <= 0:
        return 0
    else:
        b1_area = (x2 - x1) * (y2 - y1)
        b2_area = (u2 - u1) * (v2 - v1)
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / max(1e-6, union_area)
    return iou


def is_point_in_box(point, box):
    xmin, ymin, xmax, ymax = 0, 1, 2, 3
    return box[xmin] < point[0] < box[xmax] and box[ymin] < point[1] < box[ymax]


def nms_bbox(objs, iou_threshold=0.6):
    objs.sort(key=lambda obj: obj.get_score(), reverse=True)
    ignore_idx = set()
    obj_num = len(objs)
    for idx, obj in enumerate(objs):
        if idx in ignore_idx:
            continue
        bbox = obj.get_img_bbox()
        for idx2 in range(idx+1, obj_num):
            if idx2 in ignore_idx:
                continue
            ref_bbox = objs[idx2].get_img_bbox()
            iou = get_iou(bbox, ref_bbox)
            if iou >= iou_threshold:
                ignore_idx.add(idx2)
    filtered_objs = [obj for idx, obj in enumerate(objs) if idx not in ignore_idx]
    return filtered_objs


def get_3d_corners(x, y, z, length, width, height, rot_matrix):
    x_corners = length / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = height / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])

    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(rot_matrix, corners)

    # Translate
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return np.transpose(corners)


def get_2d_corners(xmin, ymin, xmax, ymax):
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def get_3d_boxes_in_cam_view(objs, cam_extrinsic, cam_intrinsic, img_w, img_h):
    center_pt_list = np.array([obj.get_obj_center() for obj in objs])
    _, _, mask = project_lidar_to_img(center_pt_list, cam_extrinsic, cam_intrinsic, img_w, img_h)
    return np.array(objs)[np.nonzero(mask)[0].tolist()]


def project_3dbox_to_2dbox(objs, cam_extrinsic, cam_intrinsic, img_buf=None):
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2, 2, 2]), axis=1).astype(np.float32)
    corners_norm = corners_norm[[0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2], :] - 0.5

    bbox_3d_array = np.array([obj.get_bbox_3d() for obj in objs])
    rot_matrix = np.stack(list([np.transpose(obj.get_rotation_matrix()) for obj in objs]), axis=0)
    category_list = [obj.get_shape() for obj in objs]
    corners = np.multiply(bbox_3d_array[:, 3:6].reshape(-1, 1, 3), corners_norm)
    corners = np.einsum('nij,njk->nik', corners, rot_matrix) + bbox_3d_array[:, :3].reshape((-1, 1, 3))

    # 中心点
    center_point = bbox_3d_array[:, :3]  ### shape (N, 3)
    pad_center_point = np.concatenate([center_point, np.ones_like(center_point[:, 0:1])], axis=-1)
    center_point_cam = np.matmul(pad_center_point, cam_extrinsic.T)
    depth_mask = center_point_cam[:, 2] > 0

    center_points_img = np.matmul(center_point_cam, cam_intrinsic.T)
    center_points_img = center_points_img / center_points_img[:, 2:3]
    xywh_bboxes = []
    for i, corner in enumerate(corners):
        pre_label = category_list[i]
        points_homo = np.hstack([corner, np.ones(corner.shape[0], dtype=np.float32).reshape((-1, 1))])
        # points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
        points_lidar = np.dot(points_homo, cam_extrinsic.T)
        mask = points_lidar[:, 2] > 0
        points_lidar = points_lidar[mask]
        points_img = np.dot(points_lidar, cam_intrinsic.T)
        points_img = (points_img / points_img[:, [2]]).astype(np.int32)

        if points_img.shape[0] == 0:
            continue

        _points_img = points_img[:, :2].tolist()
        points_img_xywh = cv2.boundingRect(np.int32(_points_img))
        # xywh_bboxes.append(points_img_xywh)
        xywh_bboxes.append((points_img_xywh, pre_label))

        pre_color = LABEL_COLOR.get(pre_label, (255, 0, 255))
        if img_buf is None:
            continue
        for j in range(points_img.shape[0] - 1):
            if j == 9 or j == 13:
                continue
            p1 = (int(points_img[j][0]), int(points_img[j][1]))
            p2 = (int(points_img[j + 1][0]), int(points_img[j + 1][1]))
            # if j >= 5 and j <= 8:  # yaw 角方向
            #     draw_color = (0, 255, 255)
            # else:
            #     draw_color = pre_color
            draw_color = (255, 255, 0)
            try:
                cv2.line(img_buf, p1, p2, draw_color, 2, cv2.LINE_AA)
            except:
                logger.error("line error: ", p1, p2)

    # if (img_buf is not None) and (len(xywh_bboxes) > 0):
    #     img_h, img_w = img_buf.shape[:2]
    #     xywh_bboxes = clip_xywhbbox(xywh_bboxes, img_w, img_h)
    #     draw_color = (0, 0, 125)
    #     xywh_bboxes.sort(key=lambda item: item[0][2] * item[0][3])  # sort by  bbox area
    #     for points_img_xywh, _ in xywh_bboxes:
    #         _x, _y, _w, _h = points_img_xywh
    #         p1, p2 = (_x, _y), (_x + _w, _y + _h)
    #         cv2.rectangle(img_buf, p1, p2, draw_color, 2, cv2.LINE_AA)
    return xywh_bboxes, img_buf


def bbox_overlap_filter(base_bbox_list, target_objs):
    # return boxes in base_bbox_list whose center point not in any of box in target_bbox_list
    filter_idx = set()
    for idx, target_obj in enumerate(target_objs):
        xmin, ymin, xmax, ymax = target_obj.get_img_bbox()
        center_pt = [0.5 * (xmin + xmax), 0.5 * (ymin + ymax)]
        for base_bbox in base_bbox_list:
            if is_point_in_box(center_pt, base_bbox):
                filter_idx.add(idx)
                break
    return [target_box for idx, target_box in enumerate(target_objs) if idx not in filter_idx]
