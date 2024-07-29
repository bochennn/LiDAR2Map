import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import yaml
from plugin.utils import *
from scipy.spatial.transform import Rotation
from tqdm import tqdm

OD_ANNOTATION_PREFIX = [
    '3d_city_object_detection_with_fish_eye',
    'only_3d_city_object_detection',
]
USING_CAMERA = [
    'camera0',  # cam_front_center
    # 'camera1',  # cam_front_center_tele
    'camera2',  # cam_front_left
    'camera3',  # cam_rear_left
    'camera4',  # cam_front_right
    'camera5',  # cam_rear_right
    'camera6',  # cam_rear_center
    # 'camera7',  # cam_fisheye_left
    # 'camera8',  # cam_fisheye_rear
    # 'camera9',  # cam_fisheye_front
    # 'camera10', # cam_fisheye_right
]
USING_LIDAR = [
    'lidar0',   # lidar top
    # 'lidar1',   # lidar left
    # 'lidar2',   # lidar front
    # 'lidar3',   # lidar right
]
VIRTUAL2IMU = transform_matrix(
    [0., 0., 0.36], Rotation.from_rotvec([0, 0, -np.pi * 0.5]).as_matrix())


def create_zdrive_infos(root_path: Path, out_dir: Path, info_prefix: str,
                        batch_names: List[str], workers: int = 1):

    available_clips = get_available_scenes(root_path, batch_names)
    val_scenes = Path('data/zdrive/val.txt').read_text().splitlines()

    train_zd_infos, val_zd_infos = _fill_trainval_infos(available_clips, val_scenes)

    metadata = dict(version='')
    print('train sample: {}, val sample: {}'.format(
            len(train_zd_infos), len(val_zd_infos)))
    data = dict(infos=train_zd_infos, metadata=metadata)

    with open(out_dir / f'{info_prefix}_infos_train.pkl', 'wb') as f:
        pickle.dump(data, f)

    data['infos'] = val_zd_infos
    with open(out_dir / f'{info_prefix}_infos_val.pkl', 'wb') as f:
        pickle.dump(data, f)


def _fill_trainval_infos(available_clips: List[Path],
                         val_scenes: List[str],
                         max_sweeps: int = 10):
    train_zd_infos = []
    val_zd_infos = []
    # from visualization import show_o3d
    # from common.fileio import read_points_pcd

    for clip_root in tqdm(available_clips):
        imu2lidar_dict =  yaml.safe_load((clip_root / 'extrinsics/lidar2imu/lidar2imu.yaml').read_text())
        imu2lidar = transform_matrix(
            list(imu2lidar_dict['transform']['translation'].values()),
            Rotation.from_quat(list(imu2lidar_dict['transform']['rotation'].values())).as_matrix()
        )

        pose_record = json.loads((clip_root / 'localization.json').read_text())
        pose_timestamps = np.asarray([p['timestamp'] for p in pose_record])
        ann_path = clip_root / 'annotation' / \
            [p for p in OD_ANNOTATION_PREFIX if (clip_root / 'annotation' / p).exists()][0]
        clip_info = json.loads(list(ann_path.glob('clip_*.json'))[0].read_text())
        # ref2global = None

        frame_indices = np.argsort([f['frame_name'] for f in clip_info['frames']])
        # pts_list, box_list = [], []

        for frame_indx in frame_indices:
            frame_info = clip_info['frames'][frame_indx]

            closest_ind = np.fabs(pose_timestamps - frame_info['lidar_collect'] * 1e-6).argmin()

            global2imu = transform_matrix(
                list(pose_record[closest_ind]['pose']['position'].values()),
                Rotation.from_quat([
                    pose_record[closest_ind]['pose']['orientation']['qx'],
                    pose_record[closest_ind]['pose']['orientation']['qy'],
                    pose_record[closest_ind]['pose']['orientation']['qz'],
                    pose_record[closest_ind]['pose']['orientation']['qw']]).as_matrix())

            out_info = {
                'token': frame_info['frame_id'],
                'frame_name': frame_info['frame_name'],
                'scene_name': clip_root.stem,
                'lidars': dict(),
                'cams': dict(),
                'sweeps': [],
                'timestamp': frame_info['lidar_collect'],
            }

            for cam_token in USING_CAMERA:
                cam_filename = clip_root / frame_info['frame_name'] / frame_info[cam_token]
                if not cam_filename.exists():
                    continue
                extrinsics = np.asarray(clip_info['calibration'][cam_token]['extrinsic'])

                cam_info = dict(
                    filename=str(cam_filename),
                    cam_name=clip_info['mapping'][cam_token].replace(' ', '_'),
                    type=clip_info['calibration'][cam_token]['distortion_model'],
                    cam_intrinsic=np.asarray(clip_info['calibration'][cam_token]['intrinsic']),
                    distortion=np.asarray(clip_info['calibration'][cam_token]['distcoeff']).flatten(),
                    sensor2ego_translation=extrinsics[:3, 3],
                    sensor2ego_rotation=extrinsics[:3, :3],
                    timestamp=frame_info['camera_collect'],
                    width=clip_info['calibration'][cam_token]['width'],
                    height=clip_info['calibration'][cam_token]['height'],
                )
                out_info['cams'].update({cam_token: cam_info})

            for lidar_token in USING_LIDAR:
                lidar_filename = clip_root / frame_info['frame_name'] / frame_info[lidar_token]

                lidar2virtual = VIRTUAL2IMU @ imu2lidar
                lidar_info = dict(
                    filename=str(lidar_filename),
                    timestamp=frame_info['lidar_collect'],
                    sensor2ego_translation=lidar2virtual[:3, 3],
                    sensor2ego_rotation=lidar2virtual[:3, :3]
                )
                out_info['lidars'].update({lidar_token: lidar_info})

            # points = read_points_pcd(out_info['lidars']['lidar0']['filename'])
            lidar2ego = transform_matrix(
                out_info['lidars']['lidar0']['sensor2ego_translation'],
                out_info['lidars']['lidar0']['sensor2ego_rotation']
            )
            ego2global = global2imu @ np.linalg.inv(VIRTUAL2IMU)

            out_info['ego2global_translation'] = ego2global[:3, 3]
            out_info['ego2global_rotation'] = ego2global[:3, :3]
            out_info['ego_velo'] = convert_velos(
                np.asarray(list(pose_record[closest_ind]['pose']['linear_velocity'].values())),
                ego2global
            )

            # if ref2global is None:
            #     ref2global = ego2global
            # ego2ref = transform_offset(ego2global, ref2global)

            # points = convert_points(points, ego2ref @ lidar2ego)
            # pts_list.append(points)

            # in lidar0 coordinate
            annotations = frame_info['annotated_info'][f'{ann_path.stem}_annotated_info']
            od_ann_info = annotations['annotated_info']['3d_object_detection_info']['3d_object_detection_anns_info']

            locs = np.array([a['obj_center_pos'] for a in od_ann_info]).reshape(-1, 3)
            dims = np.array([a['size'] for a in od_ann_info]).reshape(-1, 3)
            rots = np.array([Rotation.from_quat(a['obj_rotation']).as_rotvec() for a in od_ann_info])

            if len(locs) > 0:
                out_info['gt_boxes'] = convert_boxes(
                    np.concatenate([locs, dims, rots[:, 2:3]], axis=1), lidar2ego)
                # out_info['gt_boxes'] = convert_boxes(out_info['gt_boxes'], ego2ref)
            else:
                out_info['gt_boxes'] = np.zeros((0, 7))
            out_info['gt_names'] = np.array([a['category'] for a in od_ann_info])
            out_info['track_ids'] = np.array([a['track_id'] for a in od_ann_info])
            out_info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in od_ann_info])

            # box_list.append(out_info['gt_boxes'])
            if out_info['scene_name'] in val_scenes:
                val_zd_infos.append(out_info)
            else:
                train_zd_infos.append(out_info)
            # show_o3d([pts_list[0]], [{'box3d': box_list[0]}])
        # show_o3d([np.vstack(pts_list)], [{'box3d': np.vstack(box_list)}])
    return train_zd_infos, val_zd_infos


def get_available_scenes(root_path: Path, batch_names: List[str]) -> List[Path]:
    clip_list = []
    for n in batch_names:
        for fpath in (root_path / n).glob('clip_*'):
            if not fpath.is_dir():
                continue
            if not (fpath / 'annotation').exists():
                continue
            if not any([(fpath / 'annotation' / ann_prefix).exists()
                for ann_prefix in OD_ANNOTATION_PREFIX]):
                # print(list((fpath / 'annotation').glob('*')))
                continue
            clip_list.append(fpath)
    return clip_list