import json
from pathlib import Path
from typing import List

import numpy as np
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


def create_zdrive_infos(root_path: Path, batch_names: List[str]):

    available_clips = get_available_scenes(root_path, batch_names)

    train_nusc_infos = []
    val_nusc_infos = []

    for clip_root in tqdm(available_clips):

        pose_record = json.loads((clip_root / 'localization.json').read_text())
        ann_path = clip_root / 'annotation' / \
            [p for p in OD_ANNOTATION_PREFIX if (clip_root / 'annotation' / p).exists()][0]
        clip_info = json.loads(list(ann_path.glob('clip_*.json'))[0].read_text())

        for frame_info in clip_info['frames']:
            # frame_name = sample.stem

            out_info = {
                'token': frame_info['frame_id'],
                'frame_name': frame_info['frame_name'],
                'lidars': dict(),
                'cams': dict(),
                'sweeps': [],
                # ego2global_translation': pose_record['translation'],
                # ego2global_rotation': pose_record['rotation'],
            }

            for cam_token in USING_CAMERA:
                extrinsics = np.asarray(clip_info['calibration'][cam_token]['extrinsic'])

                cam_info = dict(
                    filename=frame_info[cam_token],
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
                # lidar_token = sample['data'][cam]
                # cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                # cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                #                             e2g_t, e2g_r_mat, cam)
                lidar_info = dict(
                    filename=frame_info[lidar_token],
                    timestamp=frame_info['lidar_collect'],
                )
                out_info['lidars'].update({lidar_token: lidar_info})

            # print(clip_info['calibration'].keys())
            # print(out_info['lidars'])

            annotations = frame_info['annotated_info'][f'{ann_path.stem}_annotated_info']
            od_ann_info = annotations['annotated_info']['3d_object_detection_info']['3d_object_detection_anns_info']

            # subcategory, group_id, is_group
            #     track_id = od_ann['track_id']

            locs = np.array([a['obj_center_pos'] for a in od_ann_info]).reshape(-1, 3)
            dims = np.array([a['size'] for a in od_ann_info]).reshape(-1, 3)
            rots = np.array([Rotation.from_quat(a['obj_rotation']).as_rotvec() for a in od_ann_info])
            # out_info['gt_'] = np.array([a['track_id'] for a in od_ann_info])
            out_info['gt_names'] = np.array([a['category'] for a in od_ann_info])
            out_info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in od_ann_info])
            print('---')
        print(xx)


def get_available_scenes(root_path: Path, batch_names: List[str]):
    clip_list = []
    for n in batch_names:
        for fpath in (root_path / n).glob('clip_*'):
            if not fpath.is_dir():
                continue
            if not (fpath / 'annotation').exists():
                continue
            if not any([(fpath / 'annotation' / ann_prefix).exists()
                for ann_prefix in OD_ANNOTATION_PREFIX]):
                print(list((fpath / 'annotation').glob('*')))
                continue
            clip_list.append(fpath)
            break
    return clip_list