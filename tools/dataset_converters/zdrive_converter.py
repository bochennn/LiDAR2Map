import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import yaml
from plugin.utils import *
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def parse_23d_object_detection_anno_info(anno_dict: Dict, *args):
    return anno_dict['3d_object_detection_annotated_info']\
                    ['annotated_info']\
                    ['3d_object_detection_info']\
                    ['3d_object_detection_anns_info']


def parse_3d_object_detection_anno_info(anno_dict: Dict, prefix: str):
    return anno_dict[f'{prefix}_annotated_info']\
                    ['annotated_info']\
                    ['3d_object_detection_info']\
                    ['3d_object_detection_anns_info']


def get_virtual_to_imu():

    return

VIRTUAL2IMU = transform_matrix(
    [0., 0., 0.36],
    Rotation.from_rotvec([0, 0, -np.pi * 0.5]).as_matrix()
)


def filter_person_size(anno_box: Dict):
    if anno_box['category'] in ['person'] and \
        anno_box['size'][0] > 1.5 or anno_box['size'][1] > 1.5:
        return False
    return True


OD_ANNOTATION_PREFIX = {
    '3d_city_object_detection_with_fish_eye': parse_3d_object_detection_anno_info,
    '3d_highway_object_detection_with_fish_eye': parse_3d_object_detection_anno_info,
    'only_3d_city_object_detection': parse_3d_object_detection_anno_info,
    '23d_object_detection': parse_23d_object_detection_anno_info,
}

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
CLASS_NAMES = {
    'traffic_warning': 'traffic_warning',
    'traffic_cone': 'traffic_cone',
    'barrier': 'barrier',
    'person': 'person',
    'bicycle': 'bicycle',
    'tricycle': 'tricycle',
    'motorcycle': 'motorcycle',
    'car': 'car',
    'pickup_truck': 'pickup_truck',
    'recreational_vehicle': 'recreational_vehicle',
    'Recreational_vehicle': 'recreational_vehicle',
    'construction_vehicle': 'construction_vehicle',
    'bus': 'bus',
    'truck': 'truck'
}

LIDAR2IMU_FILEPATH = 'extrinsics/lidar2imu/lidar2imu.yaml'


def create_zdrive_infos(root_path: Path, out_dir: Path, batch_name: str, workers: int):

    available_clips = get_available_scenes(root_path, batch_name)
    # progressbar = tqdm(total=len(available_clips))

    info_list_by_clip = multi_process_thread(
        _fill_trainval_infos,
        [dict(clip_root=clip, max_sweeps=10,
              progressbar='[{:6d}/{:6d}] {}'.format(ind, len(available_clips), clip.name))
            for ind, clip in enumerate(available_clips)],
        nprocess=workers)

    info_list_by_frame = []
    for info_list in info_list_by_clip:
        info_list_by_frame.extend(info_list)

    if len(len(info_list_by_frame)) == 0:
        return

    info_prefix = batch_name.replace('-undownloaded', '')
    metadata = dict(
        version=info_prefix,
        num_clips=len(info_list_by_clip),
        num_frames=len(info_list_by_frame)
    )
    write_info_path = out_dir / f'{info_prefix}_infos'\
                                f'_clip_{metadata["num_clips"]}'\
                                f'_frames_{metadata["num_frames"]}.pkl'
    print(f'Num Clips: {metadata["num_clips"]}, '\
          f'Num Frames: {metadata["num_frames"]}\n'\
          f'Write info into {write_info_path}')
    data = dict(infos=info_list_by_frame, metadata=metadata)
    with open(write_info_path, 'wb') as f:
        pickle.dump(data, f)


def _fill_trainval_infos(clip_root: Path, max_sweeps: int = 10, progressbar: str = '', vis: bool = False) -> List[Dict]:
    """ """
    if vis:
        from common.fileio import read_points_pcd
        from visualization import show_o3d

    imu2lidar_dict = yaml.safe_load((clip_root / LIDAR2IMU_FILEPATH).read_text().replace('\t', '    '))
    imu2lidar = transform_matrix(
        list(imu2lidar_dict['transform']['translation'].values()),
        Rotation.from_quat(list(imu2lidar_dict['transform']['rotation'].values())).as_matrix()
    )

    pose_record = json.loads((clip_root / 'localization.json').read_text())
    pose_timestamps = np.asarray([p['timestamp'] for p in pose_record])
    ann_path = clip_root / 'annotation' / \
        [p for p in OD_ANNOTATION_PREFIX if (clip_root / 'annotation' / p).exists()][0]
    clip_info = json.loads(list(ann_path.glob('clip_*.json'))[0].read_text())

    frame_indices = np.argsort([f['frame_name'] for f in clip_info['frames']])
    info_list, pts_list, box_list, track_list, ref2global = [], [], [], [], None

    for frame_indx in tqdm(frame_indices, desc=progressbar):
        frame_info = clip_info['frames'][frame_indx]
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

        closest_ind = np.fabs(pose_timestamps - out_info['timestamp'] * 1e-6).argmin()
        global2imu = transform_matrix(
            list(pose_record[closest_ind]['pose']['position'].values()),
            Rotation.from_quat([
                pose_record[closest_ind]['pose']['orientation']['qx'],
                pose_record[closest_ind]['pose']['orientation']['qy'],
                pose_record[closest_ind]['pose']['orientation']['qz'],
                pose_record[closest_ind]['pose']['orientation']['qw']]).as_matrix())

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

        # in lidar0 coordinate
        od_ann_info = OD_ANNOTATION_PREFIX[ann_path.name](
            frame_info['annotated_info'], ann_path.name)

        locs = np.array([a['obj_center_pos'] for a in od_ann_info]).reshape(-1, 3)
        dims = np.array([a['size'] for a in od_ann_info]).reshape(-1, 3)
        rots = np.array([Rotation.from_quat(a['obj_rotation']).as_rotvec() for a in od_ann_info])

        if len(locs) > 0:
            out_info['gt_boxes'] = convert_boxes(
                np.concatenate([locs, dims, rots[:, 2:3]], axis=1), lidar2ego)
        else:
            out_info['gt_boxes'] = np.zeros((0, 7))

        valid_flag = []
        for ann in od_ann_info:
            is_valid = True
            if ann['num_lidar_pts'] < 5:
                is_valid = False
            elif ann['category'] not in CLASS_NAMES:
                is_valid = False
            elif ann['is_group']:
                is_valid = False
            elif ann['category'] in ['car', 'pickup_truck'] and \
                (ann['size'][0] > 8 or ann['size'][0] < 1 or ann['size'][1] > 2.3 or ann['size'][1] < 1):
                is_valid = False
            elif ann['category'] in ['bicycle', 'tricycle', 'motorcycle'] and \
                (ann['size'][0] > 6 or ann['size'][1] > 1.5):
                is_valid = False
            elif ann['category'] in ['truck', 'bus'] and \
                (ann['size'][0] > 30 or ann['size'][0] < 2 or ann['size'][1] > 6 or ann['size'][1] < 1):
                is_valid = False
            elif ann['category'] in ['construction_vehicle'] and \
                (ann['size'][0] < 1 or ann['size'][1] > 6):
                is_valid = False
            elif ann['category'] in ['person'] and \
                (ann['size'][0] > 1.5 or ann['size'][1] > 1.5):
                is_valid = False
            elif ann['category'] in ['traffic_cone'] and \
                (ann['size'][0] > 1 or ann['size'][1] > 1):
                is_valid = False
            valid_flag.append(is_valid)

        out_info['valid_flag'] = np.asarray(valid_flag, dtype=bool)
        out_info['gt_names'] = np.array([CLASS_NAMES[a['category']] if is_valid else 'ignore'
            for is_valid, a in zip(out_info['valid_flag'], od_ann_info)])
        out_info['track_ids'] = np.array([a['track_id'] for a in od_ann_info])
        out_info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in od_ann_info])

        info_list.append(out_info)

        if vis:
            if ref2global is None:
                ref2global = ego2global
            ego2ref = transform_offset(ego2global, ref2global)

            points = read_points_pcd(out_info['lidars']['lidar0']['filename'])
            points = convert_points(points, ego2ref @ lidar2ego)
            # points = convert_points(points, lidar2ego)
            pts_list.append(points)
            box_list.append(convert_boxes(out_info['gt_boxes'], ego2ref))
            # box_list.append(out_info['gt_boxes'])
            track_list.append(out_info['track_ids'])

            # if len(out_info['gt_names']) > 0:
            #     show_o3d([pts_list[-1]], [{'box3d': box_list[-1],
            #                                'labels': track_list[-1],
            #                                'texts': out_info['gt_names']}])
    if vis:
        show_o3d([np.vstack(pts_list)],
                 [{'box3d': np.vstack(box_list),
                   'labels': np.hstack(track_list)}])
    return info_list


def get_available_scenes(root_path: Path, batch_name: str) -> List[Path]:
    clip_list = []
    for fpath in (root_path / batch_name).glob('clip_*'):
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