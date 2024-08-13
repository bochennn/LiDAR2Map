from typing import Dict
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

CLASS_NAMES = [
    'car', 'pickup_truck', 'truck', 'construction_vehicle',
    'bus', 'tricycle', 'motorcycle', 'bicycle', 'person', 'traffic_cone',
    # 'traffic_warning' 'barrier' 'recreational_vehicle'
]

INFO_ROOT = Path('/home/bochen/workspace/LiDAR2Map/data/zdrive')
INFO_LIST = {
    # 'CITY-3D-0529_infos_clip_1134_frames_45116.pkl': 8,
    'HY-3D-0529_infos_clip_5933_frames_232061.pkl': 15,
    'E03-CITY-20240702_infos_clip_272_frames_10737.pkl': 2,
    'E03-HY-20240702_infos_clip_1250_frames_49491.pkl': 30,
    'NON-E03-CITY-20240702_infos_clip_935_frames_37048.pkl': 2,
    'NON-E03-HY-20240702_infos_clip_5583_frames_220673.pkl': 30,
}

def load_info(info_path: Path):
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)
    print('loading', info_path.name)
    return info_dict


def write_info(info_path: Path, info_dict: Dict):
    with open(info_path, 'wb') as f:
        pickle.dump(info_dict, f)
    print('write info', info_path)


def sub_set_with_interval(info_dict: Dict, interval: int):
    clip_names = {inf['scene_name'] for inf in info_dict['infos']}
    sub_clip_names = list(clip_names)[::interval]

    new_info = dict(infos=[inf for inf in info_dict['infos'] if inf['scene_name'] in sub_clip_names])
    new_info.update(metadata=dict(
        version=info_dict['metadata']['version'],
        num_clips=len(sub_clip_names), num_frames=len(new_info['infos'])
    ))
    print('create sub info', new_info['metadata'])
    return new_info


def box_size_by_class(info_dict: Dict):
    cls_box_size = dict()
    for info in info_dict['infos']:
        for box, cls_name in zip(info['gt_boxes'], info['gt_names']):
            if cls_name in cls_box_size:
                cls_box_size[cls_name].append(box[3:6])
            else:
                cls_box_size[cls_name] = [box[3:6]]
    return cls_box_size


for info_path, sub_set_interval in INFO_LIST.items():
    info_dict = load_info(INFO_ROOT / info_path)
    new_info_dict = sub_set_with_interval(info_dict, interval=sub_set_interval)

    info_name = info_path.split('_')[0]
    assert info_name == info_dict['metadata']['version'] == new_info_dict['metadata']['version']

    cls_box_size = box_size_by_class(new_info_dict)
    print(f'subset after interval {sub_set_interval}')
    print(' '.join([str(len(cls_box_size[n])) if n in cls_box_size else '0' for n in CLASS_NAMES]))

    write_info('{}_interval_{}_clip_{}_frames_{}.pkl'.\
        format(info_name, sub_set_interval,
               new_info_dict['metadata']['num_clips'],
               new_info_dict['metadata']['num_frames']), new_info_dict)

# fig = plt.figure('box size')
# ax = fig.add_subplot(111, projection='3d')

# print(' '.join([str(np.array(cls_box_size[n]).var(axis=0)[1]) if n in cls_box_size else str(0) for n in CLASS_NAMES]))
# print(' '.join([str(np.array(cls_box_size[n]).var(axis=0)[0]) if n in cls_box_size else str(0) for n in CLASS_NAMES]))
# print(' '.join([str(np.array(cls_box_size[n]).var(axis=0)[2]) if n in cls_box_size else str(0) for n in CLASS_NAMES]))

# for cls in CLASS_NAMES:
#     if cls not in cls_box_size:
#         continue
#     cls_size = cls_box_size[cls]

#     _cls_size = np.array(cls_size)
#     _cls_size_mean = _cls_size.mean(axis=0)
#     # ax.scatter(_cls_size_mean[0], _cls_size_mean[1], _cls_size_mean[2], label=cls)
#     # plt.plot(_cls_size[:, 0], _cls_size[:, 1], 'o', label=cls)
#     plt.plot(_cls_size_mean[0], _cls_size_mean[1], 'o', label=cls)

# plt.xlabel('length')
# plt.ylabel('width')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#            fancybox=True, shadow=True, ncol=5)
# plt.show()