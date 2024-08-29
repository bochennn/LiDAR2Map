import pickle
from pathlib import Path
from typing import Dict, List

import time
# import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    'car', 'pickup_truck', 'truck', 'construction_vehicle',
    'bus', 'tricycle', 'motorcycle', 'bicycle', 'person', 'traffic_cone',
    # 'traffic_warning' 'barrier' 'recreational_vehicle'
]

INFO_ROOT = Path('/data/sfs_turbo/dataset/data_info')
TEST_HIGHWAY = Path('/data/sfs_turbo/dataset/data_split/test_city.txt')

INFO_LIST = {
    'CITY-ONLY-3D-L2+-NON-E03': [
        'CITY-ONLY-3D-L2+-NON-E03-240423_infos_clip_1764_frame_69848.pkl',
        'CITY-ONLY-3D-L2+-NON-E03-240529_infos_clip_1134_frame_45116.pkl',
        'CITY-ONLY-3D-L2+-NON-E03-240702_infos_clip_935_frame_37048.pkl'],
    # 'HIGHWAY-23D-L2+-NON-E03': [
    #     'HIGHWAY-23D-L2+-NON-E03-240423_infos_clip_4529_frame_178509.pkl',
    #     'HIGHWAY-23D-L2+-NON-E03-240529_infos_clip_5933_frame_232061.pkl',
    #     'HIGHWAY-23D-L2+-NON-E03-240702_infos_clip_5583_frame_220673.pkl'],
}


def to_format_time(timestamp, time_format='%Y%m%d%H'):
    return time.strftime(time_format, time.localtime(timestamp * 1e-3))


def load_info(info_path: Path):
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)
    print('loading', info_path.name)
    return info_dict


def write_info(info_path: Path, info_dict: Dict):
    with open(info_path, 'wb') as f:
        pickle.dump(info_dict, f)
    print('write info', info_path)


def sample_with_interval(info_list: List, info_name: str, interval: int):
    clip_names = np.unique([inf['scene_name'] for inf in info_list])
    clip_names.sort()

    clip_time_h, clip_indices = np.unique([
        to_format_time(int(n.split('_')[-1])) for n in clip_names], return_inverse=True)
    clip_exclude_date = np.hstack([clip_names[clip_indices == i] for i, n in 
        enumerate(clip_time_h) if n not in testset_by_hour])
    sample_clip_names = clip_exclude_date[::interval]

    # clip_included = [n for n in clip_names if n in testset_highway]
    # sample_clip_names = clip_included[::interval]

    new_info = dict(infos=[inf for inf in info_list if inf['scene_name'] in sample_clip_names])
    new_info.update(metadata=dict(
        version=info_name,
        num_clips=len(sample_clip_names), num_frames=len(new_info['infos'])
    ))
    return new_info


def box_size_by_class(info_list: List):
    cls_box_size = dict()
    for info in info_list:
        for box, cls_name in zip(info['gt_boxes'], info['gt_names']):
            if cls_name in cls_box_size:
                cls_box_size[cls_name].append(box[3:6])
            else:
                cls_box_size[cls_name] = [box[3:6]]
    return cls_box_size


testset_highway = TEST_HIGHWAY.read_text().splitlines()
testset_by_hour = np.unique([
    to_format_time(int(n.split('_')[-1])) for n in testset_highway])
sample_interval = 2

for batch_name, info_path_list in INFO_LIST.items():
    info_list = [info for p in info_path_list for info in load_info(INFO_ROOT / p)['infos']]

    new_info_dict = sample_with_interval(info_list, batch_name, interval=sample_interval)
    print('create sampled info', new_info_dict['metadata'])

    write_info('{}_interval_{}_clip_{}_frame_{}.pkl'.\
        format(batch_name, sample_interval,
               new_info_dict['metadata']['num_clips'],
               new_info_dict['metadata']['num_frames']), new_info_dict)
    # write_info('{}_val_clip_{}_frame_{}.pkl'.\
    #     format(batch_name,
    #            new_info_dict['metadata']['num_clips'],
    #            new_info_dict['metadata']['num_frames']), new_info_dict)

# cls_box_size = box_size_by_class(new_info_dict)
# print(f'subset after interval {sub_set_interval}')
# print(' '.join([str(len(cls_box_size[n])) if n in cls_box_size else '0' for n in CLASS_NAMES]))

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