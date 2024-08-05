import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    'car',
    'pickup_truck',
    'truck',
    'construction_vehicle',
    'bus',
    'tricycle',
    'motorcycle',
    'bicycle',
    'person',
    'traffic_cone',
    # 'traffic_warning'
    # 'barrier'
    # 'recreational_vehicle'
]

cls_box_size = dict()
for info_path in list(Path('data/zdrive_all').glob('*.pkl')):
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)
    print('loading', info_path.name)

    # for info in info_dict['infos']:
    #     for box, cls in zip(info['gt_boxes'], info['gt_names']):
    #         # if not is_valid:
    #         #     print(info_dict['metadata']['version'], info['scene_name'], info['frame_name'], info['token'])

    #         if cls in cls_box_size:
    #             cls_box_size[cls].append(box[3:6])
    #         else:
    #             cls_box_size[cls] = [box[3:6]]

# print(cls_box_size.keys())



fig = plt.figure('box size')
# ax = fig.add_subplot(111, projection='3d')

# print(' '.join([str(len(cls_box_size[n])) if n in cls_box_size else str(0) for n in CLASS_NAMES]))
# print(' '.join([str(np.array(cls_box_size[n]).var(axis=0)[0]) if n in cls_box_size else str(0) for n in CLASS_NAMES]))
# print(' '.join([str(np.array(cls_box_size[n]).var(axis=0)[1]) if n in cls_box_size else str(0) for n in CLASS_NAMES]))
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