import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np

info_list, cls_box_size = [], dict()

for info_path in Path('data/zdrive').glob('*.pkl'):
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)
    info_list.extend(info_dict['infos'])

for info in info_list:
    for box, cls in zip(info['gt_boxes'], info['gt_names']):
        if cls in cls_box_size:
            cls_box_size[cls].append(box[3:6])
        else:
            cls_box_size[cls] = [box[3:6]]

fig = plt.figure('box whl')
ax = fig.add_subplot(111, projection='3d')

for cls, cls_size in cls_box_size.items():
    _cls_size = np.array(cls_size)
    print(cls, _cls_size.mean(axis=0))
    ax.scatter(_cls_size[:, 0], _cls_size[:, 1], _cls_size[:, 2], label=cls)

plt.legend()
plt.show()