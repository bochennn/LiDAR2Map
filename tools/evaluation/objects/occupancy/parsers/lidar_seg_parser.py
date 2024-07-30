import os

import numpy as np
import pandas as pd

from attribute_tool import Attr


def lidar_seg_to_voxel(lidar_seg, lidar2voxel_config):
    grid_size = np.asarray(lidar2voxel_config["grid_size"])
    range_max = np.array(lidar2voxel_config["range_max"])
    range_min = np.array(lidar2voxel_config["range_min"])
    ignore_label = lidar2voxel_config["ignore_label"]

    crop_range = range_max - range_min
    intervals = crop_range / (grid_size - 1)
    points_xyz = lidar_seg[:, :3]
    labels = lidar_seg[:, 3]

    grid_index = (np.floor((np.clip(points_xyz, range_min, range_max) - range_min) / intervals)).astype(np.int)
    labeled_voxel = np.ones(grid_size, dtype=np.uint8) * ignore_label
    voxel_label_pair = np.concatenate([grid_index, labels], axis=1)
    voxel_label_pair = voxel_label_pair[np.lexsort((grid_index[:, 0], grid_index[:, 1], grid_index[:, 2])), :]

    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[voxel_label_pair[0, 3]] = 1
    previous_voxel = voxel_label_pair[0, :3]
    for idx in range(1, voxel_label_pair.shape[0]):
        current_voxel = voxel_label_pair[idx, :3]
        if not np.all(np.equal(current_voxel, previous_voxel)):
            labeled_voxel[previous_voxel[0], previous_voxel[1], previous_voxel[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            previous_voxel = current_voxel
        counter[voxel_label_pair[idx, 3]] += 1
    labeled_voxel[previous_voxel[0], previous_voxel[1], previous_voxel[2]] = np.argmax(counter)
    return labeled_voxel


def parser(data_path, lidar2voxel_config):
    extract_data = {Attr.ts: [],
                    Attr.labeled_voxel: []}
    for frame_seq, file_name in sorted(os.listdir(data_path)):
        ts = os.path.splitext(file_name)[0]
        try:
            ts = float(ts)
        except ValueError:
            ts = ts
        lidar_seg = np.fromfile(os.path.join(data_path, file_name), dtype=np.float16)
        frame_data = lidar_seg_to_voxel(lidar_seg, lidar2voxel_config)
        extract_data[Attr.ts].append(ts)
        extract_data[Attr.labeled_voxel].append(frame_data)
    pd_data = pd.DataFrame(extract_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    return pd_data, pd_data.index.unique().tolist()

