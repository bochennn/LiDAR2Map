import os
import pickle

import pandas as pd

from .attribute_tool import Attr


def parser(data_path):
    extract_data = {Attr.ts: [],
                    Attr.labeled_voxel: []}
    for frame_seq, file_name in sorted(os.listdir(data_path)):
        with open(os.path.join(data_path, file_name), 'rb') as f:
            frame_data = pickle.load(f)
        ts = os.path.splitext(file_name)[0]
        try:
            ts = float(ts)
        except ValueError:
            ts = ts
        extract_data[Attr.ts].append(ts)
        extract_data[Attr.labeled_voxel].append(frame_data)
    pd_data = pd.DataFrame(extract_data)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    return pd_data, pd_data.index.unique().tolist()
