import json

import pandas as pd

from ....utils.bbox_ops import get_2d_corners
from .attribute_tool import Attr


def parser(data_path):
    """
    For corner case mining task
    Args:
        data_path: Absolute path of json file, which contains the 2D detection results from images.

    Returns:
        DataFrame which saves each single obj as one row.
    """
    extract_data = {Attr.frame_seq: [],
                    Attr.sample_id: [],
                    Attr.img_xmin: [],
                    Attr.img_xmax: [],
                    Attr.img_ymin: [],
                    Attr.img_ymax: [],
                    Attr.img_width: [],
                    Attr.img_height: [],
                    Attr.corners_2d: [],
                    Attr.bbox_img: [],
                    Attr.category: [],
                    Attr.score: []}
    with open(data_path, 'r') as f:
        data = json.load(f)
    for frame_seq, (sample_id, objs) in enumerate(data.items()):
        for obj in objs:
            extract_data[Attr.frame_seq].append(frame_seq)
            extract_data[Attr.sample_id].append(sample_id)
            xmin, ymin, xmax, ymax = obj["bbox"]
            extract_data[Attr.img_xmin].append(xmin)
            extract_data[Attr.img_xmax].append(xmax)
            extract_data[Attr.img_ymin].append(ymin)
            extract_data[Attr.img_ymax].append(ymax)
            extract_data[Attr.img_width].append(xmax - xmin)
            extract_data[Attr.img_height].append(ymax - ymin)
            extract_data[Attr.corners_2d].append(get_2d_corners(xmin, ymin, xmax, ymax))
            extract_data[Attr.bbox_img].append(obj["bbox"])
            extract_data[Attr.category].append(obj["category"])
            extract_data[Attr.score].append(obj["score"])
    pd_data = pd.DataFrame(extract_data)
    pd_data.set_index(Attr.sample_id, inplace=True)
    return pd_data, pd_data.index.unique().tolist()
