import os
import json
from collections import OrderedDict

import numpy as np
import pandas as pd

from objects.trafficlight.parsers.attribute_tool import Attr, TrafficLightEnum


def get_corners_2d(xmin, ymin, xmax, ymax):
    return np.array([[xmin, ymin],
                     [xmax, ymin],
                     [xmax, ymax],
                     [xmin, ymax]])


def parser(data_path):
    columns = [Attr.ts, Attr.light_shape, Attr.color, Attr.score, Attr.xmin, Attr.ymin, Attr.xmax, Attr.ymax,
               Attr.width, Attr.height, Attr.corners_2d, Attr.roi_xmin, Attr.roi_ymin, Attr.roi_xmax, Attr.roi_ymax,
               Attr.roi_width, Attr.roi_height, Attr.infer_stage]
    extract_data = OrderedDict({attr_name: [] for attr_name in columns})
    for file_name in sorted(os.listdir(data_path)):
        if file_name.endswith(".json"):
            with open(os.path.join(data_path, file_name), 'r') as f:
                frame_data = json.load(f)
            for light in frame_data:
                light_shape = "vertical_light" if "light_shape" not in light else light["light_shape"]
                light_color = TrafficLightEnum.onboard_color_enum[light["color_id"]]
                extract_data[Attr.ts].append(light["ts"])
                extract_data[Attr.light_shape].append(light_shape)
                extract_data[Attr.color].append(light_color)
                extract_data[Attr.score].append(light["score"])
                extract_data[Attr.xmin].append(light["xmin"])
                extract_data[Attr.ymin].append(light["ymin"])
                extract_data[Attr.xmax].append(light["xmax"])
                extract_data[Attr.ymax].append(light["ymax"])
                extract_data[Attr.width].append(light["width"])
                extract_data[Attr.height].append(light["height"])
                extract_data[Attr.corners_2d].append(get_corners_2d(light["xmin"],
                                                                    light["ymin"],
                                                                    light["xmax"],
                                                                    light["ymax"]))
                extract_data[Attr.roi_xmin].append(light["roi_xmin"])
                extract_data[Attr.roi_ymin].append(light["roi_ymin"])
                extract_data[Attr.roi_xmax].append(light["roi_xmax"])
                extract_data[Attr.roi_ymax].append(light["roi_ymax"])
                extract_data[Attr.roi_width].append(light["roi_width"])
                extract_data[Attr.roi_height].append(light["roi_height"])
                extract_data[Attr.infer_stage].append(["detection", "end_to_end"])
    pd_data = pd.DataFrame(extract_data, columns=columns)
    pd_data.set_index([Attr.ts], inplace=True)
    pd_data.sort_index(inplace=True)
    return pd_data, pd_data.index.unique().tolist()
