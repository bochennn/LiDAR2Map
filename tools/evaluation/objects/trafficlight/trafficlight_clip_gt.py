import json
import os

# import numpy as np
import pandas as pd

from ..base_objs.base_clip_obj import ClipBase
from .parsers.attribute_tool import TrafficLightEnum
from .trafficlight_frame_obj import TrafficlightFrameObj


class TrafficlightClipGt(ClipBase):
    def __init__(self, config):
        self.data_path = config["data"]["gt"]["data_path"]
        self.category = config["category"]
        self.category_map = {idx: name for idx, name in enumerate(self.category)}
        self.imgH = config["imgH"]
        self.imgW = config["imgW"]
        self.frame_objs = dict()
        super().__init__()

    @staticmethod
    def get_corners_2d(xmin, ymin, xmax, ymax):
        p1 = [xmin, ymin]
        p2 = [xmax, ymin]
        p3 = [xmax, ymax]
        p4 = [xmin, ymax]
        return [p1, p2, p3, p4]

    def parse(self):
        extract_data = {"ts": [],
                        "light_shape": [],
                        "color": [],
                        "xmin": [],
                        "ymin": [],
                        "xmax": [],
                        "ymax": [],
                        "width": [],
                        "height": [],
                        "corners_2d": []}
        for json_file in os.listdir(self.data_path):
            token = os.path.splitext(json_file)[0]
            ts = token.split("_")[0]
            ts = self.ts_round(ts)
            with open(os.path.join(self.data_path, json_file), 'r') as f:
                frame_data = json.load(f)
            if isinstance(frame_data, dict):
                lights = [frame_data]
            else:
                lights = frame_data
            for light in lights:
                if light.get("bbox") is None:
                    continue
                extract_data["ts"].append(token)
                extract_data["light_shape"].append(light["shape"])
                color = TrafficLightEnum.super_color_enum.get(light["color"])
                extract_data["color"].append(color)
                xmin, ymin, xmax, ymax = light["bbox"]
                width = xmax - xmin
                height = ymax - ymin
                corner_2d = self.get_corners_2d(xmin, ymin, xmax, ymax)
                extract_data["xmin"].append(xmin)
                extract_data["ymin"].append(ymin)
                extract_data["xmax"].append(xmax)
                extract_data["ymax"].append(ymax)
                extract_data["width"].append(width)
                extract_data["height"].append(height)
                extract_data["corners_2d"].append(corner_2d)
        columns = ["ts", "light_shape", "color", "xmin", "ymin", "xmax", "ymax",
                   "width", "height", "corners_2d"]
        pd_data = pd.DataFrame(extract_data, columns=columns)
        pd_data.set_index(["ts"], inplace=True)
        pd_data.sort_index(inplace=True)
        return pd_data, pd_data.index.unique().tolist()

    def get_data_by_ts(self, ts):
        return self.data.loc[[ts]]

    def get_frame_obj_by_ts(self, ts):
        return TrafficlightFrameObj(self.get_data_by_ts(ts), ts)
