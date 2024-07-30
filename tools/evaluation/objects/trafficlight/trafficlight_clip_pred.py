import os
import json

import numpy as np
import pandas as pd

from objects.base_objs.base_clip_obj import ClipBase
from objects.trafficlight.trafficlight_frame_obj import TrafficlightFrameObj
from objects.trafficlight.parsers.onboard_parser import parser as onboard_parser


class TrafficlightClipPred(ClipBase):
    def __init__(self, config):
        self.data_path = config["data"]["pred"]["data_path"]
        self.category = config["category"]
        self.onboard = config["onboard"]
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

    def acquire_light_info(self, extract_data, token, lights, inf_stage):
        for light in lights:
            extract_data["token"].append(token)
            extract_data["light_shape"].append(light["shape"])
            extract_data["color"].append(light.get("color"))
            score = light.get("score")
            extract_data["score"].append(score)
            xmin, ymin, xmax, ymax = light["bbox"]
            width = xmax - xmin
            height = ymax - ymin
            corners_2d = self.get_corners_2d(xmin, ymin, xmax, ymax)
            extract_data["xmin"].append(xmin)
            extract_data["ymin"].append(ymin)
            extract_data["xmax"].append(xmax)
            extract_data["ymax"].append(ymax)
            extract_data["width"].append(width)
            extract_data["height"].append(height)
            extract_data["corners_2d"].append(corners_2d)
            extract_data["inf_stage"].append(inf_stage)

    def parse(self):
        # return onboard_parser(self.data_path)
        extract_data = {"token": [],
                        "light_shape": [],
                        "color": [],
                        "score": [],
                        "xmin": [],
                        "ymin": [],
                        "xmax": [],
                        "ymax": [],
                        "width": [],
                        "height": [],
                        "corners_2d": [],
                        "inf_stage": []}
        for json_file in os.listdir(self.data_path):
            token = os.path.splitext(json_file)[0]
            ts = token.split("_")[0]
            ts = self.ts_round(ts)
            with open(os.path.join(self.data_path, json_file), 'r') as f:
                frame_data = json.load(f)
            self.acquire_light_info(extract_data, token, frame_data["det"], "det")
            # self.acquire_light_info(extract_data, ts, frame_data["classify"], "classify")
            # self.acquire_light_info(extract_data, token, frame_data["end_to_end"], "end_to_end")
        columns = ["token", "light_shape", "color", "score", "xmin", "ymin", "xmax", "ymax",
                   "width", "height", "corners_2d", "inf_stage"]
        pd_data = pd.DataFrame(extract_data, columns=columns)
        pd_data.set_index(["token"], inplace=True)
        pd_data.sort_index(inplace=True)
        return pd_data, pd_data.index.unique().tolist()

    def get_data_by_ts(self, ts):
        return self.data.loc[[ts]]

    def get_frame_obj_by_ts(self, ts):
        return TrafficlightFrameObj(self.get_data_by_ts(ts), ts)
