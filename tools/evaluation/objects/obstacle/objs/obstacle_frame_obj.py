import os
from collections import defaultdict
# from decimal import Decimal

import simplejson as json

from ....utils.bbox_ops import get_3d_boxes_in_cam_view, project_3dbox_to_2dbox
from ...base_objs.base_frame_obj import FrameBase
from ..parsers.attribute_tool import Attr
from .obstacle_instant_obj import ObstacleInstantObj


class ObstacleFrameObj(FrameBase):
    def __init__(self, data, ts, score_thr=0.3):
        super().__init__(data, ts)
        self.instant_objs = []
        self.uuid_record = None
        self.initialized = False
        self.score_thr = score_thr
        # self.add_key_target_tag()

    def add_key_target_tag(self):
        if not self.is_empty():
            all_instants = self.get_instant_objects()
            for instant in all_instants:
                instant.setattr(Attr.is_key_obj, False)
            potential_instants = [instant for instant in all_instants if (-1.75 < instant.get_lidar_pos_y() < 1.75)
                                  and (0 < instant.get_lidar_pos_x() < 80)]
            if potential_instants:
                target_instant = min(potential_instants, key=lambda x: x.get_lidar_pos_x())
                target_instant.setattr(Attr.is_key_obj, True)

    def get_instant_objects(self):
        if not self.initialized:
            for ts, row_data in self.data.iterrows():
                if hasattr(row_data, Attr.score) and getattr(row_data, Attr.score) < self.score_thr:
                    continue
                self.instant_objs.append(ObstacleInstantObj(row_data, ts))
            self.initialized = True
        return self.instant_objs

    def get_instant_objects_by_category(self, target_category):
        return [instant for instant in self.get_instant_objects() if instant.get_category() == target_category]

    def get_instant_objects_by_uuid(self, uuid):
        if self.uuid_record is None:
            self.uuid_record = {instant.get_uuid(): instant for instant in self.get_instant_objects()}
        return self.uuid_record[uuid]

    def groupby_category(self):
        record_by_category = defaultdict(list)
        for instant in self.get_instant_objects():
            record_by_category[instant.get_category()].append(instant)
        for category, instants in record_by_category.items():
            return category, instants

    def is_empty(self):
        return self.data.isna()[Attr.category].sum() > 0

    @staticmethod
    def project_3d_to_2d(instants, cam_extrinsic, cam_intrinsic, img_mat=None):
        xywh_box_list, img_mat = project_3dbox_to_2dbox(instants,
                                                        cam_extrinsic,
                                                        cam_intrinsic,
                                                        img_mat)
        bbox_2d_list = []
        x, y, w, h = 0, 1, 2, 3
        for xywh_box, _ in xywh_box_list:
            bbox_2d_list.append([xywh_box[x],
                                 xywh_box[y],
                                 xywh_box[x] + xywh_box[w],
                                 xywh_box[y] + xywh_box[h]])
        return bbox_2d_list, img_mat

    def get_instants_in_cam_view(self, cam_extrinsic, cam_intrinsic, img_w, img_h):
        return get_3d_boxes_in_cam_view(self.get_instant_objects(),
                                        cam_extrinsic,
                                        cam_intrinsic,
                                        img_w,
                                        img_h)

    def to_visual_json(self, out_path):
        out_objs = []
        ts_str = "{:6f}".format(self.get_ts())
        file_name = "{}.json".format(ts_str)
        out_file_path = os.path.join(out_path, file_name)
        for instant in self.get_instant_objects():
            out_objs.append(instant.get_visual_json_obj())
        with open(out_file_path, 'w') as f:
            json.dump(out_objs, f, use_decimal=True)

    def to_visual_json_with_velocity(self, out_path):
        out_objs = []
        ts_str = "{:6f}".format(self.get_ts())
        file_name = "{}.json".format(ts_str)
        out_file_path = os.path.join(out_path, file_name)
        for instant in self.get_instant_objects():
            out_objs.append(instant.get_visual_json_obj_with_velocity())
        with open(out_file_path, 'w') as f:
            json.dump(out_objs, f, use_decimal=True)
