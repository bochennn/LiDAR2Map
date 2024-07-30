import os
import pandas as pd

from log_mgr import logger


class EventName:
    category_error = "category_error"
    iou_error = "iou_error"
    iou_cate_error = "category_iou_error"


class AnnotationQa:
    def __init__(self, match_pair_list, out_path):
        self.match_pair_list = match_pair_list
        self.out_path = out_path
        self.events = []

    def generate_output(self):
        if len(self.events) <= 0:
            logger.info("no abnormal event found during QA task, exist")
            return
        out_path = os.path.join(self.out_path, "QA_ret")
        os.makedirs(out_path, exist_ok=True)
        out_data = pd.DataFrame(self.events)
        out_data.to_csv(os.path.join(out_path, "qa_ret.csv"), index=False)

    def run(self):
        for match in self.match_pair_list:
            if match.is_tp() or not match.gt_valid():
                continue
            event_name = None
            if match.is_category_fp_only():
                event_name = EventName.category_error
            elif match.is_iou_fp_only():
                event_name = EventName.iou_error
            elif match.is_iou_category_fp():
                event_name = EventName.iou_cate_error
            if event_name is not None:
                meta_info = {"clip_id": match.gt_instant.get_clip_id(),
                             "frame_id": match.gt_instant.get_frame_id(),
                             "ts": match.gt_instant.get_ts(),
                             "gt_category": match.gt_instant.get_category(),
                             "pred_category": match.pred_instant.get_category(),
                             "gt_obj_id": match.gt_instant.get_obj_id(),
                             "error_msg": event_name,
                             "iou": match.iou}
                self.events.append(meta_info)
        self.generate_output()
