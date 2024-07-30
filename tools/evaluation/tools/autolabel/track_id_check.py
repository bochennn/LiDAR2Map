import os
import json

from collections import defaultdict

def track_id_check(data_path, out_file_path, clip_name=None):
    record = defaultdict(list)
    for file_name in sorted(os.listdir(data_path)):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_path, file_name)
            with open(file_path, 'r') as f:
                objs = json.load(f)
            for obj in objs:
                record[obj["track_id"]].append([obj["category"], file_name])

    if os.path.exists(out_file_path):
        with open(out_file_path, 'r') as f:
            failed_record = json.load(f)
    else:
        failed_record = dict()
    clip_name = os.path.basename(os.path.dirname(data_path)) if clip_name is None else clip_name
    failed_record[clip_name] = []
    for track_id, obj_list in record.items():
        prev_category = None
        prev_file_name = None
        for category, file_name in obj_list:
            if prev_category is not None and category != prev_category:
                failed_record[clip_name].append({"track_id": track_id,
                                                 "last": [prev_category, prev_file_name],
                                                 "next": [category, file_name]})
            prev_category = category
            prev_file_name = file_name
    with open(out_file_path, 'w') as f:
        json.dump(failed_record, f)


if __name__ == "__main__":
    data_path = "/mnt/data/autolabel/velocity_gt_sample_clip/clip_1693892849299/updated_fusion_0625"
    out_file_path = "/mnt/data/autolabel/velocity_gt_sample_clip/clip_1693892849299/fusion_check_0625.json"
    track_id_check(data_path, out_file_path)