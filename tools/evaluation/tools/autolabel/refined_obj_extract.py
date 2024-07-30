import os
import math

import simplejson as json


def get_refined_obj(frame_file_path):
    refined_objs = []
    with open(frame_file_path, 'r') as f:
        objs = json.load(f)
    for obj in objs:
        if obj.get("obj_type") == "Car":
            obj["category"] = "Car"
            if "ori_score" in obj:
                obj["refined_score"] = obj["obj_score"]
                obj["obj_score"] = obj["ori_score"]
            refined_objs.append(obj)
    return refined_objs


def reformat(data_path, out_path):
    for root, dirs, names in os.walk(data_path):
        for name in names:
            if name.endswith(".json"):
                prefix = os.path.splitext(name)[0]
                if "." not in prefix:
                    prefix = prefix[:10] + "." + prefix[10:]
                    out_name = "{}.json".format(prefix)
                else:
                    out_name = name
                objs = get_refined_obj(os.path.join(root, name))
                with open(os.path.join(out_path, out_name), 'w') as f:
                    json.dump(objs, f)


def raw_refined_merge(raw_data_path, refined_data_path, out_path):
    def get_distance(obj_):
        x_ = obj_["psr"]["position"]["x"]
        y_ = obj_["psr"]["position"]["y"]
        return math.sqrt(x_ ** 2 + y_ ** 2)

    for name in os.listdir(refined_data_path):
        if name.endswith(".json"):
            raw_file_path = os.path.join(raw_data_path, name)
            with open(raw_file_path, 'r') as f:
                raw_objs = json.load(f)
            refined_file_path = os.path.join(refined_data_path, name)
            with open(refined_file_path, 'r') as f:
                refined_objs = json.load(f)
            merge_objs = []
            for obj in raw_objs:
                if get_distance(obj) > 50:
                    obj.pop("utm_position", None)
                    merge_objs.append(obj)
            for obj in refined_objs:
                if get_distance(obj) <= 50:
                    merge_objs.append(obj)
            merge_file_path = os.path.join(out_path, name)
            with open(merge_file_path, 'w') as f:
                json.dump(merge_objs, f)


if __name__ == "__main__":
    data_path = "/mnt/data/lidar_detection/results/tmp/final_ret/tta_0201_tmp"
    out_path = "/mnt/data/lidar_detection/results/tmp/final_ret/tta_0201"
    reformat(data_path, out_path)

    # raw_refined_merge("/mnt/data/lidar_detection/results/tmp/20231028_20clips/tracking",
    #                   "/mnt/data/lidar_detection/results/tmp/final_ret/raw_refined_0131_det",
    #                   "/mnt/data/lidar_detection/results/tmp/final_ret/raw_refined_merged_0131")
