import os

# from tasks.corner_case_task.long_distance_case import LongDistChecker
from tasks.corner_case_task.heading_revere_case import HeadingReverseChecker
from tasks.corner_case_task.size_inconsistent_case import SizeConsistentChecker
from tasks.corner_case_task.heading_revere_case import HeadingReverseChecker
from config import ConfigParser


if __name__ == "__main__":
    config = ConfigParser.parse("/home/wuchuanpan/PycharmProjects/experiment/config/config_files/corner_case.yaml")
    size_checker = SizeConsistentChecker(config)
    heading_checker = HeadingReverseChecker(config)
    heading_checker.start()
    #
    # import copy
    # import os
    # from pprint import pprint
    # import pickle
    # root_path = "/mnt/data/20230717/onboard_ret_full_two_stage"
    # out_path = "/mnt/data/20230717/corner_case_heading_reverse"
    # total_ret = []
    # for clip_name in sorted(os.listdir(root_path)):
    #     if clip_name.startswith("clip_"):
    #         one_config = copy.deepcopy(config)
    #         one_config["data"]["pred"]["data_path"] = os.path.join(root_path, clip_name, "detection")
    #         one_config["data"]["tracking"]["data_path"] = os.path.join(root_path, clip_name, "tracking")
    #         ret = HeadingReverseChecker(one_config).start()
    #         if ret:
    #             total_ret.append((clip_name, ret))
    # with open(os.path.join(out_path, "heading_reverse.pkl"), 'wb') as f:
    #     pickle.dump(total_ret, f)
    # for clip_name, ret in total_ret:
    #     print("clip name: {}".format(clip_name))
    #     pprint(ret)
