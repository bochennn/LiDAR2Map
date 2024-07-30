import os
from abc import ABC
from collections import OrderedDict

import simplejson as json

from ...log_mgr import logger


class EvalBase(ABC):
    def __init__(self, config, gt_obj, pred_obj, match_obj, match_method="iou"):
        self.config = config
        self.gt_obj = gt_obj
        self.pred_obj = pred_obj
        self.base_match_obj = match_obj
        self.match_method = match_method

    def run(self):
        pass

    def get_serialize_config(self):
        config = OrderedDict()
        for key, value in self.config.items():
            try:
                json.dumps({key: value})
                config[key] = value
            except TypeError:
                config[key] = value.__str__()
        return config

    def preprocess(self):
        out_path = self.config.get("out_path")
        if out_path is not None:
            os.makedirs(out_path, exist_ok=True)
            config_out_path = os.path.join(out_path, "evaluation_config.json")
            config = self.get_serialize_config()
            with open(config_out_path, 'w') as f:
                json.dump(config, f, allow_nan=True)
            logger.info("configuration of evaluation saved to {}".format(config_out_path))

    def postprocess(self):
        pass

    def start(self):
        self.preprocess()
        self.run()
        self.postprocess()

