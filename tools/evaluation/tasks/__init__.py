import pprint

from .eval_tasks import ObstacleEval, TrafficlightEval, ObstacleEvalPose, ObstacleEvalCluster
from config import ConfigParser
from log_mgr import logger


class TaskBuilder:
    obj_table = {"3d_object": ObstacleEval,
                 "trafficlight": TrafficlightEval,
                 "obj_pose": ObstacleEvalPose,
                 "cluster_object": ObstacleEvalCluster}

    @staticmethod
    def build_task(args):
        task_type = args.task_type
        config = ConfigParser.parse(task_type)
        task_class = TaskBuilder.obj_table.get(task_type)
        config.update(vars(args))
        logger.info(pprint.pformat(config))
        if task_class:
            return task_class(config)
        else:
            raise NotImplementedError("no task class was implemented for {}".format(task_type))
