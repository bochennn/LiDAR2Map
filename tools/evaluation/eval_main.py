import argparse

from tasks import TaskBuilder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="obj_pose",
                        help="available option: 3d_object, trafficlight, obj_pose")
    args = parser.parse_args()
    eval_task = TaskBuilder.build_task(args)
    eval_task.start()

