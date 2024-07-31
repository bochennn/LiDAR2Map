from multiprocessing import Process, Pipe
# import torch
# from torch.multiprocessing import Process, Pipe
# torch.multiprocessing.set_start_method('spawn')
from itertools import islice
import math

# import numpy as np


def init_process(func, arg_list, arg_star=False):
    pipe_list = []
    process_list = []
    for arg in arg_list:
        conns = Pipe()
        if arg_star:
            args = (*arg, conns[1])
        else:
            args = (arg, conns[1])
        p = Process(target=func, args=args)
        pipe_list.append(conns)
        process_list.append(p)
    return pipe_list, process_list


def start_processes(pipe_list, process_list):
    res = []
    for p in process_list:
        p.start()
    for conn in pipe_list:
        res.extend(conn[0].recv())
    for process in process_list:
        process.join()
        process.close()
    return res


def arg_split(arg_list, split_num):
    it = iter(arg_list)
    chunk_size = math.ceil(len(arg_list) / split_num)
    return list(iter(lambda: tuple(islice(it, chunk_size)), ()))


def arg_split_by_bin_size(arg_list, bin_size):
    it = iter(arg_list)
    return list(iter(lambda: tuple(islice(it, bin_size)), ()))

def executor(args, child_conn):
    print("process id: {}, num of tasks: {}".format(os.getpid(), len(args)))

def multiprocess_execute(func, arg_list, processes=8, arg_star=False):
    split_arg_list = arg_split(arg_list, processes)
    pipe_list, process_list = init_process(func, split_arg_list, arg_star=arg_star)
    return start_processes(pipe_list, process_list)


if __name__ == "__main__":
    import os
    task_group = [{"seq_num": idx} for idx in range(10000)]
    task_groups = arg_split(task_group, 8)
    for idx, one_group in enumerate(task_groups):
        for task in one_group:
            task["gpu_id"] = idx

    pipe_list, process_list = init_process(executor, task_groups, False)
    start_processes(pipe_list, process_list)
