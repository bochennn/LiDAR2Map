from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from sys import modules


def process_worker_wrapper(args):
    return args[0](**args[1])

def multi_process_thread(func, kwargs, nprocess=cpu_count(), pool_func='Pool', map_func='map'):
    """
    Create process / thread pool

    @param func: process / thread function
    @param kwargs: process / thread keyword argument list
    @param nprocess: number of process / thread
    @param pool_func: [Pool(multi process), ThreadPoolExecutor(multi thread)]
    @param map_func: [map, imap]
    """
    assert nprocess > 0
    if nprocess > 1:
        with getattr(modules[__name__], pool_func)(nprocess) as pool:
            return list(getattr(pool, map_func)(
                process_worker_wrapper, [(func, kwags) for kwags in kwargs]))
    else:
        return [func(**kwags) for kwags in kwargs]
