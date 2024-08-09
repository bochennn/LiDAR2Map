import numpy as np
from scipy.spatial import KDTree

from ..log_mgr import logger


def token_match(token_lists, verbose=False):
    diff_ts_list = list(set(token_lists[0]) - set(token_lists[1]))
    logger.info("diff ts list: {}".format(diff_ts_list))
    token_sets = [set(token_list) for token_list in token_lists]
    matched_token_list = list(set.intersection(*token_sets))
    if verbose:
        for idx, tokens in enumerate(token_lists):
            logger.info("length of list {}: {}".format(idx+1, len(tokens)))
        logger.info("length of matched list: {}".format(len(matched_token_list)))
    return list(zip(matched_token_list, matched_token_list))


def cal_freq(ts_list):
    diff = np.diff(ts_list, axis=0)
    diff = diff[(diff > 0) & (diff < 10)]
    return round(1 / diff.mean(), 2) if len(diff) > 0 else 0.


def ts_match(ts_lists, verbose=False, max_allowed_gap=0.08):
    ts_lists = [np.array(sorted(ts_list), dtype=np.float64) for ts_list in ts_lists]
    freq_seq = [cal_freq(ts_list) for ts_list in ts_lists]

    start_ts = max(ts_list[0] for ts_list in ts_lists if len(ts_list) > 0)
    end_ts = min(ts_list[-1] for ts_list in ts_lists if len(ts_list) > 0)

    base_data_index = np.argmin(freq_seq)
    base_data = np.array([ts for ts in ts_lists[base_data_index] if start_ts <= ts <= end_ts])
    base_data = base_data.reshape((len(base_data), -1))
    results = []
    for ts_idx, ts_list in enumerate(ts_lists):
        max_allowed_gap = 1 / freq_seq[ts_idx] if max_allowed_gap is None else max_allowed_gap
        ts_array = ts_list.reshape((len(ts_list), -1))
        tree = KDTree(ts_array.reshape((len(ts_array), -1)))
        dist, index = tree.query(base_data)
        ts_array = np.ravel(ts_array)
        ts_ret = ts_array[index]
        ts_ret[dist > max_allowed_gap] = np.nan
        results.append(ts_ret)
    results = np.array(results).T
    results = results[~np.isnan(results).any(axis=1)]
    if verbose:
        input_info_txt = "\n".join(["{}. length: {}, frequency: {}".format(idx+1, len(ts_list), freq_seq[idx])
                                   for idx, ts_list in enumerate(ts_lists)])
        input_info_txt = "info of input lists: \n" + input_info_txt
        output_info_txt = "info of matched list: \n" + "length: {}, frequency: {}".format(len(results),
                                                                                          cal_freq(results))
        logger.info(input_info_txt)
        logger.info(output_info_txt)
    return results


def ts_match_brute_force(ts_lists):
    ts_lists = [sorted(ts_list) for ts_list in ts_lists]
    # find intersection region of all lists
    start_ts = max(ts_list[0] for ts_list in ts_lists)
    end_ts = min(ts_list[-1] for ts_list in ts_lists)
    # select ts_list with the one with the lowest FPS as base list
    fps_list = [cal_freq(ts_list) for ts_list in ts_lists]
    fps_mask = np.argsort(fps_list)
    tree_data = [ts for ts in ts_lists[fps_mask[0]] if start_ts <= ts <= end_ts]
    matched_ts_list = [[0 for _ in range(len(tree_data))] for _ in range(len(ts_lists))]
    for base_idx, base_ts in enumerate(tree_data):
        for idx, ts_list in enumerate(ts_lists):
            max_allowed_gap = 0.5 / fps_list[idx]
            min_gap_ts = min(ts_list, key=lambda x: abs(x - base_ts))
            if abs(min_gap_ts - base_ts) <= max_allowed_gap:
                matched_ts_list[idx][base_idx] = min_gap_ts
    return matched_ts_list


if __name__ == "__main__":
    test_data1 = [np.linspace(0, 10000, num=10000, endpoint=True),
                  np.linspace(0, 10000, num=10000, endpoint=True),
                  np.linspace(0, 10000, num=10000, endpoint=True)]
    ret1 = ts_match(test_data1)
    ret2 = ts_match_brute_force(test_data1)
    # assert ret1 == ret2
