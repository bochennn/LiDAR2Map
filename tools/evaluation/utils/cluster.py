"""
pcd加载成shape=(n, 3)的np.array, 调用下面的函数 cluster_pcd 快速聚类找 cluster中心.
"""

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture

from log_mgr import logger


def gaussian_cluster(datas, max_cluster=7):
    lowest_bic = np.infty
    bic = []
    max_cluster = max(min(len(datas), max_cluster), len(datas) // 7)
    n_components_range = range(1, max_cluster)
    cv_types = ["spherical", ]  # "tied", "diag", "full"]
    best_gmm = None
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            try:
                gmm.fit(datas)
            except ValueError as e:
                continue
            bic.append(gmm.bic(datas))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    assert best_gmm is not None
    labels = best_gmm.predict(datas)
    return labels


def run_meanshift_clustering(datas, dx_thr=2, dy_thr=2, dz_thr=3):
    """如果候选点群, 集中在一个3d-box size=(dx_thr, dy_thr, dz_thr)内, 直接看成一个cluster输出。
    否则, 采用MeanShift聚类再输出. 适用于datas规模小, 暂定15以内。

    Args:
        datas (np.array), with shape (n, 3)
        (dx_thr, dy_thr, dz_thr), 3d-box ranging.
    Return:
        clusters (list), each item is a np.array with shape (k, 3)
        centers (list), each item is a np.array with shape (3, )
    """
    delta_xyz = np.max(datas, axis=0) - np.min(datas, axis=0)
    if delta_xyz[0] < dx_thr and delta_xyz[1] < dy_thr and delta_xyz[2] < dz_thr:
        clusters, centers = [datas], [np.mean(datas, axis=0)]
    else:
        try:
            cluster_pred = MeanShift().fit(datas)
            pred_labels = cluster_pred.labels_
        except ValueError as e:
            pred_labels = np.zeros_like(datas[:, 0], dtype=np.int32)
            logger.error(f"Directly output datas as a cluster since ValueError in MeanShift().fit(datas): {e}")
            logger.error(datas)
        cluster_ids = np.unique(pred_labels).tolist()
        cluster_masks = [(pred_labels == cluster_id) for cluster_id in cluster_ids]
        clusters, centers = [], []
        for _m in cluster_masks:
            _cluster = datas[_m]
            _delta_xyz = np.max(_cluster, axis=0) - np.min(_cluster, axis=0)
            if _delta_xyz[0] < dx_thr and _delta_xyz[1] < dy_thr and _delta_xyz[2] < dz_thr:
                clusters.append(_cluster)
                centers.append(np.mean(_cluster, axis=0))
            elif np.sum(_m).item() == _m.shape[0]:  # all members belong to one cluster
                clusters.append(_cluster)
                centers.append(np.mean(_cluster, axis=0))
            else:
                _clusters, _centers = run_meanshift_clustering(_cluster, dx_thr, dy_thr, dz_thr)
                clusters.extend(_clusters)
                centers.extend(_centers)
    return clusters, centers


def run_gaussian_clustering(datas, dx_thr=2, dy_thr=2, dz_thr=3):
    """如果候选点群, 集中在一个3d-box size=(dx_thr, dy_thr, dz_thr)内, 直接看成一个cluster输出。
    否则, 采用Gaussian mixture models输出. 适用于datas规模偏大, 暂定15以上。每个cluster过滤掉距
    离中心太远的点直接快速输出center, 不再对每个cluster递归聚类。

    Args:
        datas (np.array), with shape (n, 3)
        (dx_thr, dy_thr, dz_thr), 3d-box ranging.
    Return:
        clusters (list), each item is a np.array with shape (k, 3)
        centers (list), each item is a np.array with shape (3, )
    """
    delta_xyz = np.max(datas, axis=0) - np.min(datas, axis=0)
    if delta_xyz[0] < dx_thr and delta_xyz[1] < dy_thr and delta_xyz[2] < dz_thr:
        clusters, centers = [datas], [np.mean(datas, axis=0)]
    else:
        num_datas = len(datas)
        pred_labels = gaussian_cluster(datas)
        cluster_ids = np.unique(pred_labels).tolist()
        cluster_masks = [(pred_labels == cluster_id) for cluster_id in cluster_ids]
        clusters, centers = [], []
        for _m in cluster_masks:
            _cluster = datas[_m]
            _median = np.median(_cluster, axis=0)
            _delta_xyz = np.max(_cluster, axis=0) - np.min(_cluster, axis=0)
            if (_delta_xyz[0] >= dx_thr) or (_delta_xyz[1] >= dy_thr) or (_delta_xyz[2] < dz_thr):
                offsets2center = np.abs(_cluster - _median)
                filter_mask = ((offsets2center[:, 0] < dx_thr) &
                               (offsets2center[:, 1] < dy_thr) &
                               (offsets2center[:, 2] < dz_thr))
                _cluster = _cluster[filter_mask]
            if _cluster.shape[0] == 0:
                continue
            clusters.append(_cluster)
            centers.append(np.mean(_cluster, axis=0))
    return clusters, centers


def merge_close_clusters(clusters, centers, dx_thr=2, dy_thr=2, dz_thr=3):
    """如果候选clusters, 他们的cluster center靠近处于一个3d-box size=(dx_thr, dy_thr, dz_thr)内,
    直接合并成一个cluster输出。
    Args:
        clusters (list), each item is a np.array with shape (k, 3)
        centers (list), each item is a np.array with shape (3, )
        datas (np.array), with shape (n, 3)
        (dx_thr, dy_thr, dz_thr), 3d-box ranging.
    Return:
        cluster_infos (list), like [(cluster, center), ... ],
            each cluster is a np.array with shape (k, 3)
            each center is a np.array with shape (3, )
    """
    cluster_infos = [(cluster, center) for cluster, center in zip(clusters, centers)]
    cluster_infos.sort(key=lambda ii: len(ii[0]), reverse=True)
    ignore_ids = set()
    N = len(cluster_infos)
    for i in range(N):
        if i in ignore_ids:
            continue
        anchor_cluster, anchor_center = cluster_infos[i]
        for j in range(i + 1, N):
            if j in ignore_ids:
                continue
            refer_cluster, refer_center = cluster_infos[j]
            delta_xyz = np.abs(refer_center - anchor_center)
            if delta_xyz[0] < dx_thr / 2 and delta_xyz[1] < dy_thr / 2 and delta_xyz[2] < dz_thr / 2:
                ignore_ids.add(j)
                anchor_cluster = np.concatenate([anchor_cluster, refer_cluster], axis=0)
                anchor_center = np.mean(anchor_cluster, axis=0)
        cluster_infos[i] = (anchor_cluster, anchor_center)
    cluster_infos = [cluster_infos[i] for i in range(N) if i not in ignore_ids]
    cluster_infos.sort(key=lambda ii: len(ii[0]), reverse=True)
    return cluster_infos


def cluster_pcd(datas, dx_thr=2, dy_thr=2, dz_thr=3):
    """如果候选点群, 集中在一个3d-box size=(dx_thr, dy_thr, dz_thr)内, 直接看成一个cluster输出。
    否则, 采用MeanShift聚类再输出. 如果候选clusters, 他们的cluster center靠近处于一个3d-box size=
    (dx_thr, dy_thr, dz_thr)内, 直接合并成一个cluster输出.

    Args:
        datas (np.array), with shape (n, 3)
        (dx_thr, dy_thr, dz_thr), 3d-box ranging.

    Return:
        cluster_infos (list), like [(cluster, center), ... ],
            each cluster is a np.array with shape (k, 3)
            each center is a np.array with shape (3, )
    """
    if len(datas) < 15:
        clusters, centers = run_meanshift_clustering(datas, dx_thr, dy_thr, dz_thr)
    else:
        clusters, centers = run_gaussian_clustering(datas, dx_thr, dy_thr, dz_thr)
        if len(clusters) == 0:
            clusters, centers = run_meanshift_clustering(datas, dx_thr, dy_thr, dz_thr)
    cluster_infos = merge_close_clusters(clusters, centers, dx_thr, dy_thr, dz_thr)
    return cluster_infos

