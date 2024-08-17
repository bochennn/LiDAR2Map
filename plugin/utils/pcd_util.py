from pathlib import Path
from typing import List

import mmcv
import numpy as np


def load_points(pts_filename: str, load_dim: int = 5):
    """Private function to load point clouds data.

    Args:
        pts_filename (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    file_client = mmcv.FileClient(backend='disk')

    try:
        if pts_filename.endswith('.pcd'):
            points = read_points_pcd(pts_filename)[:, :load_dim]
            points[:, 3] /= 255.
        elif pts_filename.endswith('.bin'):
            pts_bytes = file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32).reshape(-1, load_dim)
        else:
            raise NotImplementedError(pts_filename)

    except ConnectionError:
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)

    return points


def read_points_pcd(
    filepath: str,
    fields: List = ['x', 'y', 'z', 'intensity', 'timestamp', 'ring']
) -> np.ndarray:
    # pip3 install git+https://github.com/bochennn/pypcd.git
    from pypcd import pypcd
    pcd = pypcd.point_cloud_from_path(filepath)

    pcd_metadata = pcd.get_metadata()
    data_length = len([_ for _ in pcd_metadata['fields'] if _ != '_' and _ in fields])
    data_np = np.zeros((pcd_metadata['width'], data_length), dtype=float)

    for name in pcd_metadata['fields']:
        data_np[:, fields.index(name)] = pcd.pc_data[name]

    return data_np


def write_points_pcd(
    filepath: str,
    output: np.ndarray,
    fields: List = ['x', 'y', 'z', 'intensity', 'timestamp', 'ring'],
    viewpoint: List = None,
    verbose: bool = False
):
    _filepath = Path(filepath) if not isinstance(filepath, Path) else filepath

    from pypcd.pypcd import (make_xyz_point_cloud,
                             save_point_cloud_bin_compressed)
    metadata_map = {
        'x': {'type': 'F', 'size': 4},
        'y': {'type': 'F', 'size': 4},
        'z': {'type': 'F', 'size': 4},
        'intensity': {'type': 'U', 'size': 4},
        'timestamp': {'type': 'F', 'size': 8},
        'ring': {'type': 'U', 'size': 2}
    }
    pts_metadata = {
        'fields': fields,
        'size': [metadata_map[fname]['size'] for fname in fields],
        'type': [metadata_map[fname]['type'] for fname in fields],
        'count': [1 for _ in fields]
    }

    if viewpoint is not None:
        pts_metadata.update(viewpoint=viewpoint)

    pts_obj = make_xyz_point_cloud(output, pts_metadata)
    save_point_cloud_bin_compressed(pts_obj, _filepath)
    if verbose:
        print('write points', _filepath)
