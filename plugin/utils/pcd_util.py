from pathlib import Path
from typing import List

import numpy as np


def read_points_pcd(
    filepath: str,
    fields: List = ['x', 'y', 'z', 'intensity', 'timestamp', 'ring']
) -> np.ndarray:
    # pip install git+https//github.com/bochennn/pypcd.git
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
