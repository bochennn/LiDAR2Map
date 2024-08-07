from typing import List

from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.transforms_3d import \
    PointsRangeFilter as _PointsRangeFilter


@PIPELINES.register_module(force=True)
class PointsRangeFilter(_PointsRangeFilter):

    def __init__(self, point_cloud_range: List, min_point_cloud_range: List):
        super(PointsRangeFilter, self).__init__(
            point_cloud_range=point_cloud_range)
        self.min_point_cloud_range = min_point_cloud_range

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        points_mask &= ~points.in_range_bev(self.min_point_cloud_range)

        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict


