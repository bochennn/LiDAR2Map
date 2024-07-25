from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.loading import \
    LoadAnnotations3D as _LoadAnnotations3D


@PIPELINES.register_module(force=True)
class LoadAnnotations3D(_LoadAnnotations3D):

    def _load_semantic_seg(self, results):
        gt_semantic_seg = results['ann_info']['gt_semantic_seg']

        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results
