from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.formating import DefaultFormatBundle3D as _DefaultFormatBundle3D

from mmcv.parallel import DataContainer as DC

@PIPELINES.register_module(force=True)
class DefaultFormatBundle3D(_DefaultFormatBundle3D):

    def __call__(self, results):
        results = super(DefaultFormatBundle3D, self).__call__(results)

        if 'gt_semantic_seg' in results and \
            isinstance(results['gt_semantic_seg'], DC):
            results['gt_semantic_seg'] = DC(
                results['gt_semantic_seg'].data.squeeze(0), stack=True)

        return results