from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.loading import \
    LoadAnnotations3D as _LoadAnnotations3D


@PIPELINES.register_module(force=True)
class LoadAnnotations3D(_LoadAnnotations3D):
    def __init__(self, **kwargs):
        super(LoadAnnotations3D, self).__init__(**kwargs)

    # def __call__(self, results):
    #     print(xx)

    #     return