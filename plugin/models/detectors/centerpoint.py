from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors.centerpoint import CenterPoint as _CenterPoint

from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module(force=True)
class CenterPoint(MVXTwoStageDetector, _CenterPoint):

    pass
