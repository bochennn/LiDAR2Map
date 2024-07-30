from ..base_objs.base_frame_obj import FrameBase
from .trafficlight_instant_obj import TrafficlightInstantObj


class TrafficlightFrameObj(FrameBase):
    def __init__(self, data, ts):
        super().__init__(data, ts)
        self.instant_objs = []
        self.initialized = False
        self.empty = None

    def get_det_instant_objects(self):
        return [instant for instant in self.get_instant_objects() if "det" == instant.get_inf_stage()]

    def get_classify_instant_objects(self):
        return [instant for instant in self.get_instant_objects() if "classify" == instant.get_inf_stage()]

    def get_e2e_instant_objects(self):
        return [instant for instant in self.get_instant_objects() if "end_to_end" == instant.get_inf_stage()]

    def get_instant_objects(self):
        if not self.initialized:
            for ts, row_data in self.data.iterrows():
                if getattr(row_data, "light_shape") is None:
                    continue
                self.instant_objs.append(TrafficlightInstantObj(row_data, ts))
            self.initialized = True
        return self.instant_objs

    def is_empty(self):
        if self.empty is None:
            self.empty = self.data.isna()["light_shape"].sum() > 0
        return self.empty
