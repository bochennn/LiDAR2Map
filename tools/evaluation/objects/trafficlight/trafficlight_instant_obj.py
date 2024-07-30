from shapely.geometry import Polygon

from objects.base_objs.base_instant_obj import InstantBase


class TrafficlightInstantObj(InstantBase):
    def __init__(self, data, ts):
        super().__init__(data, ts)
        self.polygon = None

    def getattr(self, key, default=None):
        return getattr(self.data, key, default)

    def get_bbox(self):
        return self.getattr("bbox")

    def get_2_corners(self):
        return self.get_corners_2d()[0], self.get_corners_2d()[2]

    def get_inf_stage(self):
        return self.getattr("inf_stage")

    def get_shape(self):
        return self.getattr("light_shape")

    def get_width(self):
        return self.getattr("width")

    def get_height(self):
        return self.getattr("height")

    def get_score(self):
        return self.getattr("score")

    def get_corners_2d(self):
        return self.getattr("corners_2d")

    def get_polygon(self):
        if self.polygon is None:
            self.polygon = Polygon(self.get_corners_2d())
        return self.polygon

    def get_color(self):
        return self.getattr("color")
