from objects.base_objs.base_match_obj import BaseMatch


class TrafficlightMatchObj(BaseMatch):
    def __init__(self, gt_instant, pred_instant, iou, threshold_map):
        super().__init__(gt_instant, pred_instant, iou)
        self.iou = iou
        self.threshold_map = threshold_map
        self.tp_flag = None

    def iou_match(self):
        return self.iou > self.threshold_map.get(self.gt_instant.get_shape())

    def category_match(self):
        return True
        return self.gt_instant.get_shape() == self.pred_instant.get_shape()

    def is_tp(self):
        if self.tp_flag is None:
            self.tp_flag = self.match_valid() and self.iou_match() and self.category_match()
        return self.tp_flag

    def is_color_tp(self):
        return (self.match_valid() and self.gt_instant.get_color() == self.pred_instant.get_color()) or \
            (not self.gt_valid() and self.pred_instant.get_color() == "unknown")

    def is_color_fp(self):
        return not self.is_color_tp() and self.pred_valid()

    def is_color_fn(self):
        return not self.is_color_tp() and self.gt_valid()

    def is_fp(self):
        return self.pred_valid() and not self.is_tp()

    def is_fn(self):
        return self.gt_valid() and not self.is_tp()

    def get_score(self):
        score = 0.0
        if self.pred_instant:
            score = self.pred_instant.get_score()
        return score
