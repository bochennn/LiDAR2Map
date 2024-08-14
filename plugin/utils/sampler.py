import torch
from mmdet3d.core.bbox.samplers import \
    IoUNegPiecewiseSampler as _IoUNegPiecewiseSampler
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.samplers.sampling_result import \
    SamplingResult as _SamplingResult

__init__ = ['IoUNegPiecewiseSampler']


class SamplingResult(_SamplingResult):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 7)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 7)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long(), :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @classmethod
    def random(cls, rng=None, **kwargs):
        raise NotImplementedError('3D random sampler')


@BBOX_SAMPLERS.register_module(force=True)
class IoUNegPiecewiseSampler(_IoUNegPiecewiseSampler):

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (torch.Tensor): Boxes to be sampled from.
            gt_bboxes (torch.Tensor): Ground truth bboxes.
            gt_labels (torch.Tensor, optional): Class labels of ground truth
                bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.bool)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.bool)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        if self.return_iou:
            # PartA2 needs iou score to regression.
            sampling_result.iou = assign_result.max_overlaps[torch.cat(
                [pos_inds, neg_inds])]
            sampling_result.iou.detach_()

        return sampling_result
