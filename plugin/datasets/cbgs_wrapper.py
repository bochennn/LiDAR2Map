from logging import Logger
from typing import Dict, List

import numpy as np
from mmdet3d.datasets.builder import DATASETS, build_dataset
from mmdet3d.datasets.dataset_wrappers import CBGSDataset


@DATASETS.register_module()
class CBGSWrapper(CBGSDataset):

    def __init__(self, dataset: Dict, logger: Logger = None, **kwargs: Dict):
        if logger is not None:
            self.logger = logger
        super(CBGSWrapper, self).__init__(
            dataset=build_dataset(dataset, kwargs))

    def _num_sample_each_class(self, sample_idices: List[int]):
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in sample_idices:
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)

        if hasattr(self, 'logger'):
            for i, v in class_sample_idxs.items():
                self.logger.info(f'Class {self.CLASSES[i]} Has Num of Samples {len(v)}')

        return class_sample_idxs

    def _get_sample_indices(self):
        dataset_len = len(self.dataset)
        class_sample_idxs = self._num_sample_each_class(
            list(range(dataset_len)))

        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()}

        sample_indices = []
        frac = 1.0 / len(self.CLASSES)
        for cat_id, cls_inds in class_sample_idxs.items():
            sample_ratio = frac / class_distribution[cat_id] if class_distribution[cat_id] > 0 else 1
            # class_sample_idxs
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * sample_ratio)).tolist()

        self.logger.info('===== After CBGS =====')
        self._num_sample_each_class(sample_indices)
        self.logger.info('Dataset enlarged from {} to {}, '\
                         'upsample ratio {:.2f}'.format(
                             dataset_len, len(sample_indices), 
                             len(sample_indices) / dataset_len))

        return sample_indices



