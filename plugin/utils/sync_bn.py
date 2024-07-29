import torch
from torch import nn

__all__ = ['convert_sync_batchnorm']


def convert_sync_batchnorm(module: nn.Module,
                           implementation='torch') -> nn.Module:
    """Helper function to convert all `BatchNorm` layers in the model to
    `SyncBatchNorm` (SyncBN) or `mmcv.ops.sync_bn.SyncBatchNorm` (MMSyncBN)
    layers. Adapted from `PyTorch convert sync batchnorm`_.

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.
        implementation (str): The type of `SyncBatchNorm` to convert to.

            - 'torch': convert to `torch.nn.modules.batchnorm.SyncBatchNorm`.
            - 'mmcv': convert to `mmcv.ops.sync_bn.SyncBatchNorm`.

    Returns:
        nn.Module: The converted module with `SyncBatchNorm` layers.

    .. _PyTorch convert sync batchnorm:
       https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm
    """  # noqa: E501
    module_output = module

    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if implementation == 'torch':
            SyncBatchNorm = torch.nn.modules.batchnorm.SyncBatchNorm
        elif implementation == 'mmcv':
            from mmcv.ops import SyncBatchNorm  # type: ignore
        else:
            raise ValueError('sync_bn should be "torch" or "mmcv", but got '
                             f'{implementation}')

        module_output = SyncBatchNorm(module.num_features, module.eps,
                                      module.momentum, module.affine,
                                      module.track_running_stats)

        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name,
                                 convert_sync_batchnorm(child, implementation))
    del module
    return module_output
