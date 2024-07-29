# optimizer
lr = 0.003  # max learning rate
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),  # the momentum is change during training
    weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

momentum_config = dict(
    policy='CosineAnnealing',
    min_momentum=0.85 / 0.95
)

runner = dict(type='EpochBasedRunner', max_epochs=50)

# param_scheduler = [
#     dict(
#         type='CosineAnnealingLR',
#         T_max=8,
#         eta_min=lr * 10,
#         begin=0,
#         end=8,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=12,
#         eta_min=lr * 1e-4,
#         begin=8,
#         end=20,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=8,
#         eta_min=0.85 / 0.95,
#         begin=0,
#         end=8,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=12,
#         eta_min=1,
#         begin=8,
#         end=20,
#         by_epoch=True,
#         convert_to_iter_based=True)
# ]