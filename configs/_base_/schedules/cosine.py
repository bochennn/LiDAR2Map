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
    min_lr_ratio=1e-5)

momentum_config = dict(
    policy='CosineAnnealing',
    min_momentum=0.85 / 0.95)

runner = dict(type='EpochBasedRunner', max_epochs=36)