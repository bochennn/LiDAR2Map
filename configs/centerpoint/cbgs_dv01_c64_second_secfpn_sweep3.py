from configs.centerpoint.hv01_second_secfpn import train_pipeline, eval_pipeline

_base_ = ['./cbgs_dv01_c64_second_secfpn.py']

search = lambda lst, val: next(d for d in lst if d.get('type') == val)

search(train_pipeline, 'LoadPointsFromFile').update(
    type='LoadPointsFromMultiSweeps',
    sweep_indices=[-2, -1, 0], load_dim=5, use_dim=[0, 1, 2, 3, 4])

search(eval_pipeline, 'LoadPointsFromFile').update(
    type='LoadPointsFromMultiSweeps',
    sweep_indices=[-2, -1, 0], load_dim=5, use_dim=[0, 1, 2, 3, 4])

model = dict(
    pts_middle_encoder=dict(in_channels=5)
)

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=eval_pipeline),
    test=dict(pipeline=eval_pipeline),
)