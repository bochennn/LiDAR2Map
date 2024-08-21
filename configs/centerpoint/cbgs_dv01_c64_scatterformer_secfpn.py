
_base_ = ['./cbgs_dv01_c64_second_secfpn.py']

model = dict(
    pts_middle_encoder=dict(
        type='ScatterFormer',
        output_channels=256,
        encoder_channels=((64, 64, 128), (128, 128, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, (0, 1, 1)), (0, 0)),
        out_padding=(1, 0, 0),
        attn_window_size=20,
    ),
)