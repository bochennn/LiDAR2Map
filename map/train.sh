NUM_GPU=$1

torchrun --nproc_per_node=$NUM_GPU train_lidar2map.py \
    --version v1.0-mini \
    --dataroot /mnt/data/nuscenes/nuscenes-v1.0-mini

# python3 -m torch.distributed.launch \
#     --nproc_per_node=1 \
#     --master_port=6668 \
#     train_lidar2map.py \
#     --distributed \
#     --version v1.0-mini
