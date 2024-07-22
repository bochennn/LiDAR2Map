GPUS=$1

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

OMP_NUM_THREADS=1 torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train_lidar2map.py \
    --launcher pytorch \
    --version v1.0-trainval \
    --dataroot /data/sfs_turbo/nuscenes \
    --bsz 2

# python3 -m torch.distributed.launch \
#     --nproc_per_node=1 \
#     --master_port=6668 \
#     train_lidar2map.py \
#     --distributed \
#     --version v1.0-mini
