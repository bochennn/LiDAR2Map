CODE_ROOT=$(realpath $(dirname $0)/../..)
GPUS=$1

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=10086
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

cd $CODE_ROOT
OMP_NUM_THREADS=1 PYTHONPATH=$CODE_ROOT \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    --launcher pytorch ${@:2}
    # --version v1.0-trainval \
    # --dataroot /data/sfs_turbo/nuscenes \
    # --bsz 2