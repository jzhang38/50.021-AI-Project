#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-1236}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --resume-from /home/peiyuan_zhang/Peiyuan/AIProj/mmediting/work_dirs/basicvsr_reds4_custom_load/iter_7000.pth \
    --seed 0 \
    --launcher pytorch ${@:3}
