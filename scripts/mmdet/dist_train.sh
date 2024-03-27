#!/usr/bin/env bash

CONFIG=$1
DIR=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=2 \
    --master_port=$PORT \
    /mmdetection/tools/train.py \
    $CONFIG \
    --work-dir $DIR \
    --launcher pytorch ${@:3}
