#!/bin/bash

# run.sh

# source activate pytorch_p36
# cd ~/software/parlai

mkdir -p experiments

OUTPATH="./experiments/v0"

rm -rf $OUTPATH
mkdir -p $OUTPATH

python -u parlai/scripts/train_model.py \
    --task              convai2               \
    --model             bert_ranker/bi_encoder_ranker \
    --dict-file         $OUTPATH/dictionary   \
    --model-file        $OUTPATH/my_model     \
    --shuffle           true                  \
    --type-optimization all_encoder_layers    \
    --log_every_n_secs  10                    \
    --eval-batchsize    8                     \
    --data-parallel     true                  \
    --batchsize                 32               \
    --learningrate              5e-5             \
    --history-size              20               \
    --label-truncate            300              \
    --text-truncate             300              \
    --num-epochs                3                \
    --lr-scheduler              reduceonplateau  \
    --lr-scheduler-patience     1                \
    --lr-scheduler-decay        0.4              \
    --validation-every-n-epochs 0.5              \
    --warmup_updates            100              \
    --fp16                      true
