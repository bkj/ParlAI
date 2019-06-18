#!/bin/bash

# run-cross_encoder.sh

source activate pytorch_p36

mkdir -p experiments/cross_encoder
OUTPATH="./experiments/cross_encoder/v0"
rm -rf $OUTPATH && mkdir -p $OUTPATH
python -u parlai/scripts/train_model.py                  \
    --task              convai2                          \
    --model             bert_ranker/cross_encoder_ranker \
    --dict-file         $OUTPATH/dictionary              \
    --model-file        $OUTPATH/model                   \
    --shuffle           true                             \
    --log_every_n_secs  10                               \
    --eval-batchsize    16                               \
    --data-parallel     true                             \
    --type-optimization         all_encoder_layers       \
    --batchsize                 16                       \
    --learningrate              5e-5                     \
    --history-size              20                       \
    --label-truncate            72                       \
    --text-truncate             360                      \
    --lr-scheduler              reduceonplateau          \
    --lr-scheduler-patience     0                        \
    --lr-scheduler-decay        0.4                      \
    --validation-every-n-epochs 0.5                      \
    --warmup_updates            1000                     \
    --fp16                      true

# "batch size fixed at 16 and provide as negatives random samples from the training set"
#   convai2 provides 19 negative samples
# 1000 warmup iterations for cross-encoder


