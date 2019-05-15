#!/bin/bash

# run.sh

# source activate pytorch_p36
# cd ~/software/parlai

mkdir -p experiments

OUTPATH="./experiments/v0"

rm -rf $OUTPATH
mkdir -p $OUTPATH

python -u parlai/scripts/train_model.py \
    --task              convai2                       \
    --model             bert_ranker/bi_encoder_ranker \
    --dict-file         $OUTPATH/dictionary           \
    --model-file        $OUTPATH/my_model             \
    --shuffle           true                          \
    --log_every_n_secs  10                            \
    --eval-batchsize    8                             \
    --data-parallel     true                          \
    --type-optimization         all_encoder_layers \
    --batchsize                 32                 \
    --learningrate              5e-5               \
    --history-size              20                 \
    --label-truncate            72                 \
    --text-truncate             360                \
    --lr-scheduler              reduceonplateau    \
    --lr-scheduler-patience     1                  \
    --lr-scheduler-decay        0.4                \
    --validation-every-n-epochs 0.5                \
    --warmup_updates            100                \
    --fp16                      true               
    # --num-epochs                3                  \

# Params in top block shouldn't make a difference
# Most uncertain about 
#   --lr-scheduler-patience
#   --

# stephenroller says:
# -cands batch -ecands inline


python -u parlai/scripts/train_model.py -candidates