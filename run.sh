#!/bin/bash

# run.sh

source activate pytorch_p36

mkdir -p experiments/bi_encoder
OUTPATH="./experiments/bi_encoder/v0"

# --
# Train

rm -rf $OUTPATH && mkdir -p $OUTPATH
python -u parlai/scripts/train_model.py               \
    --task              convai2                       \
    --model             bert_ranker/bi_encoder_ranker \
    --dict-file         $OUTPATH/dictionary           \
    --model-file        $OUTPATH/model                \
    --shuffle           true                          \
    --log_every_n_secs  10                            \
    --eval-batchsize    256                           \
    --data-parallel     true                          \
    --type-optimization         all_encoder_layers    \
    --batchsize                 256                   \
    --learningrate              5e-5                  \
    --history-size              20                    \
    --label-truncate            72                    \
    --text-truncate             360                   \
    --lr-scheduler              reduceonplateau       \
    --lr-scheduler-patience     0                     \
    --lr-scheduler-decay        0.4                   \
    --validation-every-n-epochs 0.5                   \
    --warmup_updates            100                   \
    --fp16                      true

cat $OUTPATH/model.trainstats | jq -rc '.valid_reports | .[]'

# --
# Eval

mkdir -p $OUTPATH/vecs
python -u parlai/scripts/eval_model.py          \
    --task        convai2                       \
    --model       bert_ranker/bi_encoder_ranker \
    --model-file  $OUTPATH/model                \
    --numthreads  1                             \
    --batchsize   32