#!/bin/bash

# run.sh

source activate pytorch_p36
cd ~/software/parlai

mkdir -p experiments

OUTPATH="./experiments/v0"
rm -rf $OUTPATH && mkdir -p $OUTPATH
python -u parlai/scripts/train_model.py \
    --task              convai2                       \
    --model             bert_ranker/bi_encoder_ranker \
    --dict-file         $OUTPATH/dictionary           \
    --model-file        $OUTPATH/my_model             \
    --shuffle           true                          \
    --log_every_n_secs  10                            \
    --eval-batchsize    256                           \
    --data-parallel     true                          \
    --type-optimization         all_encoder_layers \
    --batchsize                 256                \
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

cat my_model.trainstats | jq -rc '.valid_reports | .[]'

# {"exs":7801,"accuracy":0.7047,"f1":0.7366,"hits@1":0.705,"hits@5":0.945,"hits@10":0.987,"hits@100":1,"bleu":0.7044,"lr":5e-05,"num_updates":257,"examples":7801,"loss":32.28,"mean_loss":0.004138,"mean_rank":1.873,"mrr":0.8063,"train_time":537.2963297367096}
# {"exs":7801,"accuracy":0.749,"f1":0.7767,"hits@1":0.749,"hits@5":0.956,"hits@10":0.989,"hits@100":1,"bleu":0.7491,"lr":5e-05,"num_updates":514,"examples":7801,"loss":27.85,"mean_loss":0.00357,"mean_rank":1.704,"mrr":0.8385,"train_time":1025.2623586654663}
# {"exs":7801,"accuracy":0.7608,"f1":0.7861,"hits@1":0.761,"hits@5":0.96,"hits@10":0.991,"hits@100":1,"bleu":0.7608,"lr":5e-05,"num_updates":771,"examples":7801,"loss":26.52,"mean_loss":0.003399,"mean_rank":1.645,"mrr":0.8472,"train_time":1529.0462341308594}
# {"exs":7801,"accuracy":0.7734,"f1":0.7978,"hits@1":0.773,"hits@5":0.965,"hits@10":0.991,"hits@100":1,"bleu":0.7732,"lr":5e-05,"num_updates":1028,"examples":7801,"loss":25.73,"mean_loss":0.003298,"mean_rank":1.608,"mrr":0.8552,"train_time":2034.2776310443878}
# {"exs":7801,"accuracy":0.7762,"f1":0.8005,"hits@1":0.776,"hits@5":0.965,"hits@10":0.992,"hits@100":1,"bleu":0.7763,"lr":5e-05,"num_updates":1285,"examples":7801,"loss":25.3,"mean_loss":0.003243,"mean_rank":1.595,"mrr":0.8579,"train_time":2539.547839164734}
# {"exs":7801,"accuracy":0.7737,"f1":0.7985,"hits@1":0.774,"hits@5":0.965,"hits@10":0.991,"hits@100":1,"bleu":0.7738,"lr":5e-05,"num_updates":1542,"examples":7801,"loss":26.08,"mean_loss":0.003344,"mean_rank":1.607,"mrr":0.8561,"train_time":3045.046471118927}
# {"exs":7801,"accuracy":0.7717,"f1":0.7962,"hits@1":0.772,"hits@5":0.964,"hits@10":0.992,"hits@100":1,"bleu":0.7717,"lr":5e-05,"num_updates":1799,"examples":7801,"loss":26.97,"mean_loss":0.003457,"mean_rank":1.615,"mrr":0.8543,"train_time":3529.8368356227875}
# {"exs":7801,"accuracy":0.785,"f1":0.8082,"hits@1":0.785,"hits@5":0.97,"hits@10":0.993,"hits@100":1,"bleu":0.785,"lr":2e-05,"num_updates":2056,"examples":7801,"loss":25.55,"mean_loss":0.003275,"mean_rank":1.541,"mrr":0.8649,"train_time":4015.5114421844482}
# {"exs":7801,"accuracy":0.7855,"f1":0.8088,"hits@1":0.786,"hits@5":0.969,"hits@10":0.992,"hits@100":1,"bleu":0.7855,"lr":2e-05,"num_updates":2313,"examples":7801,"loss":26.32,"mean_loss":0.003374,"mean_rank":1.556,"mrr":0.8645,"train_time":4519.821675300598}
# {"exs":7801,"accuracy":0.7882,"f1":0.8109,"hits@1":0.788,"hits@5":0.971,"hits@10":0.993,"hits@100":1,"bleu":0.7882,"lr":8e-06,"num_updates":2570,"examples":7801,"loss":26.17,"mean_loss":0.003354,"mean_rank":1.547,"mrr":0.8659,"train_time":5025.581673622131}
# {"exs":7801,"accuracy":0.7903,"f1":0.8127,"hits@1":0.79,"hits@5":0.971,"hits@10":0.993,"hits@100":1,"bleu":0.7903,"lr":8e-06,"num_updates":2827,"examples":7801,"loss":26.37,"mean_loss":0.00338,"mean_rank":1.54,"mrr":0.8678,"train_time":5531.12650847435}





# `helpers.sh` says `--num-epochs 3`, but lets just let run
# Paper says batchsize of 512 is best, but I'm getting OOM errors
# 256 and 128 are worse (83.4 to 83.0 or 82.3), but not substantially

# Params in top block shouldn't make a difference
# Most uncertain about 
#   --lr-scheduler-patience -- maybe should be 0

# stephenroller says:
# -cands batch -ecands inline
# but these are set by default


python -u parlai/scripts/train_model.py -candidates

# !! Need more logging during evaluation