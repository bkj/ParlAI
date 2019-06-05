#!/bin/bash

# run.sh

# --
# Setup

source activate pytorch_p36
cd ~/software/parlai

pip uninstall parlai -y
pip install -e .

# --
# Run

mkdir -p experiments
OUTPATH="./experiments/nolinear-nopatience"
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
    --lr-scheduler-patience     0                  \
    --lr-scheduler-decay        0.4                \
    --validation-every-n-epochs 0.5                \
    --warmup_updates            100                \
    --fp16                      true

cat my_model.trainstats | jq -rc '.valid_reports | .[]'

# w/ linear layer
# 
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

# w/o linear layer
cat experiments/v1-nolinear/my_model.trainstats | jq -rc '.valid_reports | .[]'
# {"exs":7801,"accuracy":0.7557,"f1":0.7821,"hits@1":0.756,"hits@5":0.957,"hits@10":0.99,"hits@100":1,"bleu":0.7557,"lr":5e-05,"num_updates":257,"examples":7801,"loss":27.65,"mean_loss":0.003545,"mean_rank":1.691,"mrr":0.8423,"train_time":535.7856335639954}
# {"exs":7801,"accuracy":0.7891,"f1":0.8117,"hits@1":0.789,"hits@5":0.969,"hits@10":0.992,"hits@100":1,"bleu":0.7891,"lr":5e-05,"num_updates":514,"examples":7801,"loss":23.14,"mean_loss":0.002967,"mean_rank":1.553,"mrr":0.8662,"train_time":1014.2737891674042}
# {"exs":7801,"accuracy":0.809,"f1":0.8298,"hits@1":0.809,"hits@5":0.967,"hits@10":0.993,"hits@100":1,"bleu":0.8089,"lr":5e-05,"num_updates":771,"examples":7801,"loss":22.39,"mean_loss":0.002871,"mean_rank":1.518,"mrr":0.878,"train_time":1509.8880500793457}
# {"exs":7801,"accuracy":0.8118,"f1":0.8321,"hits@1":0.812,"hits@5":0.97,"hits@10":0.993,"hits@100":1,"bleu":0.8118,"lr":5e-05,"num_updates":1028,"examples":7801,"loss":21.24,"mean_loss":0.002723,"mean_rank":1.508,"mrr":0.88,"train_time":2006.254047870636}
# {"exs":7801,"accuracy":0.811,"f1":0.8312,"hits@1":0.811,"hits@5":0.972,"hits@10":0.993,"hits@100":1,"bleu":0.8109,"lr":5e-05,"num_updates":1285,"examples":7801,"loss":21.92,"mean_loss":0.00281,"mean_rank":1.493,"mrr":0.8803,"train_time":2502.2443747520447}
# {"exs":7801,"accuracy":0.8093,"f1":0.8306,"hits@1":0.809,"hits@5":0.973,"hits@10":0.992,"hits@100":1,"bleu":0.8091,"lr":5e-05,"num_updates":1542,"examples":7801,"loss":21.98,"mean_loss":0.002818,"mean_rank":1.494,"mrr":0.8798,"train_time":2978.481077194214}
# {"exs":7801,"accuracy":0.8166,"f1":0.8364,"hits@1":0.817,"hits@5":0.976,"hits@10":0.993,"hits@100":1,"bleu":0.8165,"lr":2e-05,"num_updates":1799,"examples":7801,"loss":21.35,"mean_loss":0.002737,"mean_rank":1.469,"mrr":0.8852,"train_time":3456.119448900223}
# {"exs":7801,"accuracy":0.8164,"f1":0.836,"hits@1":0.816,"hits@5":0.975,"hits@10":0.993,"hits@100":1,"bleu":0.8163,"lr":2e-05,"num_updates":2056,"examples":7801,"loss":21.8,"mean_loss":0.002795,"mean_rank":1.468,"mrr":0.8851,"train_time":3952.0169672966003}
# {"exs":7801,"accuracy":0.8205,"f1":0.8399,"hits@1":0.821,"hits@5":0.975,"hits@10":0.994,"hits@100":1,"bleu":0.8204,"lr":8e-06,"num_updates":2313,"examples":7801,"loss":22,"mean_loss":0.00282,"mean_rank":1.463,"mrr":0.8873,"train_time":4428.604891061783}
# {"exs":7801,"accuracy":0.8213,"f1":0.8411,"hits@1":0.821,"hits@5":0.975,"hits@10":0.994,"hits@100":1,"bleu":0.8212,"lr":8e-06,"num_updates":2570,"examples":7801,"loss":22.32,"mean_loss":0.002862,"mean_rank":1.455,"mrr":0.8882,"train_time":4925.51379942894}

# w/o linear layer + lr-scheduler-patience=0
cat experiments/v2-nolinear/my_model.trainstats | jq -rc '.valid_reports | .[]'
# {"exs":7801,"accuracy":0.7611,"f1":0.7859,"hits@1":0.761,"hits@5":0.958,"hits@10":0.989,"hits@100":1,"bleu":0.7611,"lr":5e-05,"num_updates":257,"examples":7801,"loss":26.3,"mean_loss":0.003371,"mean_rank":1.676,"mrr":0.8456,"train_time":538.8439116477966}
# {"exs":7801,"accuracy":0.795,"f1":0.8175,"hits@1":0.795,"hits@5":0.971,"hits@10":0.993,"hits@100":1,"bleu":0.795,"lr":5e-05,"num_updates":514,"examples":7801,"loss":22.53,"mean_loss":0.002888,"mean_rank":1.535,"mrr":0.8704,"train_time":1026.8353519439697}
# {"exs":7801,"accuracy":0.7993,"f1":0.8212,"hits@1":0.799,"hits@5":0.968,"hits@10":0.992,"hits@100":1,"bleu":0.7991,"lr":5e-05,"num_updates":771,"examples":7801,"loss":21.83,"mean_loss":0.002798,"mean_rank":1.526,"mrr":0.8727,"train_time":1531.4706733226776}
# {"exs":7801,"accuracy":0.814,"f1":0.8339,"hits@1":0.814,"hits@5":0.971,"hits@10":0.993,"hits@100":1,"bleu":0.8138,"lr":5e-05,"num_updates":1028,"examples":7801,"loss":21.14,"mean_loss":0.00271,"mean_rank":1.493,"mrr":0.8816,"train_time":2035.4154632091522}
# {"exs":7801,"accuracy":0.8163,"f1":0.8365,"hits@1":0.816,"hits@5":0.972,"hits@10":0.994,"hits@100":1,"bleu":0.8161,"lr":5e-05,"num_updates":1285,"examples":7801,"loss":21.03,"mean_loss":0.002695,"mean_rank":1.472,"mrr":0.884,"train_time":2538.34698843956}
# {"exs":7801,"accuracy":0.8125,"f1":0.8331,"hits@1":0.812,"hits@5":0.972,"hits@10":0.993,"hits@100":1,"bleu":0.8123,"lr":5e-05,"num_updates":1542,"examples":7801,"loss":22.4,"mean_loss":0.002871,"mean_rank":1.489,"mrr":0.8809,"train_time":3041.653540611267}
# {"exs":7801,"accuracy":0.8223,"f1":0.842,"hits@1":0.822,"hits@5":0.975,"hits@10":0.993,"hits@100":1,"bleu":0.8223,"lr":2e-05,"num_updates":1799,"examples":7801,"loss":21.53,"mean_loss":0.00276,"mean_rank":1.462,"mrr":0.8874,"train_time":3524.147078037262}
# {"exs":7801,"accuracy":0.8252,"f1":0.8447,"hits@1":0.825,"hits@5":0.977,"hits@10":0.995,"hits@100":1,"bleu":0.825,"lr":8e-06,"num_updates":2056,"examples":7801,"loss":20.9,"mean_loss":0.002679,"mean_rank":1.434,"mrr":0.8903,"train_time":4024.2351970672607}
# {"exs":7801,"accuracy":0.8255,"f1":0.8449,"hits@1":0.826,"hits@5":0.976,"hits@10":0.995,"hits@100":1,"bleu":0.8254,"lr":8e-06,"num_updates":2313,"examples":7801,"loss":21.33,"mean_loss":0.002735,"mean_rank":1.435,"mrr":0.8902,"train_time":4528.047632694244}
# {"exs":7801,"accuracy":0.828,"f1":0.847,"hits@1":0.828,"hits@5":0.977,"hits@10":0.994,"hits@100":1,"bleu":0.8278,"lr":3.2e-06,"num_updates":2570,"examples":7801,"loss":21.3,"mean_loss":0.00273,"mean_rank":1.43,"mrr":0.8919,"train_time":5033.247322559357}
# {"exs":7801,"accuracy":0.8289,"f1":0.8479,"hits@1":0.829,"hits@5":0.977,"hits@10":0.995,"hits@100":1,"bleu":0.8287,"lr":1.28e-06,"num_updates":2827,"examples":7801,"loss":21.37,"mean_loss":0.00274,"mean_rank":1.429,"mrr":0.8923,"train_time":5536.921379566193}

# --------------------------------------------------------------------------------
# # Notes
#     `helpers.sh` says `--num-epochs 3`, but lets just let run
#     Paper says batchsize of 512 is best, but I'm getting OOM errors
#     256 and 128 are worse (83.4 to 83.0 or 82.3), but not substantially

#     Params in top block shouldn't make a difference
#     Most uncertain about 
#       --lr-scheduler-patience -- maybe should be 0

#     stephenroller says:
#     -cands batch -ecands inline
#     but these are set by default
