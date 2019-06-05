#!/bin/bash

# setup-parlai.sh

# pip install --upgrade awscli
# aws ec2 run-instances \
#     --image-id ami-027a93c0b4e47aab8 \
#     --count 1 \
#     --instance-type p3.2xlarge \
#     --key-name ~/.ssh/rzj.pem \
#     --security-group-ids sg-685ddc15


source activate pytorch_p36
pip install --upgrade pip

mkdir software

# Install rsub
sudo wget -O /usr/local/bin/rsub https://raw.github.com/aurora/rmate/master/rmate
sudo chmod +x /usr/local/bin/rsub

# Install APEX
cd ~/software
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" .

# Install ParlAI
cd ~/software
git clone https://github.com/facebookresearch/ParlAI
mv ParlAI parlai
cd parlai
python setup.py install

# Install huggingface BERT
pip install pytorch-pretrained-bert==0.6.2