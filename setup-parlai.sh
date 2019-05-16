#!/bin/bash

# setup-parlai.sh

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