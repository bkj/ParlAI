
import json
import torch
import numpy as np
import pandas as pd
from itertools import chain

# --
# IO

ctx_emb = torch.load('_all_ctx_emb.83f69181-220d-404d-a9a8-223f2c466e02')
ctx_txt = torch.load('_all_ctx_txt.83f69181-220d-404d-a9a8-223f2c466e02')

cnd_emb = torch.load('_all_cnd_emb.83f69181-220d-404d-a9a8-223f2c466e02')
cnd_txt = torch.load('_all_cnd_txt.83f69181-220d-404d-a9a8-223f2c466e02')

# --
# Format

ctx_emb = np.vstack(ctx_emb).astype(np.float32)
cnd_emb = np.vstack(cnd_emb).astype(np.float32)
cnd_txt = np.vstack(cnd_txt)

flat_cnd_emb = cnd_emb.transpose((0, 2, 1)).reshape(7801 * 20, 768)
flat_cnd_txt = cnd_txt.reshape(7801 * 20)

_, uidx = np.unique(flat_cnd_txt, return_index=True)
uflat_cnd_emb, uflat_cnd_txt = flat_cnd_emb[uidx], flat_cnd_txt[uidx]

gold = cnd_txt[:,-1]

oh_gold = gold.reshape(-1, 1) == uflat_cnd_txt.reshape(1, -1)
assert (oh_gold.sum(axis=-1) == 1).all()
idx_gold = np.where(oh_gold == 1)[-1]

# --

sim = ctx_emb @ uflat_cnd_emb.T

pred = sim.argmax(axis=-1)

(idx_gold == pred).mean()
# 0.097 22687 distractors

# How many are from the right conversation? 

def print_random(i=None):
    if i is None:
        i = np.random.choice(ctx_emb.shape[0])
    
    print('-' * 100)
    print(ctx_txt[i])
    print('-' * 100)
    topk = np.argsort(-sim[i])[:10]
    for cnd in uflat_cnd_txt[topk]:
        print('+' if cnd == gold[i] else ' ', cnd)
    print('-' * 100)
    print('GOLD:', gold[i])
    
    return i

i = print_random()
