'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
let's start transformer!
1. dataset from wmt 2014 English-German or newstest2013 for dev
2. tokenize them
3. make transformer model
4. train and evaluate model
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
imports
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
import torchtext
import torch.nn as nn
import numpy as np

import pretrain
import hyperparameters_pytorch as hparams

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
preparing data
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# load data
SRC = torchtext.legacy.data.Field(lower=True,
                                  include_lengths=True,
                                  batch_first=True)

TRG = torchtext.legacy.data.Field(lower=True,
                                  include_lengths=True,
                                  batch_first=True)

train, valid, test = torchtext.legacy.datasets.WMT14.splits(exts=('.en', '.de'),
                                                            fields=(SRC, TRG),
                                                            root='.data',
                                                            train='train.tok.clean.bpe.32000',
                                                            validation='newstest2013.tok.bpe.32000',
                                                            test='newstest2014.tok.bpe.32000')
# create vocabulary
SRC.build_vocab(train)
TRG.build_vocab(train)

# save vocab
def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()

def load_vocab(path):
    import pickle
    return pickle.load(path)

save_vocab(SRC.vocab, 'src.pkl')
save_vocab(TRG.vocab, 'trg.pkl')

test_save = load_vocab('src.pkl')
test_save = load_vocab('trg.pkl')

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
embedding
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
src_model, trg_model = pretrain.Custom_WordEmbedding(train)

e_embed = torch.randn(len(SRC.vocab.stoi), hparams.d_model, requires_grad=True)
d_embed = torch.randn(len(TRG.vocab.stoi), hparams.d_model, requires_grad=True)

SRC.vocab.set_vectors(stoi=SRC.vocab.stoi, vectors=e_embed, dim=hparams.d_model)
TRG.vocab.set_vectors(stoi=TRG.vocab.stoi, vectors=d_embed, dim=hparams.d_model)

encoder_embed = nn.Embedding.from_pretrained(e_embed)
decoder_embed = nn.Embedding.from_pretrained(d_embed)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
positional encoding
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def get_sinusoid_encoding_table(t_seq, d_model):
    def cal_angle(pos, i_model):
        return pos / np.power(10000, 2 * (i_model // 2) / d_model)

    def get_position_vec(pos):
        return [cal_angle(pos, i_model) for i_model in range(d_model)]

    sinusoid_table = np.array([get_position_vec(pos) for pos in range(t_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Self Attention
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = 1 / (d_k ** 0.5)