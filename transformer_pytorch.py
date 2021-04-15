'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
let's start transformer!
the code refered to
https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb

1. dataset from wmt 2014 English-German
2. tokenize them
3. make transformer model
4. train and evaluate model
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
imports
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, Sampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
# from torch.nn.parallel.distributed import DistributedDataParallel
# from parallel import DataParallelModel, DataParallelCriterion

import torchtext
from torchtext.data import Field, BucketIterator
import spacy # for tokenizer

from gensim.models import Word2Vec
from glove import Glove
import glove

import numpy as np
import sys
import os
import os.path
import random
import pickle
import gzip

import hyperparameters_pytorch as hparams
import customutils_pytorch as utils

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
preparing data and environment

# torchtext==0.6.0
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''sample for time saving'''
sample_valid_size = 300
sample_test_size = 50

model_name = 'transformer_en_de_gl_3'
model_filepath = f'{os.getcwd()}/{model_name}.pt'
vocab_filepath = f'{os.getcwd()}/.data/wmt14/vocab.bpe.32000'

saved_train_path = f'{os.getcwd()}/.data/wmt14/saved_train.pickle'
saved_valid_path = f'{os.getcwd()}/.data/wmt14/saved_valid.pickle'
saved_test_path = f'{os.getcwd()}/.data/wmt14/saved_test.pickle'

saved_train_pad_path = f'{os.getcwd()}/saved_train_pad.pickle'
saved_valid_pad_path = f'{os.getcwd()}/saved_valid_pad.pickle'
saved_test_pad_path = f'{os.getcwd()}/saved_test_pad.pickle'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

share_vocab = True
pretrained_embed = None

def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

'''load data'''

SRC = Field(tokenize = tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize = tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

st = utils.time.time()
if os.path.isfile(saved_train_path):
    train, valid, test = torchtext.datasets.WMT14.splits(exts=('.en', '.de'), fields=(SRC, TRG),
	                                                     train='dummy', validation='dummy', test='dummy')
    with gzip.open(saved_train_path, 'rb') as f:
        train.examples = pickle.load(f)
    with gzip.open(saved_valid_path, 'rb') as f:
        valid.examples = pickle.load(f)
    with gzip.open(saved_test_path, 'rb') as f:
        test.examples = pickle.load(f)

else:
    train, valid, test = torchtext.datasets.WMT14.splits(exts=('.en', '.de'),
                                                     fields=(SRC, TRG))
    with gzip.open(saved_train_path, 'wb') as f:
        pickle.dump(train.examples, f)
    with gzip.open(saved_valid_path, 'wb') as f:
        pickle.dump(valid.examples, f)
    with gzip.open(saved_test_path, 'wb') as f:
        pickle.dump(test.examples, f)

# reduce size of data to save time
# train.examples = random.sample(train.examples, sample_valid_size)
# valid.examples = random.sample(valid.examples, sample_valid_size)
# test.examples = random.sample(test.examples, sample_test_size)

et = utils.time.time()
m, s = utils.epoch_time(st, et)
print(f'data split completed | time : {m}m {s}s')
sys.stdout.flush()

st = utils.time.time()

if os.path.isfile(vocab_filepath):
    with open(vocab_filepath, 'r', encoding='utf-8') as f:
        vocablist = [i for i in f.read().split('\n')]
    print('src vocab_file loaded')
    sys.stdout.flush()
    SRC.build_vocab(train, valid, test, [vocablist])
else:
    SRC.build_vocab(train, valid, test)

et = utils.time.time()
m, s = utils.epoch_time(st, et)
print(f"SRC build success | time : {m}m {s}s")
sys.stdout.flush()

st = utils.time.time()

if os.path.isfile(vocab_filepath):
    with open(vocab_filepath, 'r', encoding='utf-8') as f:
        vocablist = [i for i in f.read().split('\n')]
    print('trg vocab_file loaded')
    sys.stdout.flush()
    TRG.build_vocab(train, valid, test, [vocablist])
else:
    TRG.build_vocab(train, valid, test)

et = utils.time.time()
m, s = utils.epoch_time(st, et)
print(f"TRG build success | time : {m}m {s}s")
sys.stdout.flush()

if share_vocab:
    print('Merging two vocabulary...')
    for w, _ in SRC.vocab.stoi.items():
        if w not in TRG.vocab.stoi:
            TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
    TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
    for w, i in TRG.vocab.stoi.items():
        TRG.vocab.itos[i] = w
    SRC.vocab.stoi = TRG.vocab.stoi
    SRC.vocab.itos = TRG.vocab.itos
    print('Get merged vocabulary size: ', len(TRG.vocab))
    sys.stdout.flush()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
BucketIterator

if your computer has just a few cores, maybe able to go with it.
since bucketiterator can not fix number of workers,
DataLoader is recommended
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# st = utils.time.time()
# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train, valid, test),
#     batch_size=hparams.batch_size,
#     device=device)
# et = utils.time.time()
# m, s = utils.epoch_time(st, et)
# print(f"bucketiterator splits complete | time : {m}m {s}s")
# sys.stdout.flush()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
since BucketIterator is too slow (low GPU, high CPU),
encouraged to use torch.utils.data.DataLoader
by https://github.com/pytorch/text/issues/664
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
train_ds = sorted([(len(each.src),
                    len(each.trg),
                    [SRC.vocab[i] for i in each.src],
                    [TRG.vocab[i] for i in each.trg])
                    for i, each in enumerate(train)], key=lambda data:(data[0], data[1]), reverse=True)
valid_ds = sorted([(len(each.src),
                    len(each.trg),
                    [SRC.vocab[i] for i in each.src],
                    [TRG.vocab[i] for i in each.trg])
                   for i, each in enumerate(valid)], key=lambda data:(data[0], data[1]), reverse=True)
test_ds = sorted([(len(each.src),
                   len(each.trg),
                   [SRC.vocab[i] for i in each.src],
                   [TRG.vocab[i] for i in each.trg])
                  for i, each in enumerate(test)], key=lambda data:(data[0], data[1]), reverse=True)

max_len_sentence = 0

def pad_data(data):
    '''Find max length of the mini-batch'''

    '''look data as column'''
    global max_len_sentence

    max_len_trg = max(list(zip(*data))[1])
    max_len_src = max(list(zip(*data))[0])
    src_ = list(zip(*data))[2]
    trg_ = list(zip(*data))[3]

    '''eos + pad'''
    padded_src = torch.stack([torch.cat((torch.tensor(txt).to('cpu'), torch.tensor([SRC.vocab.stoi[SRC.eos_token]]+([SRC.vocab.stoi[SRC.pad_token]] * (max_len_src - len(txt)))).long().to('cpu'))) for txt in src_])
    '''init token'''
    padded_src = torch.cat((torch.tensor([[SRC.vocab.stoi[SRC.init_token]]] * len(data)).to('cpu'), padded_src), dim=1)

    '''eos + pad'''
    padded_trg = torch.stack([torch.cat((torch.tensor(txt).to('cpu'), torch.tensor([TRG.vocab.stoi[TRG.eos_token]]+([TRG.vocab.stoi[TRG.pad_token]] * (max_len_trg - len(txt)))).long().to('cpu'))) for txt in trg_])
    '''init token'''
    padded_trg = torch.cat((torch.tensor([[TRG.vocab.stoi[TRG.init_token]]] * len(data)).to('cpu'), padded_trg), dim=1)
    max_len_sentence = max(max_len_sentence, len(padded_src[0]), len(padded_trg[0]))
    return [(s,t) for s,t in zip(padded_src, padded_trg)]
    # return padded_src, padded_trg

def chunker(data, batch_size):
    result = []
    for i in range(0, len(data), batch_size):
       result += [pad_data(data[i:i+batch_size]) for i in range(i, i+batch_size)] 
    return result

st = utils.time.time()

if os.path.isfile(saved_train_pad_path):
    with gzip.open(saved_train_pad_path, 'rb') as f:
        train_ds = pickle.load(f)
else:
    train_ds = pad_data(train_ds)
    with gzip.open(saved_train_pad_path, 'wb') as f:
        pickle.dump(train_ds, f)

if os.path.isfile(saved_valid_pad_path):
    with gzip.open(saved_valid_pad_path, 'rb') as f:
        valid_ds = pickle.load(f)
else:
    valid_ds = pad_data(valid_ds)
    with gzip.open(saved_valid_pad_path, 'wb') as f:
        pickle.dump(valid_ds, f)

if os.path.isfile(saved_test_pad_path):
    with gzip.open(saved_test_pad_path, 'rb') as f:
        test_ds = pickle.load(f)
else:
    test_ds = pad_data(test_ds)
    with gzip.open(saved_test_pad_path, 'wb') as f:
        pickle.dump(test_ds, f)

et = utils.time.time()

m, s = utils.epoch_time(st, et)
print(f"data is ready | time : {m}m {s}s")
sys.stdout.flush()

train_loader = DataLoader(train_ds, batch_size=hparams.batch_size, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=hparams.batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=hparams.batch_size, num_workers=4, pin_memory=True)

# example
# for i, batch in enumerate(dataloader):
#     src = batch[0]
#     trg = batch[1]
#     break

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
embedding
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class PretrainedEmbedding(pl.LightningModule):
    def __init__(self):
        global pretrained_embed

        if pretrained_embed is not None:
            self.src_embed_mtrx = pretrained_embed.src.embed_mtrx
            self.trg_embed_mtrx = pretrained_embed.trg.embed_mtrx
            return

        src_vocabsize = len(SRC.vocab.stoi)
        trg_vocabsize = len(TRG.vocab.stoi)

        self.src_embed_mtrx = torch.randn(src_vocabsize, hparams.d_model)
        self.trg_embed_mtrx = torch.randn(trg_vocabsize, hparams.d_model)

        '''
        for word2vec
        '''
        # src_word2vec = Word2Vec.load('src_embedd.model')
        # trg_word2vec = Word2Vec.load('trg_embedd.model')

        # for i in range(src_vocabsize):
        #     word = list(SRC.vocab.stoi.keys())[i]
        #     if word in src_word2vec.wv.index2word:
        #         src_embed_mtrx[SRC.vocab.stoi[word]] = torch.tensor(src_word2vec.wv[word].copy()).to(device)
        #
        # for i in range(trg_vocabsize):
        #     word = list(TRG.vocab.stoi.keys())[i]
        #     if word in trg_word2vec.wv.index2word:
        #         trg_embed_mtrx[TRG.vocab.stoi[word]] = torch.tensor(trg_word2vec.wv[word].copy()).to(device)

        '''
        for glove
        '''
        glove = Glove()
        src_glove = glove.load('src_glove.model')
        trg_glove = glove.load('trg_glove.model')

        for word in list(SRC.vocab.stoi.keys()):
            if word in src_glove.dictionary:
                self.src_embed_mtrx[SRC.vocab.stoi[word]] = torch.tensor(src_glove.word_vectors[src_glove.dictionary[word]].copy()).to(device)

        for word in list(TRG.vocab.stoi.keys()):
            if word in trg_glove.dictionary:
                self.trg_embed_mtrx[SRC.vocab.stoi[word]] = torch.tensor(trg_glove.word_vectors[trg_glove.dictionary[word]].copy()).to(device)

        pretrained_embed = self

        print("pretrained word embeddings loaded")
        sys.stdout.flush()


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

# sinusoid_encoding_table = torch.FloatTensor(get_sinusoid_encoding_table(max_len_sentence, hparams.d_model)).to(device)
class PositionalEncoding(pl.LightningModule):
    def __init__(self, t_seq, d_model):
        super().__init__()
        self.sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(t_seq, d_model))

    def forward(self, x):
        x_len = x.shape[1]
        with torch.no_grad():
            result = torch.add(x, self.sinusoid_table[:x_len, :])
        return result
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Self Attention
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class MultiHeadAttentionLayer(pl.LightningModule):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout_ratio):
        super().__init__()
        # since d_v * n_heads = d_model in the paper,
        assert d_model % n_heads == 0

        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_k * n_heads)
        self.w_k = nn.Linear(d_model, d_k * n_heads)
        self.w_v = nn.Linear(d_model, d_v * n_heads)

        self.w_o = nn.Linear(d_v * n_heads, d_model)

        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # make seperate heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).permute(0,2,1,3)

        similarity = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        # similarity: [batch_size, n_heads, query_len, key_len]

        if mask is not None:
            similarity = similarity.masked_fill(mask==0, -1e10)

        similarity_norm = torch.softmax(similarity, dim=-1)

        # dot product attention
        x = torch.matmul(self.dropout(similarity_norm), V)

        # x: [batch_size, n_heads, query_len, key_len]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x: [batch_size, query_len, n_heads, d_v]
        x = x.view(batch_size, -1, self.d_model)
        # x: [batch_size, query_len, d_model]
        x = self.w_o(x)
        # x: [batch_size, query_len, d_model]
        return x, similarity_norm

class PositionwiseFeedforwardLayer(pl.LightningModule):
    def __init__(self, d_model, d_ff, dropout_ratio):
        super().__init__()

        self.w_ff1 = nn.Linear(d_model, d_ff)
        self.w_ff2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.dropout(torch.relu(self.w_ff1(x)))
        # x: [batch_size, seq_len, d_ff]
        x = self.w_ff2(x)
        # x: [batch_size, seq_len, d_model]
        return x

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
TransformerEncoderLayer
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class TransformerEncoderLayer(pl.LightningModule):
    def __init__(self, d_k, d_v, d_model, n_heads, d_ff, dropout_ratio):
        super().__init__()

        self.self_attn_layer = MultiHeadAttentionLayer(d_k=d_k,
                                                       d_v=d_v,
                                                       d_model=d_model,
                                                       n_heads=n_heads,
                                                       dropout_ratio=dropout_ratio)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.positionwise_ff_layer = PositionwiseFeedforwardLayer(d_model=d_model,
                                                                  d_ff=d_ff,
                                                                  dropout_ratio=dropout_ratio)
        self.positionwise_ff_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src, src_mask):

        # src: [batch_size, src_len, d_model]
        # src_mask: [batch_size, src_len]

        attn_src, _ = self.self_attn_layer(src, src, src, src_mask)
        attn_add_norm_src = self.self_attn_layer_norm(src + self.dropout(attn_src))

        ff_src = self.positionwise_ff_layer(attn_add_norm_src)
        ff_add_norm_src = self.positionwise_ff_layer_norm(self.dropout(attn_add_norm_src) + ff_src)

        return ff_add_norm_src

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
TransformerEncoder
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class TransformerEncoder(pl.LightningModule):
    def __init__(self, input_dim, d_k, d_v, d_model, n_layers, n_heads, d_ff, dropout_ratio, device, max_length=100):
        super().__init__()

        self.pretained_embedding = PretrainedEmbedding()
        self.tok_embedding = nn.Embedding(input_dim, d_model).from_pretrained(self.pretained_embedding.src_embed_mtrx).requires_grad_(False)
        self.positional_encoding = PositionalEncoding(max_len_sentence, hparams.d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_k=d_k,
                                                             d_v=d_v,
                                                             d_model=d_model,
                                                             n_heads=n_heads,
                                                             d_ff=d_ff,
                                                             dropout_ratio=dropout_ratio) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([d_k]))

    def forward(self, src, src_mask):
        batch_size =src.shape[0]
        src_len = src.shape[1]

        '''to map position index information'''

        # pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout(self.tok_embedding(src))
        # pe = get_sinusoid_encoding_table(src_len, src.shape[2], self.device)
        # +positional encoding
        src = self.positional_encoding(src)
        # del pe
        # src: [batch_size, src_len, d_model]
        for layer in self.layers:
            src = layer(src, src_mask)

        return src

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
TransformerDecoderLayer
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class TransformerDecoderLayer(pl.LightningModule):
    def __init__(self, d_k, d_v, d_model, n_heads, d_ff, dropout_ratio):
        super().__init__()
        self.self_attn_layer = MultiHeadAttentionLayer(d_k=d_k,
                                                       d_v=d_v,
                                                       d_model=d_model,
                                                       n_heads=n_heads,
                                                       dropout_ratio=dropout_ratio)

        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.enc_dec_attn_layer = MultiHeadAttentionLayer(d_k=d_k,
                                                          d_v=d_v,
                                                          d_model=d_model,
                                                          n_heads=n_heads,
                                                          dropout_ratio=dropout_ratio,
                                                          device=device)

        self.enc_dec_attn_layer_norm = nn.LayerNorm(d_model)

        self.positionwise_ff_layer = PositionwiseFeedforwardLayer(d_model=d_model,
                                                                  d_ff=d_ff,
                                                                  dropout_ratio=dropout_ratio)
        self.positionwise_ff_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch_size, trg_len, d_model]
        # enc_src: [batch_size, src_len, d_model]
        # trg_mask: [batch_size, trg_len]
        # enc_mask: [batch_size, src_len]

        self_attn_trg, _ = self.self_attn_layer(trg, trg, trg, trg_mask)
        self_attn_add_norm_trg = self.self_attn_layer_norm(trg + self.dropout(self_attn_trg))
        enc_dec_attn_trg, attention = self.enc_dec_attn_layer(self_attn_add_norm_trg, enc_src, enc_src, src_mask)
        enc_dec_add_norm_trg = self.enc_dec_attn_layer_norm(self_attn_add_norm_trg + self.dropout(enc_dec_attn_trg))
        ff_trg = self.positionwise_ff_layer(enc_dec_add_norm_trg)
        ff_add_norm_trg = self.positionwise_ff_layer_norm(enc_dec_add_norm_trg + self.dropout(ff_trg))

        return ff_add_norm_trg, attention


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
TransformerDecoder
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class TransformerDecoder(pl.LightningModule):
    def __init__(self, output_dim, d_k, d_v, d_model, n_layers, n_heads, d_ff, dropout_ratio, max_length=100):
        super().__init__()
        self.pretrained_embedding = PretrainedEmbedding()
        self.tok_embedding = nn.Embedding(output_dim, d_model).from_pretrained(self.pretrained_embedding.trg_embed_mtrx).requires_grad_(False)
        self.positional_encoding = PositionalEncoding(max_len_sentence, hparams.d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_k=d_k,
                                                             d_v=d_v,
                                                             d_model=d_model,
                                                             n_heads=n_heads,
                                                             d_ff=d_ff,
                                                             dropout_ratio=dropout_ratio) for _ in range(n_layers)])
        self.affine = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([d_k]))

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos: [batch_size, trg_len]
        trg = self.dropout(self.tok_embedding(trg))
        # pe = get_sinusoid_encoding_table(trg_len, trg.shape[2], self.device)
        trg = self.positional_encoding(trg)

        # '''+positional encoding'''
        # with torch.no_grad():
        #     trg += sinusoid_encoding_table[:trg_len, :] 

        # del pe
        # trg: [batch_size, trg_len, d_model]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.affine(trg)
        # output: [batch_size, trg_len, output_len]

        return output, attention

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Transformer
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class Transformer(pl.LightningModule):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        src_mask = src_mask.type_as(src, device=self.device)
        # src_mask: [batch_size, 1, 1, src_len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg: [batch_size, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.type_as(trg, device=self.device)
        # trg_pad_mask = [batch_size, 1, 1, trg_len]
        trg_len = trg.shape[1]
        trg_attn_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()
        trg_attn_mask = trg_attn_mask.type_as(trg, device=self.device)
        # trg_attn_mask = [trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_attn_mask
        return trg_mask

    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask: [batch_size, 1, 1, src_len]
        # trg_mask: [batch_size, 1, trg_len, trg_len]

        enc_src = self.encoder(src, src_mask)
        # enc_src: [batch_size, src_len, d_model]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output: [batch_size, trg_len, output_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]
        return output, attention

    def configure_optimizers(self):
        warmup_steps = 4000
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=hparams.learning_rate)
        # loss = LabelSmoothingLoss(smoothing=hparams.label_smoothing, classes=len(TRG.vocab.stoi), ignore_index=TRG_PAD_IDX)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                lr_lambda=lambda steps: (hparams.d_model ** (-0.5)) * min(
                                                    (steps + 1) ** (-0.5), (steps + 1) * warmup_steps ** (-1.5)),
                                                last_epoch=-1,
                                                verbose=False)
        return [optimizer], [scheduler]

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
bleu score
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from torchtext.data.metrics import bleu_score

def show_bleu(data, SRC, TRG, model, device, logging = False, max_len=50):
    trgs = []
    pred_trgs = []
    index = 0

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, SRC, TRG, model, device, max_len, logging=False)

        # remove <eos>
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

        index+=1
        if (index + 1) % 100 == 0 and logging:
            print(f'[{index+1}/{len(data)}]')
            print(f'pred: {pred_trg}')
            print(f'answer: {trg}')
    bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    print(f'Total BLEU Score = {bleu*100:.2f}')
    sys.stdout.flush()

    # individual_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1,0,0,0])
    # individual_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0,1,0,0])
    # individual_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0,0,1,0])
    # individual_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0,0,0,1])
    #
    # cumulative_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1,0,0,0])
    # cumulative_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/2,1/2,0,0])
    # cumulative_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/3,1/3,1/3,0])
    # cumulative_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/4,1/4,1/4,1/4])
    #
    # print(f'Individual BLEU1 score = {individual_bleu1_score * 100:.2f}')
    # print(f'Individual BLEU2 score = {individual_bleu2_score * 100:.2f}')
    # print(f'Individual BLEU3 score = {individual_bleu3_score * 100:.2f}')
    # print(f'Individual BLEU4 score = {individual_bleu4_score * 100:.2f}')
    #
    # print(f'Cumulative BLEU1 score = {cumulative_bleu1_score * 100:.2f}')
    # print(f'Cumulative BLEU2 score = {cumulative_bleu2_score * 100:.2f}')
    # print(f'Cumulative BLEU3 score = {cumulative_bleu3_score * 100:.2f}')
    # print(f'Cumulative BLEU4 score = {cumulative_bleu4_score * 100:.2f}')

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Label Smoothing
refered to https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""
output (FloatTensor): batch_size x n_classes
target (LongTensor): batch_size
"""
class LabelSmoothingLoss(pl.LightningModule):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None, ignore_index=1):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
            true_len = len([i for i in target if i!=self.ignore_index])

        return torch.sum(torch.sum(-true_dist * pred, dim=self.dim)) / true_len

INPUT_DIM = len(SRC.vocab.stoi)
OUTPUT_DIM = len(TRG.vocab.stoi)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Preparing for Training
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# prepare model

if os.path.isfile(model_filepath):
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    print('model loaded from saved file')
    sys.stdout.flush()
else:
    model.apply(utils.initalize_weights)

print(f'The model has {utils.count_parameters(model):,} trainable parameters')
sys.stdout.flush()

'''set optimizer, scheduler and loss function'''
# optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=hparams.learning_rate)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
#                                         lr_lambda=lambda steps:(hparams.d_model**(-0.5))*min((steps+1)**(-0.5), (steps+1)*warmup_steps**(-1.5)),
#                                         last_epoch=-1,
#                                         verbose=False)

# loss_fn = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
# loss_fn = LabelSmoothingLoss(smoothing=hparams.label_smoothing, classes=len(TRG.vocab.stoi), ignore_index=TRG_PAD_IDX)

# parallel_model = DataParallelModel(model, device_ids=[0,1])
# parallel_model.to(device)
# parallel_loss = DataParallelCriterion(loss_fn, device_ids=[0,1])
# parallel_loss.to(device)

best_valid_loss = float('inf')
'''
since working enviornment takes too long to complete 1 epoch, make frequent log and save model
logged after (iter_part * batch_size) are completed
'''
def train_model(model, iterator, optimizer, loss_fn, epoch_num, iter_part=150):
    global best_valid_loss
    total_length = len(train.examples)
    total_parts = total_length // (hparams.batch_size * iter_part)

    if total_length % (hparams.batch_size * iter_part) != 0:
        total_parts+=1

    current_part = 1
    model.train()
    epoch_loss = 0

    part_start_time = utils.time.time()

    for i, batch in enumerate(iterator):
        src = batch[0].to(device, non_blocking=True)
        trg = batch[1].to(device, non_blocking=True)

        '''for bucketiterator users'''
        # src = batch.src
        # trg = batch.trg
        ''''''''''''''''''''''''''''''

        optimizer.zero_grad(set_to_none=True)

        '''exclude <eos> for decoder input'''
        # output, _ = model(src, trg[:, :-1])
        output, _ = model(src, trg[:, :-1])
        # output: [batch_size, trg_len-1, output_dim]

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)

        # output: [batch_size*trg_len-1, output_dim]

        trg = trg[:,1:].contiguous().view(-1)
        # trg: [batch_size*trg_len-1]

        # loss = loss_fn(output, trg)
        loss = loss_fn(output, trg)
        loss.backward()

        '''graident clipping'''
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        '''parameter update'''
        optimizer.step()
        scheduler.step()

        '''total loss in each epochs'''
        epoch_loss += float(loss.item())

        '''part log part'''
        if (i+1) % iter_part == 0:
            print(f'{total_length if (i+1)*hparams.batch_size > total_length else (i+1)*hparams.batch_size} / {total_length}, part {current_part} / {total_parts} complete...')
            part_end_time = utils.time.time()
            part_mins, part_secs = utils.epoch_time(part_start_time, part_end_time)
            print(f'Part Train Loss: {epoch_loss / (i+1):.3f} | Part Train PPL: {utils.math.exp(epoch_loss / (i+1)):.3f} | Time : {part_mins}m {part_secs}s')
            sys.stdout.flush()
            valid_loss = evaluate_model(model, valid_loader, loss_fn)
            print(f'Validation Loss: {valid_loss:.3f} | Validation PPL: {utils.math.exp(valid_loss):.3f}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'{model_name}.pt')
                print(f'model saved at {model_name}')
            # show_bleu(test, SRC, TRG, model, device) '''show_bleu is too slow'''
            part_end_time = utils.time.time()
            part_mins, part_secs = utils.epoch_time(part_start_time, part_end_time)
            print(f'{part_mins}m {part_secs}s')
            current_part += 1
            sys.stdout.flush()
            model.train()
            part_start_time = utils.time.time()
        del loss
    return epoch_loss / len(iterator

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
TrainModel for pytorch lightning
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class TrainModel(pl.LightningModule):
    def __init__(self):
        enc = TransformerEncoder(input_dim=INPUT_DIM,
                                 d_k=hparams.d_k,
                                 d_v=hparams.d_v,
                                 d_model=hparams.d_model,
                                 n_layers=hparams.n_decoder,
                                 n_heads=hparams.n_heads,
                                 d_ff=hparams.d_ff,
                                 dropout_ratio=hparams.dropout_ratio)

        dec = TransformerDecoder(output_dim=OUTPUT_DIM,
                                 d_k=hparams.d_k,
                                 d_v=hparams.d_v,
                                 d_model=hparams.d_model,
                                 n_layers=hparams.n_decoder,
                                 n_heads=hparams.n_heads,
                                 d_ff=hparams.d_ff,
                                 dropout_ratio=hparams.dropout_ratio)

        self.model = Transformer(encoder=enc,
                                 decoder=dec,
                                 src_pad_idx=SRC_PAD_IDX,
                                 trg_pad_idx=TRG_PAD_IDX)

        self.loss = LabelSmoothingLoss(smoothing=hparams.label_smoothing, classes=len(TRG.vocab.stoi),
                                       ignore_index=TRG_PAD_IDX)

    def forward(self, x, y):
        output, _ = self.model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output_dim)
        return output

    def training_step(self, x, y):
        output = self(x, y)
        y = y[:, 1:].contiguous().view(-1)
        loss = self.loss(output, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=hparams.learning_rate)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                lr_lambda=lambda steps:(hparams.d_model**(-0.5))*min((steps+1)**(-0.5), (steps+1)*warmup_steps**(-1.5)),
                                                last_epoch=-1,
                                                verbose=False)
        return optimizer, scheduler

    def translate_sentences(self, sentence, SRC, TRG, max_len=50, logging=False):

        self.model.eval()

        if isinstance(sentence, str):
            tokenizer = spacy.load('de_core_news_sm')
            tokens = [token.text.lower() for token in tokenizer(sentence)]
        else:
            tokens = [token.lower() for token in sentence]

        '''put <sos> in the first, <eos> in the end.'''
        tokens = [SRC.init_token] + tokens + [SRC.eos_token]
        '''convert to indexes'''
        src_indexes = [SRC.vocab.stoi[token] for token in tokens]

        if logging:
            print(f'src tokens : {tokens}')
            print(f'src indexes : {src_indexes}')

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

        src_pad_mask = model.make_src_mask(src_tensor)

        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_pad_mask)

        '''always start with first token'''
        trg_indexes = [TRG.vocab.stoi[TRG.init_token]]

        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_src_mask(trg_tensor)

            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_pad_mask)

            # output: [batch_size, trg_len, output_dim]
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

            if pred_token == TRG.vocab.stoi[TRG.eos_token]:
                break

        trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes]

        return trg_tokens[1:], attention

    def show_bleu_score(self, data, SRC, TRG, logging = False, max_len=50):
        trgs = []
        pred_trgs = []
        index = 0

        for datum in data:
            src = vars(datum)['src']
            trg = vars(datum)['trg']

            pred_trg, _ = self.translate_sentences(src, SRC, TRG, max_len, logging=False)

            # remove <eos>
            pred_trg = pred_trg[:-1]

            pred_trgs.append(pred_trg)
            trgs.append([trg])

            index += 1
            if (index + 1) % 100 == 0 and logging:
                print(f'[{index + 1}/{len(data)}]')
                print(f'pred: {pred_trg}')
                print(f'answer: {trg}')

        bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
        print(f'Total BLEU Score = {bleu * 100:.2f}')
        sys.stdout.flush()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Evaluation
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def evaluate_model(model, iterator, loss_fn):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device, non_blocking=True)
            trg = batch[1].to(device, non_blocking=True)

            output, _ = model(src, trg[:, :-1])

            output_dim = output.shape[-1]

            '''exclude <eos> for decoder input'''
            output, _ = model(src, trg[:, :-1])
            # output: [batch_size, trg_len-1, output_dim]

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            # output: [batch_size*trg_len-1, output_dim]

            trg = trg[:, 1:].contiguous().view(-1)
            # trg: [batch_size*trg_len-1]

            loss = loss_fn(output, trg)

            '''total loss in each epochs'''
            epoch_loss += float(loss.item())

    return epoch_loss / len(iterator)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Generation
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def translate_sentence(sentence, SRC, TRG, model, device, max_len=50, logging=False):
    model.eval()

    if isinstance(sentence, str):
        tokenizer = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    '''put <sos> in the first, <eos> in the end.'''
    tokens = [SRC.init_token] + tokens + [SRC.eos_token]
    '''convert to indexes'''
    src_indexes = [SRC.vocab.stoi[token] for token in tokens]

    if logging:
        print(f'src tokens : {tokens}')
        print(f'src indexes : {src_indexes}')

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_pad_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_pad_mask)

    '''always start with first token'''
    trg_indexes = [TRG.vocab.stoi[TRG.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_src_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_pad_mask)

        # output: [batch_size, trg_len, output_dim]
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)

        if pred_token == TRG.vocab.stoi[TRG.eos_token]:
            break

    trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
custom callback
from https://github.com/PyTorchLightning/pytorch-lightning/issues/2534#issuecomment-674582085
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        global best_valid_loss
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step + 1
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            if trainer.callback_metrics['loss'] < best_valid_loss:
                ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
                trainer.save_checkpoint(ckpt_path)
                best_valid_loss = trainer.callback_metrics['loss']
            trainer.run_evaluation()
            trainer.model.show_bleu_score(test, SRC, TRG)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Training for pytorch lightning
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(gpus=[0,1],
                     max_steps=100000,
                     limit_train_batches=512,
                     callbacks=[lr_monitor, CheckpointEveryNSteps(save_step_frequency=1)],
                     deterministic=True)
trainer.fit(model, train_loader, valid_loader)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Training for pytorch
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
for epoch in range(hparams.n_epochs):
    start_time = utils.time.time()
    train_loss = train_model(model, train_loader, optimizer, parallel_loss, epoch)
    valid_loss = evaluate_model(model, valid_loader, parallel_loss)
    end_time = utils.time.time()
    epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '{model_name}.pt')
        print('model saved')
    print('---------------------------------------------------------')
    print(f'Epoch: {epoch+1:03} Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss: {train_loss:.3f} Train PPL: {utils.math.exp(train_loss):.3f}')
    print(f'Validation Loss: {valid_loss:.3f} Validation PPL: {utils.math.exp(valid_loss):.3f}')
    show_bleu(test, SRC, TRG, model, device)
    print('---------------------------------------------------------')
    print('\n')
    sys.stdout.flush()
'''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Generation Test
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
example_idx=10
src = vars(test.examples[example_idx])['src']
trg = vars(test.examples[example_idx])['trg']
print('generation:')
print(f'src : {src}')
print(f'trg : {trg}')
translation, attention = translate_sentence(sentence=src,
                                            SRC=SRC,
                                            TRG=TRG,
                                            model=model,
                                            device=device)
print('result :', ' '.join(translation))



show_bleu(test, SRC, TRG, model, device)
'''