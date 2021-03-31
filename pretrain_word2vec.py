'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
imports
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
from gensim.models import Word2Vec

import torchtext
from torchtext.data import Field

import spacy # for tokenizer

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
preparing data and environment

# torchtext==0.6.0
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

# load data
SRC = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

train, valid, test = torchtext.datasets.WMT14.splits(exts=('.en', '.de'),
                                                     fields=(SRC, TRG))

length = len(train.examples)
src_sentences = []
trg_sentences = []
for i in range(length):
    src_sentences.append(vars(train.examples[i])['src'])
    trg_sentences.append(vars(train.examples[i])['trg'])

src_model = Word2Vec(src_sentences, size = 512, window=4, min_count=1)
trg_model = Word2Vec(trg_sentences, size = 512, window=4, min_count=1)

# src_model.train(src_sentences, total_examples=length, epochs=50)
# trg_model.train(trg_sentences, total_examples=length, epochs=50)

src_model.save('src_embedd.model')
trg_model.save('trg_embedd.model')