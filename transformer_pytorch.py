'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
let's start transformer!
1. dataset from wmt 2014 English-German or newstest2013 for dev
2. tokenize them
3. make transformer model
4. train and evaluate model
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from torchtext import data
from torchtext import datasets

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
comments here
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SRC = data.Field(lower=True,
                 include_lengths=True,
                 batch_first=True)

TRG = data.Field(sequential=False)

train, valid, test = datasets.WMT14.splits(exts=('.en','.de'),
                                           fields=(SRC, TRG),
                                           root='.data',
                                           train= 'train.tok.clean.bpe.32000',
                                           validation= 'newstest2013.tok.bpe.32000',
                                           test= 'newstest2014.tok.bpe.32000')