import glove
from glove import Glove

import torchtext
from torchtext.data import Field
import spacy # for tokenizer

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

corpus = glove.Corpus()
corpus2 = glove.Corpus()
corpus.fit(src_sentences, window=10)
corpus2.fit(trg_sentences, window=10)

glove = Glove(no_components=512, learning_rate=0.05)

glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('src_glove.model')

glove.fit(corpus2.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus2.dictionary)
glove.save('trg_glove.model')