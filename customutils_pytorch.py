import torch.nn as nn
import math
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initalize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1 and model.__class__.__name__ != 'Embedding':
        nn.init.xavier_uniform_(model.weight.data)

def epoch_time(start_time, end_time):
    mean_time = end_time - start_time
    mins = int(mean_time/60)
    secs = int(mean_time - (mins * 60))
    return mins, secs

def save_vocab(vocab, path):
    with open(path, 'w+', encoding='utf-8') as f:
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')

def read_vocab(path):
    vocab = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab

def save_pickle(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic

