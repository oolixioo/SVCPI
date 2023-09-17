import numpy as np
from collections import defaultdict
import pickle, torch

def dict_func():
    return len(WORD_DICT)

WORD_DICT = defaultdict(dict_func)

def split_prot_seq(seq, n=3, cover=1):
    words = []
    for c in range(cover):
        l = [WORD_DICT['^'*n]] + [WORD_DICT[seq[i+c:i+c+n]] for i in range(len(seq)-n-c+1)] + [WORD_DICT['$'*n]]
        words.extend(l)

    return torch.IntTensor(words)

def dump_word_dict(fn='data/protein_word_dict.pkl'):
    pickle.dump(WORD_DICT, open(fn,'wb'))
    print('Dump protein WORD_DICT finished. Nwords = %s' %len(WORD_DICT.keys()))

def load_word_dict(fn='data/protein_word_dict.pkl'):
    global WORD_DICT
    try:
        WORD_DICT = pickle.load(open(fn, 'rb'))
        print('Load protein WORD_DICT finished. Nwords = %s' %len(WORD_DICT.keys()))
    except Exception as e:
        print('Load protein WORD_DICT failed: %s' %e)