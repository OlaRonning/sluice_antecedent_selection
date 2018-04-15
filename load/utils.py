from load.readers.ant_reader import ant_reader
from load.readers.conll_reader import reader as conll_reader
from load.readers.opensub_reader import opensub_reader
from load.core import Dataset
from load import split
import paths as ps
from load import task

import json
import numpy as np
import itertools
from collections import defaultdict

def chain(*args):
    if len(args) == 1:
        l = itertools.chain.from_iterable(*args)
    else:
        l = itertools.chain(*args)
    return list(l)


class dispatcher:
    def __init__(self,tr_data,tr_labels,max_length,e2i,test=False,val_data=None,val_labels=None):
        self.max_len = max_length
        self.auxiliaries = defaultdict(task.Task)
        if test:
            self.main_task = task.TestTask(tr_data,tr_labels,max_length,e2i)
        else:
            if val_data is None:
                self.main_task = task.TrTask(tr_data,tr_labels,max_length,e2i)
            else:
                self.main_task = task.TrValMainTask(tr_data,tr_labels,val_data,val_labels,max_length,e2i)

    def add_auxiliary(self,name,tr_data,tr_labels,max_length,lbl2i,w2i,val_data=None,val_labels=None):
        if val_data is None:
            t = task.TrTask(tr_data,tr_labels,max_length,w2i)
        else:
            t = task.TrValTask(name,tr_data,tr_labels,val_data,val_labels,max_length,lbl2i,w2i)
        self.auxiliaries[name] = t

    @property
    def aux_keys(self):
        """aux_keys"""
        return self.auxiliaries.keys()

    def __getitem__(self,key):
        return self.auxiliaries[key]


def opensub(test=True,embedded=True,prefix=True,max_len=700):
    if embedded:
        dp = ps.OPENSUB_EMBEDDED_TEST
    else:
        dp = ps.OPENSUB_ROOT_TEST
    reader = opensub_reader(data_path=dp,prefix=prefix,max_len=max_len)
    data = reader.sentences
    labels = reader.labels
    vocab = list(np.load(ps.VOCAB))
    word2int = reader.vocab2int(vocab,len(vocab) - 1)
    dispatch = dispatcher(data,labels,max_len,word2int,test)

    return dispatch


def ant_seq(test=False,before=None,after=None,prefix=None,max_len=None):
    if test:
        dp = ps.ESC_TEST
        reader = ant_reader(data_path=dp)
        val_data = None
        val_labels = None
    else:
        dp = ps.ESC_TRAIN
        reader = ant_reader(data_path=dp)
        val_reader = ant_reader(data_path=ps.ESC_VAL)
        val_data = val_reader.truncated_sentences(before,after,prefix)
        val_labels = val_reader.token_lvl_labels
    data = reader.truncated_sentences(before,after,prefix)
    labels = reader.token_lvl_labels
    vocab = list(np.load(ps.VOCAB))
    word2int = reader.vocab2int(vocab,len(vocab) - 1)
    if max_len is not None:
        ml = max_len
    else:
        ml = reader.max_length
    dispatch = dispatcher(data, labels, ml, word2int, test, val_data, val_labels)
    return dispatch


def conll(dispatcher,dataset='chunk'):
    def vocab2int(vocab,other):
        """ Writes the index of word for the used vocab
            The vocab is determined by the embeddings
            Note: Build for use with pretrained embeddings vectors
        """
        v2i_dict = defaultdict(None)
        for idx,word in enumerate(vocab):
            v2i_dict[word] = idx
        def word2int(word):
            try:
                idx = v2i_dict[word]
                return idx
            except KeyError:
                return other
        return word2int

    if dataset == 'chunk':
        dp_train = ps.CHUNK_TRAIN
        dp_dev = ps.CHUNK_DEV
        dp_test = ps.CHUNK_TEST
    elif dataset ==  'com':
        dp_train = ps.COM_TRAIN
        dp_dev = ps.COM_DEV
        dp_test = ps.COM_TEST
    elif dataset == 'clause':
        dp_train = ps.CLAUSE_ALL
        dp_dev = None
        dp_test = None
    elif dataset == 'ner':
        dp_train = ps.NER_TRAIN
        dp_dev = ps.NER_DEV
        dp_test = ps.NER_TEST
    elif dataset == 'ccg':
        dp_train = ps.CCG_TRAIN
        dp_dev = ps.CCG_DEV
        dp_test = ps.CCG_TEST
    else:
        dataset = 'pos'
        dp_train = ps.POS_TRAIN
        dp_dev = None
        dp_test = ps.POS_TEST
    tr_data,tr_labels,lbl2int = conll_reader(dp_train)
    dev_data,dev_labels,_ = conll_reader(dp_dev,lbl2int)
    te_data,te_labels,_ = conll_reader(dp_test,lbl2int)
    vocab = list(np.load(ps.VOCAB))
    word2int = vocab2int(vocab,len(vocab) - 1)
    dispatcher.add_auxiliary(dataset,tr_data,tr_labels,dispatcher.main_task.max_len,lbl2int,word2int,dev_data,dev_labels)
    return dispatcher
