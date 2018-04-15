#!/usr/bin/env python

import paths as ps
from collections import defaultdict
import json
import re
import functools
import itertools

def chain(*args):
    if len(args) == 1:
        l = itertools.chain.from_iterable(*args)
    else:
        l = itertools.chain(*args)
    return list(l)


def lazy_property(func):
    attribute = '_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self,attribute):
            setattr(self,attribute,func(self))
        return getattr(self,attribute)
    return wrapper

class ant_reader(object):
    def __init__(self,null_cls=0,positive_cls=1,prefix=True,before=-1,after=-1,data_path=ps.ESC_TRAIN):
        self.null_cls = null_cls
        self.positive_cls = positive_cls
        self.data_path = data_path
        self.data
        self._length = len(self.data)
        self._before = before
        self._after = after
        self._prefix = prefix

    @property
    def data(self):
        if not hasattr(self,'_data'):
            data = defaultdict(list)
            with open(self.data_path,'r') as raw_data:
                for datum in raw_data:
                    d = json.loads(datum)
                    try:
                        annotators = d['annotations']
                        sluiceId = "{0[file]}_{0[line]}_{0[treeNode]}".format(d['metadata'])
                        data[sluiceId] = []
                    except:
                        continue
                    for i in range(len(annotators)):
                        di = d
                        di['annotations'] = [annotators[i]]
                        data[sluiceId].append(di)
            self._data = data
        return self._data

    def sluice_sentence(self):
        sluice_sents = {}
        for key, datum in self.data.items():
            try:
                sluice_sents[key] = datum['match']['sentence']['string']
            except:
                continue
        return sluice_sents

    def truncated_sentences(self,before=None,after=None,prefix=None):
        """ Re constructs raw sentence in text
            params:
                before:
                    (num) how many sentences before the match should be included
                          if negative all are included
                    (None) no sentences before match included
                after:
                    (num) how many sentences after the match should be included
                          if negative all are included
                    (None) no setences after match included
                prefix:
                    (bool) add WH-remant to front of sentence
            NOTE:
                params are store in object for consitency with subsequent calls to max length
            NOTE: all None will use init before,after,prefix
        """
        if before is None and after is None and prefix is None:
            before = self._before
            after = self._after
            prefix = self._prefix
        if before == 0:
            raise ValueError('Before must be != 0 or None')
        if after == 0:
            raise ValueError('After must be != 0 or None')
        self._before = before
        self._after = after
        self._prefix = prefix
        raw_sentences = defaultdict(list)
        for key,datum in self.data.items():
            raw_sentences[key] = []
            for annotator in datum:
                sent = ' '.join(self.truncate_datum(annotator))
                sent = re.sub('[ ]+',' ',sent)
                raw_sentences[key].append(sent)
        self.sentences = raw_sentences
        return raw_sentences


    @property
    def max_length(self):
        """ Computes maximum length sequence
            will respect boundaries set by prefix,before and, after
            default is all sentences with no prefix
        """
        def length(s):
            """ Computes length of a list of strings assuming delimiter is space
            """
            return len(s.split(' '))

        max_len = 0
        for datum in self.data.values():
            for annotator in datum:
                sent = ' '.join(self.truncate_datum(annotator))
                sent = re.sub('[ ]+',' ',sent)
                max_len = max(max_len,length(sent))
        return max_len

    def __len__(self):
        return self._length


    @property
    def token_lvl_labels(self):
        """ Generate token level BI0-labels for raw sentences
        """
        def to_null(ls):
            return list(map(lambda s: self.null_cls,ls.split(' ')))

        labels = defaultdict(list)
        for key,datum in self.data.items():
            labels[key] = []
            for annotator in datum:
                # only include datum with annotated antecedent
                try:
                    annot = annotator['annotations'][0]
                except:
                    break
                try:
                    ant = annot['tags']['Antecedent'][0]
                except:
                    break

                lineNos = [x["lineNo"] for x in ant["offsets"]]
                uniqLineNos = set(lineNos)
                if len(uniqLineNos) > 1:
                    break

                lineText = ant["offsets"][0]["lineText"]

                starts = [x["start"] for x in ant["offsets"]]
                ends = [x["end"] for x in ant["offsets"]]

                start = min(starts)
                end = max(ends)

                ant_start = lineText[:start].count(' ')
                ant_end = 1 + ant_start + lineText[start:end].count(' ')

                sent = self.truncate_datum(annotator)

                try:
                    ant_idx = sent.index(lineText)
                except ValueError as snf:
                    break

                label = list(map(to_null,sent))
                label[ant_idx][ant_start] = 1
                label[ant_idx][(ant_start+1):ant_end] = [2] * (ant_end - ant_start - 1)

                label = chain(label)
                labels[key].append(label)
            if not labels[key]:
                del labels[key]
        return labels

    def truncate_datum(self,annotator):
        before = []
        after = []

        if self._prefix:
            prefix = annotator['match']['sluice']['string']

        if isinstance(self._before,int):
            if self._before > 0:
                before = [sent['string'] for sent in annotator['before'][-self._before:]]
            else:
                before = [sent['string'] for sent in annotator['before']]

        match = annotator['match']['sentence']['string']

        if isinstance(self._after,int):
            if self._after > 0:
                after = [sent['string'] for sent in annotator['after'][:self._after]]
            else:
                after = [sent['string'] for sent in annotator['after']]

        if self._prefix:
            return chain([prefix],before,[match],after)
        else:
            return chain(before,[match],after)

    @staticmethod
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
