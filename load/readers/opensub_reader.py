import paths
import functools
import json
import re
from collections import defaultdict

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
        """wrapper"""
        if not hasattr(self,attribute):
            setattr(self,attribute,func(self))
        return getattr(self,attribute)
    return wrapper

class opensub_reader(object):
    def __init__(self,prefix=True,data_path=paths.OPENSUB_EMBEDDED_TEST,max_len=700):
        self.data_path = data_path
        self.prefix = prefix
        self.max_len = max_len
        self.data

    @property
    def data(self):
        """data"""
        data = defaultdict(None)
        with open(self.data_path,'r') as raw_data:
            for key,datum in enumerate(raw_data):
                d = json.loads(datum)
                try:
                    example = {'sluice': d['sluice'], 'antecedent': d['antecedant'], 'text': d['entire_sluice_utterance']}
                    example['text'] = '\n'.join(example['text'].split('\n')[:-2])
                except:
                    continue
                data[key] = example
        return data

    @property
    def sentences(self):
        """sentences"""
        sentences = defaultdict(None)
        for key,datum in self.data.items():
            sentence = self.proc_str(datum['text'])
            if self.prefix:
                sentence = ' '.join([self.proc_str(datum['sluice']),sentence])
            if len(sentence.split(' ')) >  self.max_len:
                continue
            sentences[key] = sentence.split(' ')
        return sentences

    @property
    def max_length(self):
        """max_length"""
        if not hasattr(self,'_max_length'):
            lens = list(map(len,self.sentences.values()))
            max_ = functools.reduce(lambda c1,c2: max(c1,c2),lens)
            self._max_len = max_
        return self._max_len
            
    def proc_str(self,str_):
        str_ = str_.replace('\n',' ').lower().strip()
        str_.replace('...','')
        str_.replace('-','')
        str_ = re.sub('[ ]+',' ',str_)
        return str_


    @property
    def labels(self):
        """labels"""
        def subfinder(list_,pattern):
            match_idx = []
            for i in range(len(list_)):
                if list_[i] == pattern[0] and list_[i+1:i+len(pattern)] == pattern[1:]:
                    match_idx.append(i)
            return match_idx
        labels = defaultdict(None)
        for key,datum in self.data.items():
            sentence = datum['text'].replace('\n',' ').split(' ')
            ant = datum['antecedent']
            if not ant:
                continue
            ant = ant.split(' ')
            match_idx = subfinder(sentence,ant)
            if not len(match_idx) == 1:
                continue
            match_idx = match_idx[0]
            label_seq = [0] * len(sentence)
            label_seq[match_idx] = 1
            label_seq[match_idx + 1:match_idx+len(ant)] = [2] * (len(ant) -1)

            labels[key] = label_seq

        return labels

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
