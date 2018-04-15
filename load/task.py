from load.core import Dataset
import random
import itertools
import numpy as np


def chain(*args):
    if len(args) == 1:
        return list(itertools.chain.from_iterable(*args))
    else:
        return list(itertools.chain(*args))


class Task:
    def split(self,  p_size_train=0.66):
        raise NotImplementedError()

    def sents(self):
        """sents"""
        raise NotImplementedError()

    @property
    def train_len(self):
        """train_len"""
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()

    def batch_iter(self, batch_size):
        raise NotImplementedError()

    def __len__(self):
        """__len__"""
        raise NotImplementedError()


class TrValTask(Task):
    def __init__(self, name, tr_data, tr_labels, val_data, val_labels, max_len, lbl2i, w2i):
        def word2int(sents):
            sents = list(map(lambda s: list(map(
                lambda w: w2i(w.lower()), s)), sents))
            return sents

        def pad(l, length, fill_val):
            return l + [fill_val]*(length-len(l))

        def pad_all(ls, length, fill_val):
            padded = list(map(lambda l: pad(l, length, fill_val), ls))
            return np.array(padded)

        self.tr_data = tr_data
        self.tr_labels = tr_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.max_len = max_len
        self.w2i = w2i
        self.lbl2i = lbl2i
        self.train_set = None
        self.val_set = None
        self.name = name
        train_data = tr_data
        train_data = word2int(train_data)
        train_labels = tr_labels
        dev_data = val_data
        dev_data = word2int(dev_data)
        dev_labels = val_labels
        train_data = pad_all(train_data, self.max_len, -1)
        train_labels = pad_all(train_labels, self.max_len, 0)
        dev_data = pad_all(dev_data, self.max_len, -1)
        dev_labels = pad_all(dev_labels, self.max_len, 0)
        assert train_data.shape == train_labels.shape, \
            '{}: Train data shape {}, train label shape {}'.format(
                self.name, train_data.shape, train_labels.shape)
        assert dev_data.shape == dev_labels.shape, \
            '{}: val data shape {}, val label shape {}'.format(
                self.name, train_data.shape, train_labels.shape)
        self.train_set = Dataset(data=train_data, target=train_labels)
        self.val_set = Dataset(data=val_data, target=val_labels)

    def split(self, p_size_train=0.66):
        print("Data in Object already split")
        pass

    def sents(self):
        """sents"""
        return self.val_data

    @property
    def train_len(self):
        """train_len"""
        return len(self.tr_data)

    def sample(self, batch_size):
        return self.train_set.sample(batch_size)

    def batch_iter(self, batch_size):
        batches = range(0, len(self.val_set) - len(self.val_set) % batch_size,
                        batch_size)
        for b in batches:
            yield self.val_set[b:b+batch_size]

    def __len__(self):
        """__len__"""
        return len(self.val_data) + len(self.train_data)

class TrValMainTask(Task):
    def __init__(self, tr_data, tr_labels, val_data, val_labels, max_len, e2i):
        if not isinstance(tr_data, dict):
            # the data is not a dictionary so lets pretend
            # list is seq so order presevered by enurmerate
            data = {i: datum for i, datum in enumerate(data)}
            labels = {i: label for i, label in enumerate(labels)}
            assert labels.keys() == data.keys(), \
                'Data and labels does not correspond'
        self.train_keys = None
        self._val_set = None
        self.max_len = max_len
        self.e2i = e2i

        self.tr_data = tr_data
        self.tr_labels = tr_labels
        self.val_data = val_data
        self.val_labels = val_labels

        self.tr_keys = list(self.tr_data.keys() & self.tr_labels.keys())
        self.val_keys = list(self.val_data.keys() & self.val_labels.keys())

        self._length = len(self.tr_keys) + len(self.val_keys)
        def sent2int(sents):
            sents = list(map(lambda s: list(
                map(self.e2i, s.lower().split(' '))), sents))
            return sents

        def word2int(sents):
            sents = list(map(lambda s: list(
                map(self.e2i, map(str.lower, s))), sents))
            return sents

        def pad(l, length, fill_val):
            return l + [fill_val]*(length-len(l))

        def pad_all(ls, length, fill_val):
            padded = list(map(lambda l: pad(l, length, fill_val), ls))
            return np.array(padded)

        if len(self.tr_data[self.tr_keys[0]][0].split(' ')) > 1:
            train_data = chain([self.tr_data[k] for k in self.tr_keys])
            train_label = chain([self.tr_labels[k] for k in self.tr_keys])
            val_data = [self.val_data[k][0] for k in self.val_keys]
            val_label = [self.val_labels[k][0] for k in self.val_keys]
            train_data = sent2int(train_data)
            val_data = sent2int(val_data)
        else:
            train_data = [self.tr_data[k] for k in train_keys]
            train_label = [self.tr_labels[k] for k in train_keys]
            val_data = [self.val_data[k] for k in val_keys]
            val_label = [self.val_labels[k] for k in val_keys]
            train_data = word2int(train_data)
            val_data = word2int(val_data)
        train_data = pad_all(train_data, self.max_len, -1)
        train_label = pad_all(train_label, self.max_len, 0)
        val_data = pad_all(val_data, self.max_len, -1)
        val_label = pad_all(val_label, self.max_len, 0)
        assert(train_data.shape == train_label.shape)
        assert(val_data.shape == val_label.shape)
        self._train_set = Dataset(data=train_data, target=train_label)
        self._val_set = Dataset(data=val_data, target=val_label)

    def split(self, p_size_train=0.66, val_keys=None):
        print("Data in Object already split")
        pass

    @property
    def sents(self):
        """sents"""
        return [self.data[key][0] for key in self.val_keys]

    @property
    def train_len(self):
        """train_len"""
        return len(self._train_set)

    def sample(self, batch_size):
        return self._train_set.sample(batch_size)

    def batch_iter(self, batch_size):
        batches = range(0,
                        len(self._val_set) - len(self._val_set) % batch_size,
                        batch_size)
        for b in batches:
            yield self._val_set[b:b+batch_size]

    def __len__(self):
        """__len__"""
        return self._length

class TrTask(Task):
    def __init__(self, data, labels, max_len, e2i):
        if not isinstance(data, dict):
            # the data is not a dictionary so lets pretend
            # list is seq so order presevered by enurmerate
            data = {i: datum for i, datum in enumerate(data)}
            labels = {i: label for i, label in enumerate(labels)}
            assert labels.keys() == data.keys(), \
                'Data and labels does not correspond'
        self._length = len(data)
        self.train_keys = None
        self.val_keys = None
        self._train_set = None
        self._val_set = None
        self.data = data
        self.labels = labels
        self.max_len = max_len
        self.e2i = e2i

    def split(self, p_size_train=0.66, val_keys=None):
        def sent2int(sents):
            sents = list(map(lambda s: list(
                map(self.e2i, s.lower().split(' '))), sents))
            return sents

        def word2int(sents):
            sents = list(map(lambda s: list(
                map(self.e2i, map(str.lower, s))), sents))
            return sents

        def pad(l, length, fill_val):
            return l + [fill_val]*(length-len(l))

        def pad_all(ls, length, fill_val):
            padded = list(map(lambda l: pad(l, length, fill_val), ls))
            return np.array(padded)
        keys = list(self.data.keys() & self.labels.keys())
        if val_keys is None:
            random.shuffle(keys)
            train_size = int(len(self)*p_size_train)
            train_keys = keys[train_size:]
            self.train_keys = train_keys
            val_keys = keys[:train_size]
            self.val_keys = val_keys
        else:
            train_keys = list(set(keys) - set(val_keys))
            self.train_keys = train_keys
            self.val_keys = val_keys
        if len(self.data[train_keys[0]][0].split(' ')) > 1:
            train_data = chain([self.data[k] for k in train_keys])
            train_label = chain([self.labels[k] for k in train_keys])
            val_data = [self.data[k][0] for k in val_keys]
            val_label = [self.labels[k][0] for k in val_keys]
            train_data = sent2int(train_data)
            val_data = sent2int(val_data)
        else:
            train_data = [self.data[k] for k in train_keys]
            train_label = [self.labels[k] for k in train_keys]
            val_data = [self.data[k] for k in val_keys]
            val_label = [self.labels[k] for k in val_keys]
            train_data = word2int(train_data)
            val_data = word2int(val_data)
        train_data = pad_all(train_data, self.max_len, -1)
        train_label = pad_all(train_label, self.max_len, 0)
        val_data = pad_all(val_data, self.max_len, -1)
        val_label = pad_all(val_label, self.max_len, 0)
        assert(train_data.shape == train_label.shape)
        assert(val_data.shape == val_label.shape)
        self._train_set = Dataset(data=train_data, target=train_label)
        self._val_set = Dataset(data=val_data, target=val_label)

    @property
    def sents(self):
        """sents"""
        return [self.data[key][0] for key in self.val_keys]

    @property
    def train_len(self):
        """train_len"""
        return len(self._train_set)

    def sample(self, batch_size):
        return self._train_set.sample(batch_size)

    def batch_iter(self, batch_size):
        batches = range(0,
                        len(self._val_set) - len(self._val_set) % batch_size,
                        batch_size)
        for b in batches:
            yield self._val_set[b:b+batch_size]

    def __len__(self):
        """__len__"""
        return self._length


class TestTask(Task):
    def __init__(self, data, labels, max_len, e2i):
        def sent2int(sents):
            sents = list(map(lambda s: list(
                map(e2i, s.lower().split(' '))), sents))
            return sents

        def word2int(sents):
            sents = list(map(lambda s: list(
                map(e2i, map(str.lower, s))), sents))
            return sents

        def pad(l, length, fill_val):
            return l + [fill_val]*(length-len(l))

        def pad_all(ls, length, fill_val):
            padded = list(map(lambda l: pad(l, length, fill_val), ls))
            return np.array(padded)
        self.data = data
        self.labels = labels
        self.max_len = max_len
        self.e2i = e2i
        keys = list(data.keys() & labels.keys())
        self.keys = keys
        self._length = len(keys)
        if isinstance(data[next(iter(data))][0],list):
            print(data[next(iter(data))])
            data = [self.data[k][0] for k in keys]
            label = [self.labels[k][0] for k in keys]
            data = sent2int(data)
        else:
            data = [self.data[k] for k in keys]
            label = [self.labels[k] for k in keys]
            data = word2int(data)
        data = pad_all(data, self.max_len, -1)
        label = pad_all(label, self.max_len, 0)
        assert(data.shape == label.shape)
        self._set = Dataset(data=data, target=label)

    @property
    def sents(self):
        """sents"""
        return np.array([self.data[key][0] for key in self.keys])

    def sents_by_key(self,keys):
        return np.array([self.data[key][0] for key in self.keys if key in keys])

    @property
    def train_len(self):
        """train_len"""
        pass

    def sample(self, batch_size):
        pass

    def batch_iter(self, batch_size,keys=None):
        if keys is not None:
            mask = [key in keys for key in self.keys]
        else:
            mask = np.ones(len(self.keys))
        batches = range(0,
                        len(self._set) - len(self._set) % batch_size,
                        batch_size)
        for b in batches:
            if mask[b]:
                yield self._set[b:b+batch_size]

    def __len__(self):
        """__len__"""
        return self._length
