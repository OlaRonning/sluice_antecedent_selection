import random
import numpy as np
from .dataset import Dataset
import itertools as it


class inter_bin:
    """
    A mapping from column names to immutable arrays of equal length.
    """

    def __init__(self,B,lengths,data):
        if not B:
            raise NotImplementedError
        self._B = B
        self._bin = np.empty((B),dtype=Dataset)
        self._length = B
        self._data_length = len(data)
        bRange = lengths.max() // B
        self._bRange = bRange
        for bin_ in self._bin:
            bin_ = Dataset()

        binIdxs = lengths // bRange - 1
        for b in range(self._B):
            self._bin[b] = Dataset()

        tot = 0
        t = binIdxs == 0
        for b in range(self._B):
            ds = self._bin[b]
            tot += sum(b==binIdxs)
            t = np.logical_or(t,b==binIdxs)
            for col in data.columns:
                d = data[col]
                if not any(b==binIdxs): continue
                ds[col] = d[b==binIdxs]

        bin_sizes = np.zeros_like(self._bin,dtype=float)
        for b in range(self._B):
            bin_sizes[b] = np.count_nonzero(binIdxs == b)
        dist = bin_sizes / lengths.shape[0]
        self._dist = dist

    def alt_range(self,start,end):
        after = list(range(start,end))
        before = list(range(start-1,-1,-1))
        missing = after[len(before):] or before[:len(after)]
        return list(it.chain(*zip(after,before)))+missing
        
    def sample(self, size):
        assert(size < self._data_length)
        start = np.random.choice(np.array(range(self._B)),1,p=self._dist)
        d = None
        for b in self.alt_range(start[0],self._B):
            ds = self._bin[b]
            try: len(ds)
            except: continue
            if len(ds) < size:
                if d is None:
                    d = ds
                else:
                    d.append(ds)
                size -= len(ds)
            else :
                indices = random.sample(range(len(ds)), size)
                if d is None:
                    d = ds
                else:
                    d.append(ds[indices])
                break
        return d

    def __len__(self):
        return self._data_length
