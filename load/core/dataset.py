import random
import numpy as np


class Dataset:
    """
    A mapping from column names to immutable arrays of equal length.
    """

    def __init__(self, **data):
        """__init__

        Params:
                 **data

        Returns:
        """
        self._data = {}
        self._length = None
        super().__init__()
        for column, data in data.items():
            self[column] = data

    @property
    def columns(self):
        """columns"""
        return sorted(self._data.keys())

    def copy(self):
        """copy"""
        data = {x: self[x].copy() for x in self.columns}
        return type(self)(**data)

    def sample(self, size):
        """sample

        Params:
                 size

        Returns:
        """
        indices = random.sample(range(len(self)), size)
        return self[indices]

    def append(self,data):
        """append

        Params:
                 data

        Returns:
        """
        for col in data.columns:
            np.append(self[col],data[col])


    def __len__(self):
        """__len__"""
        return self._length

    def __contains__(self, column):
        """__contains__

        Params:
                 column

        Returns:
        """
        return column in self._data

    def __getattr__(self, column):
        """__getattr__

        Params:
                 column

        Returns:
        """
        if column in self:
            return self[column]
        raise AttributeError

    def __iter__(self):
        """__iter__"""
        for index in range(len(self)):
            yield tuple(self[x][index] for x in self.columns)

    def __eq__(self, other):
        """__eq__

        Params:
                 other

        Returns:
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.columns != other.columns:
            return False
        for column in self.columns:
            if not (self[column] == other[column]).all():
                return False
        return True

    def __getitem__(self, key):
        if isinstance(key, slice):
            data = {x: self[x][key] for x in self.columns}
            return type(self)(**data)
        if isinstance(key, (tuple, list)) and isinstance(key[0], int):
            data = {x: self[x][key] for x in self.columns}
            return type(self)(**data)
        if isinstance(key, (tuple, list)) and isinstance(key[0], str):
            data = {x: self[x] for x in key}
            return type(self)(**data)
        return self._data[key].copy()

    def __setitem__(self, key, data):
        """__setitem__

        Params:
                 key
                 data

        Returns:
        """
        if isinstance(key, (tuple, list)) and isinstance(key[0], str):
            for column, data in zip(key, data):
                self[column] = data
            return
        if isinstance(key, (tuple, list)) and isinstance(key[0], int):
            raise NotImplementedError('column content is immutable')
        data = np.array(data)
        data.setflags(write=False)
        if not data.size:
            raise ValueError('must not be empty')
        if not self._length:
            self._length = len(data)
        if len(data) != self._length:
            raise ValueError('must have same length')
        self._data[key] = data

    def __delitem__(self, key):
        """__delitem__

        Params:
                 key

        Returns:
        """
        if isinstance(key, (tuple, list)):
            for column in key:
                del self._data[column]
            return
        del self._data[key]

    def __str__(self):
        """__str__"""
        message = ''
        for column in self.columns:
            message += '{} ({}):\n\n'.format(column, self[column].dtype)
            message += str(self[column]) + '\n\n'
        return message

    def __getstate__(self):
        """__getstate__"""
        return {'length': self._length, 'data': self._data}

    def __setstate__(self, state):
        """__setstate__

        Params:
                 state

        Returns:
        """
        self._length = state['length']
        self._data = state['data']
