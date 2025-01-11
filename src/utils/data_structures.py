from pydantic import BaseModel
from collections import defaultdict, deque
from typing import Any, List, Optional


class KeyedBuffer:
    GLOBAL_KEY = '__GLOBAL__'

    class Params(BaseModel):
        per_key: bool = False
        max_length: int = 10

    def __init__(self, params):
        self._per_key = params.per_key
        self._max_length = params.max_length
        self._storage = defaultdict(lambda: deque(maxlen=self._max_length))

    def _get_key(self, key):
        if self._per_key and key is None:
            raise ValueError('Key is required when per_key=True')
        return self.GLOBAL_KEY if key is None else key

    def append(self, value, key=None):
        self._storage[self._get_key(key)].append(value)

    def get(self, key=None):
        return list(self._storage[self._get_key(key)])


class KeyedDict:
    GLOBAL_KEY = '__global__'

    class Params(BaseModel):
        per_key: bool = False
        initial_value: float = 0.01

    def __init__(self, params):
        self._params = params
        self.reset()

    def reset(self):
        self._storage = {} if self._params.per_key else {
            self.GLOBAL_KEY: self._params.initial_value
        }

    def _validate(self, key):
        if self._params.per_key and key is None:
            raise ValueError('Key is required when per_key mode is enabled')

    def set(self, value, key=None):
        self._validate(key)
        if self._params.per_key:
            self._storage[key] = value
        else:
            self._storage[self.GLOBAL_KEY] = value

    def get(self, key=None):
        self._validate(key)
        if self._params.per_key:
            return self._storage.get(key, self._params.initial_value)
        return self._storage[self.GLOBAL_KEY]
