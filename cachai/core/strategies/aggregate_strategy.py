import math
import numpy as np
from typing import Literal
from pydantic import BaseModel

import cachai.utils.models as M
import cachai.utils.constants as C
from cachai.core.strategies.base_strategy import BaseStrategy
from cachai.utils.data_structures import KeyedBuffer, KeyedDict


# TODO: per key, use the highest counts?
# TODO: more efficient averaging techniques over window (e.g. convolve)
# TODO: avg functions that take into account HIT/MISS over/underestimation
# An idea would be to average MISS and STALES differently to go above/under
# TODO: make params separate for each function type
class AggregrateStrategy(BaseStrategy):

    class Params(BaseModel):
        # TODO: make the options specific to each parameter
        function_type: Literal[
            'constant', 'arithmetic_mean', 'ewma',
            'min', 'max', 'median', 'mode', 'update_risk'
        ]
        per_key: bool = True
        max_length: int = 10
        initial_value: float = 10
        ewma_alpha: float | None = None
        update_risk_threshold: float | None = None
        max_value: float | None = 1e10

    def __init__(self, params: Params):
        self._params = params
        self._buffer = KeyedBuffer(params)
        self._ttl = KeyedDict(KeyedDict.Params(
            per_key=params.per_key,
            initial_value=params.initial_value
        ))

    def predict(self, X, key=None):
        return self._ttl.get(key)

    def update(self, observation_time, observation_type, key, stored_value, y_feedback):
        self._buffer.append(y_feedback, key)
        aggregate_function = getattr(self, self._params.function_type, None)
        if aggregate_function is None:
            raise ValueError(f'Invalid function type: {self._params.function_type}')
        buffer_batch = self._buffer.get(key)
        aggregated_value = aggregate_function(buffer_batch)
        self._ttl.set(math.ceil(aggregated_value), key)

    def constant(self, batch):
        return self._params.initial_value

    def arithmetic_mean(self, batch):
        return np.mean(batch)

    def ewma(self, batch):
        ewma_alpha = self._params.ewma_alpha or 0.5
        weights = (1-ewma_alpha)**np.arange(len(batch)-1, -1, -1)
        return np.average(batch, weights=weights)

    def min(self, batch):
        return min(batch)

    def max(self, batch):
        return max(batch)

    def median(self, batch):
        return np.median(batch)

    def mode(self, batch):
        return np.bincount(batch).argmax()

    def update_risk(self, batch):
        # TODO: try different avg functions
        mu = max(0.1, self.arithmetic_mean(batch))
        update_risk_threshold = self._params.update_risk_threshold
        return -np.log(1-update_risk_threshold)*mu

    def calculate_update_risk(self, mu, t):
        # TODO: Validate this function from paper
        return 1-np.exp(-(1/mu)*t)


# TODO: per key, use the highest counts?
# TODO: more efficient averaging techniques over window (e.g. convolve)
# TODO: avg functions that take into account HIT/MISS over/underestimation
# An idea would be to average MISS and STALES differently to go above/under
class IncrementStrategy(BaseStrategy):

    class Params(BaseModel):
        function_type: Literal['linear', 'scalar', 'power', 'exponential']
        per_key: bool = True
        initial_value: float = 10
        increment_feedback_ttl: bool = False
        factor: float = 1
        max_value: float | None = 1e10

    def __init__(self, params: Params):
        super().__init__()
        self._params = params
        self._ttl = KeyedDict(KeyedDict.Params(
            per_key=params.per_key,
            initial_value=params.initial_value
        ))

    def predict(self, X, key=None):
        return self._ttl.get(key)

    def update(self, observation_time, observation_type, key, stored_value, y_feedback):
        increment_function = getattr(self, self._params.function_type, None)
        if increment_function is None:
            raise ValueError(f'Invalid function type: {self._params.function_type}')
        # Is this logic what we want for HIT, MISS, EXPIRED, STALE, VALID_TTL
        # update_ttl only gets called under STALE, MISS, VALID_TTL
        if observation_type in {C.MISS, C.VALID_TTL}:
            # 1. we take previous TTL and increment using a function (e.g. 2 + (2)^3)
            # 2. we take previous TTL and increment using the scaled diff between y_feedback and previous TTL
            #       (e.g. 2 + C(5-2)^3)
            # TODO: should we use y_feedback or current TTL?
            current_ttl = self._ttl.get(key)
            if self._params.increment_feedback_ttl:
                ttl_to_increment = max(y_feedback - current_ttl, current_ttl)
            else:
                ttl_to_increment = current_ttl
            updated_ttl = current_ttl + increment_function(ttl_to_increment, self._params.factor)
            self._ttl.set(updated_ttl, key)
        elif observation_type in {C.STALE}:
            # TODO: Custom decrement functionality and not just reset
            self._ttl.set(self._params.initial_value, key)

    def linear(self, x, factor=1):
        return factor

    def scalar(self, x, factor=1/5):
        return x*factor

    def power(self, x, factor=1/10):
        return x**factor

    def exponential(self, x, factor=1.1):
        return min(factor**x, self._params.max_value)
