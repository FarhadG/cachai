import math
import numpy as np
from typing import Literal
from pydantic import BaseModel

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
