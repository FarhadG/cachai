from typing import Literal
from pydantic import BaseModel

import src.utils.models as M
import src.utils.constants as C
from src.core.strategies.base_strategy import BaseStrategy
from src.utils.data_structures import KeyedDict


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

    def __init__(self, params: Params, output_dir):
        super().__init__(params, output_dir)
        self._ttl = KeyedDict(KeyedDict.Params(
            per_key=params.per_key,
            initial_value=params.initial_value
        ))

    def predict(self, key, info={}):
        return self._ttl.get(key)

    def update(self, observation_time, observation_type, key, stored_value, y_feedback, info):
        increment_function = getattr(IncrementStrategy, self._params.function_type, None)
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
            updated_ttl = current_ttl + increment_function(ttl_to_increment, self._params)
            self._ttl.set(updated_ttl, key)
        elif observation_type in {C.STALE}:
            # TODO: Custom decrement functionality and not just reset
            self._ttl.set(self._params.initial_value, key)

    @staticmethod
    def linear(x, params):
        return params.factor

    @staticmethod
    def scalar(x, params):
        return x*params.factor

    @staticmethod
    def power(x, params):
        return x**params.factor

    @staticmethod
    def exponential(x, params):
        return min(params.factor**x, params.max_value)
