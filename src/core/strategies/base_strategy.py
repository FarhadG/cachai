import datetime
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

import src.utils.constants as C
import src.utils.models as M
from src.utils.logger import create_logger


@dataclass
class ObservedKeyValue:
    stored_time: datetime
    y_pred: float
    hits: int


class BaseStrategy(ABC):

    def __init__(self, params, output_dir):
        self._params = params
        self._observed_keys = {}
        self._strategy_logger = create_logger(
            name=C.STRATEGY_LOGGER,
            output_dir=output_dir,
            schema=M.StrategyLogSchema
        )

    @abstractmethod
    def predict(self, key, info={}) -> float:
        pass

    @abstractmethod
    def update(self, observation_time, observation_type, key, stored_value, y_feedback, info={}):
        pass

    def observe(self, observation_time, observation_type, key, info={}):
        stored_value = self._observed_keys.get(key, None)
        no_stored_value = stored_value is None
        observation_type_is_write = observation_type in {C.WRITE, C.READ_WRITE}
        if no_stored_value or observation_type_is_write:
            stored_value = ObservedKeyValue(
                hits=0,
                stored_time=observation_time,
                y_pred=info[C.Y_PRED]
            )
            self._observed_keys[key] = stored_value

        if observation_type == C.HIT:
            stored_value.hits += 1
        elif observation_type in {C.STALE, C.MISS, C.VALID_TTL}:
            y_feedback = (observation_time - stored_value.stored_time).total_seconds()
            self.update(observation_time, observation_type, key, stored_value, y_feedback, {
                C.X: info[C.X],
            })
