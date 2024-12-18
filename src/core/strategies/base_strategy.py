import datetime
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

import src.utils.constants as C


@dataclass
class ObservedKeyValue:
    stored_time: datetime
    y_pred: float
    hits: int


class BaseStrategy(ABC):

    @abstractmethod
    def predict(self, key, info={}) -> float:
        pass

    def observe(self, observation_time, observation_type, key, info):
        stored_value = self._observed_keys.get(key, None)
        no_stored_value = stored_value is None
        observation_type_is_write = observation_type in {
            C.ObservationType.WRITE.value,
            C.ObservationType.READ_WRITE.value
        }
        if no_stored_value or observation_type_is_write:
            stored_value = ObservedKeyValue(
                hits=0,
                stored_time=observation_time,
                y_pred=info[C.Y_PRED]
            )
            self._observed_keys[key] = stored_value

        if observation_type == C.ObservationType.HIT.value:
            stored_value.hits += 1
        elif observation_type in {
            C.ObservationType.STALE.value,
            C.ObservationType.MISS.value,
            C.ObservationType.VALID_TTL.value
        }:
            y_feedback = (observation_time - stored_value.stored_time).total_seconds()
            self.update(observation_time, observation_type, key, stored_value, y_feedback)
