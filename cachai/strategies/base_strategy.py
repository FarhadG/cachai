import numpy as np
from abc import ABC, abstractmethod

import cachai.utils.constants as C
import cachai.utils.types as T


class BaseStrategy(ABC):

    @property
    @abstractmethod
    def NAME(self) -> str:
        pass

    @abstractmethod
    def predict(self, X: np.array, info: dict = {}) -> np.array:
        pass

    @abstractmethod
    def observe(
        self,
        observation_time: int,
        observation_type: T.ObservationType,
        hits: int,
        y_prev: float,
        info: dict = {}
    ) -> None:
        pass
