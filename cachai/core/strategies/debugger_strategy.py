import numpy as np
from pydantic import BaseModel

import cachai.utils.models as M
import cachai.utils.constants as C
from cachai.core.strategies.base_strategy import BaseStrategy


class DebuggerStrategy(BaseStrategy):

    class Params(BaseModel):
        offset: int = 0

    def __init__(self, params):
        self._params = params

    def predict(self, X, key=None):
        return (X + self._params.offset)[0]

    def observe(self, observation_time, observation_type, key, info):
        pass
