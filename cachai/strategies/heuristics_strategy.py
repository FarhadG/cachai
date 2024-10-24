from pydantic import BaseModel

from cachai.strategies.base_strategy import BaseStrategy


class ConstantStrategy(BaseStrategy):

    class Params(BaseModel):
        initial_ttl: int

    def __init__(self, params: Params):
        self._params = params

    def predict(self, X, info):
        return [self._params.initial_ttl]

    def observe(self, observation_time, observation_type, hits, y_prev, info):
        pass
