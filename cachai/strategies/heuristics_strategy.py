import cachai.utils.constants as C
from cachai.strategies.base_strategy import BaseStrategy


class ConstantStrategy(BaseStrategy):

    def __init__(self):
        pass

    def predict(self, X, info):
        return [5]

    def observe(self, observation_time, observation_type, hits, y_prev, info):
        pass
