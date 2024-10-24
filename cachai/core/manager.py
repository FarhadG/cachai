import numpy as np

import cachai.utils.types as T
import cachai.utils.models as M
import cachai.utils.constants as C
from cachai.simulators.ttl_simulator import TTLSimulator
from cachai.strategies.base_strategy import BaseStrategy
from cachai.strategies.debugger_strategy import DebuggerStrategy
from cachai.strategies.heuristics_strategy import ConstantStrategy


class Cachai(BaseStrategy):
    def __init__(self, config: M.StrategyConfig):
        self._strategy = strategy_from_config(config)

    def predict(self, X: np.array, info: dict = {}) -> np.array:
        return self._strategy.predict(X, info)

    def observe(
        self,
        observation_time: int,
        observation_type: T.ObservationType,
        hits: int,
        y_prev: float,
        info: dict = {}
    ) -> None:
        self._strategy.observe(observation_time, observation_type, hits, y_prev, info)


def strategy_from_config(strategy_config: M.StrategyConfig) -> None:
    params = strategy_config.params
    if strategy_config.type == 'DebuggerStrategy':
        return DebuggerStrategy()
    elif strategy_config.type == 'ConstantStrategy':
        return ConstantStrategy(params)
    else:
        raise ValueError(f'Unknown strategy type: {strategy_config.type}')
