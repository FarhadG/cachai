import numpy as np

import cachai.utils.models as M
import cachai.utils.constants as C
from cachai.utils.logger import create_logger
from cachai.core.strategies.base_strategy import BaseStrategy
from cachai.core.strategies.debugger_strategy import DebuggerStrategy
from cachai.core.strategies.heuristics_strategy import AggregrateStrategy


class Advisor(BaseStrategy):

    def __init__(self, config, output_dir):
        self._strategy = strategy_from_config(config)
        self._advisor_logger = create_logger(
            name=C.ADVISOR_LOGGER,
            output_dir=output_dir,
            schema=M.AdvisorLogSchema
        )

    def predict(self, X, key=None):
        return self._strategy.predict(X, key)

    def observe(self, observation_time, observation_type, key, info={}):
        self._strategy.observe(observation_time, observation_type, key, info)
        self._advisor_logger.log(M.AdvisorLogSchema(
            observation_time=observation_time,
            observation_type=observation_type,
            key=key,
        ))


def strategy_from_config(config):
    name = config.strategy_config.name
    params = config.strategy_config.params
    if name == C.DEBUGGER_STRATEGY:
        return DebuggerStrategy(params)
    elif name == C.AGGREGATE_STRATEGY:
        return AggregrateStrategy(params)
    else:
        raise ValueError(f'Unknown strategy type: {name}')
