import src.utils.models as M
import src.utils.constants as C
from src.utils.logger import create_logger
from src.core.strategies.base_strategy import BaseStrategy
from src.core.strategies.debugger_strategy import DebuggerStrategy
from src.core.strategies.aggregate_strategy import AggregrateStrategy
from src.core.strategies.increment_strategy import IncrementStrategy


class Cachai(BaseStrategy):

    def __init__(self, config, output_dir):
        self._strategy = strategy_from_config(config)
        self._cachai_logger = create_logger(
            name=C.CACHAI_LOGGER,
            output_dir=output_dir,
            schema=M.CachaiLogSchema
        )

    def predict(self, X, key=None):
        return self._strategy.predict(X, key)

    def observe(self, observation_time, observation_type, key, info={}):
        self._strategy.observe(observation_time, observation_type, key, info)
        self._cachai_logger.log(M.CachaiLogSchema(
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
    elif name == C.INCREMENT_STRATEGY:
        return IncrementStrategy(params)
    else:
        raise ValueError(f'Unknown strategy type: {name}')
