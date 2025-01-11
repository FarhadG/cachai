import src.utils.models as M
import src.utils.constants as C
from src.utils.logger import create_logger
from src.utils.strategy_helpers import strategy_from_config


class Cachai():

    def __init__(self, config: M.CachaiConfig, output_dir):
        Strategy = strategy_from_config(config.strategy_config)
        self._strategy = Strategy(config.strategy_config.params, output_dir)
        self._cachai_logger = create_logger(
            name=C.CACHAI_LOGGER,
            output_dir=output_dir,
            schema=M.CachaiLogSchema
        )

    def predict(self, key, info={}):
        return self._strategy.predict(key, info)

    def observe(self, observation_time, observation_type, key, info={}):
        self._strategy.observe(observation_time, observation_type, key, info)
        self._cachai_logger.log(M.CachaiLogSchema(
            observation_time=observation_time,
            observation_type=observation_type,
            key=key,
        ))
