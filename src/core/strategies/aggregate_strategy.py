import math
import numpy as np
from typing import Literal
from pydantic import BaseModel

import src.utils.constants as C
from src.core.strategies.base_strategy import BaseStrategy
from src.utils.data_structures import KeyedBuffer, KeyedDict


# TODO: per key, use the highest counts?
# TODO: more efficient averaging techniques over window (e.g. convolve)
# TODO: avg functions that take into account HIT/MISS over/underestimation
# An idea would be to average MISS and STALES differently to go above/under
# TODO: make params separate for each function type
class AggregrateStrategy(BaseStrategy):

    class Params(BaseModel):
        # TODO: make the options specific to each parameter
        function_type: Literal[
            'constant', 'arithmetic_mean', 'ewma',
            'min', 'max', 'median', 'mode', 'update_risk'
        ]
        per_key: bool = True
        max_length: int = 10
        initial_value: float = 10
        ewma_alpha: float | None = None
        update_risk_threshold: float | None = None
        max_value: float | None = 1e10

    def __init__(self, params: Params, output_dir):
        super().__init__(output_dir)
        self._params = params or self.Params()
        self._buffer = KeyedBuffer(self._params)
        self._ttl = KeyedDict(KeyedDict.Params(
            per_key=self._params.per_key,
            initial_value=self._params.initial_value
        ))

    def predict(self, key, info={}):
        return self._ttl.get(key)

    def update(self, observation_time, observation_type, key, stored_value, y_feedback, info):
        self._buffer.append(y_feedback, key)
        aggregate_function = getattr(AggregrateStrategy, self._params.function_type, None)
        if aggregate_function is None:
            raise ValueError(f'Invalid function type: {self._params.function_type}')
        buffer_batch = self._buffer.get(key)
        aggregated_value = aggregate_function(buffer_batch, self._params)
        self._ttl.set(math.ceil(aggregated_value), key)

    @staticmethod
    def constant(batch, params):
        return params.initial_value

    @staticmethod
    def arithmetic_mean(batch, params):
        return np.mean(batch)

    @staticmethod
    def ewma(batch, params):
        ewma_alpha = params.ewma_alpha or 0.5
        weights = (1-ewma_alpha)**np.arange(len(batch)-1, -1, -1)
        return np.average(batch, weights=weights)

    @staticmethod
    def min(batch, params):
        return min(batch)

    @staticmethod
    def max(batch, params):
        return max(batch)

    @staticmethod
    def median(batch, params):
        return np.median(batch)

    @staticmethod
    def mode(batch, params):
        return np.bincount(batch).argmax()

    @staticmethod
    def update_risk(batch, params):
        # TODO: try different avg functions
        mu = max(0.1, AggregrateStrategy.arithmetic_mean(batch, params))
        update_risk_threshold = params.update_risk_threshold
        return -np.log(1-update_risk_threshold)*mu

    @staticmethod
    def calculate_update_risk(mu, t):
        # TODO: Validate this function from paper
        return 1-np.exp(-(1/mu)*t)

    @staticmethod
    def params_tuner(config, run_experiment):
        def update_params(config, params):
            config_clone = config.model_copy(deep=True)
            updated_params = AggregrateStrategy.Params(**params)
            config_clone.cachai_config.strategy_config.params = updated_params
            return config_clone

        def objective(trial):
            params = {
                'function_type': trial.suggest_categorical('function_type', [
                    'constant', 'arithmetic_mean', 'ewma',
                    'min', 'max', 'median', 'mode', 'update_risk'
                ]),
                # FIX: per key throws strategy off during tuning
                # 'per_key': trial.suggest_categorical('per_key', [True, False]),
                'per_key': True,
                'max_length': trial.suggest_int('max_length', 5, 50),
                'initial_value': trial.suggest_float('initial_value', 0.1, 20),
                'ewma_alpha': trial.suggest_float('ewma_alpha', 0.05, 0.95),
                'update_risk_threshold': trial.suggest_float('update_risk_threshold', 0.05, 0.95),
                'max_value': 1e10,
            }
            updated_config = update_params(config, params)
            result = run_experiment(updated_config)
            return result[C.RMSE]
        return objective, update_params
