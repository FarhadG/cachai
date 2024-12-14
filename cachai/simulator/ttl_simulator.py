import datetime
import numpy as np

import cachai.utils.models as M
import cachai.utils.constants as C
from cachai.simulator.generators import feature_generator


class TTLSimulator:

    def __init__(self, config: M.TTLSimulatorConfig, output_dir=None):
        self._config = config
        self._output_dir = output_dir
        self._target_params_options = []
        # TODO: ability to pass in record generation functions
        for i in range(config.records_count):
            mean = np.random.randint(*config.record_mean_range)
            variance = np.random.randint(*config.record_var_range)
            key = f'key={i}__mean={mean}__var={variance}'
            self._target_params_options.append((key, mean, variance))

    def feedback(self, y_true, y_pred):
        observation_type = None
        observation_time = int(min(y_true, y_pred))
        if y_pred < y_true:
            observation_type = C.ObservationType.MISS.value
            observation_time += 1
        elif y_pred > y_true:
            observation_type = C.ObservationType.STALE.value
            observation_time -= 1
        else:
            observation_type = C.ObservationType.VALID_TTL.value
        hits = max(0, observation_time-1)
        observation_time = datetime.timedelta(seconds=max(0, observation_time))
        return observation_time, observation_type, hits

    def generate(self):
        target_param_index = np.random.randint(0, len(self._target_params_options))
        key, mean, var = self._target_params_options[target_param_index]
        mean_abs = np.abs(np.random.normal(mean, var))
        y_true = round(mean_abs, 3)
        if self._config.debug:
            X = np.array([y_true])
        else:
            X = feature_generator.generate_feature(y_true)
        return key, X, y_true
