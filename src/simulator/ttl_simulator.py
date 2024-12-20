import datetime
import numpy as np
import pandas as pd

import src.utils.models as M
import src.utils.constants as C
from src.simulator.generators import feature_generator
from src.utils.file_system import create_dir


class TTLSimulator:

    def __init__(self, config: M.TTLSimulatorConfig, output_dir):
        self._config = config

        self._target_params_options = []
        # TODO: ability to pass in record generation functions
        for i in range(config.records_count):
            mean = np.random.randint(*config.record_mean_range)
            variance = np.random.randint(*config.record_var_range)
            key = f'key={i}__mean={mean}__var={variance}'
            self._target_params_options.append((key, mean, variance))

        data = []
        for _ in range(config.operations_count):
            target_param_index = np.random.randint(0, len(self._target_params_options))
            key, mean, var = self._target_params_options[target_param_index]
            mean_abs = np.abs(np.random.normal(mean, var))
            y_true = round(mean_abs, 3)
            X = feature_generator.generate_feature(y_true)
            data.append((key, X, y_true))
        self.data = pd.DataFrame(data, columns=[C.KEY, C.X, C.Y_TRUE])
        self.data.to_csv(f'{output_dir}/simulation.csv', index=False)

    def feedback(self, y_true, y_pred):
        observation_type = None
        observation_time = int(min(y_true, y_pred))
        if y_pred < y_true:
            observation_type = C.MISS
            observation_time += 1
        elif y_pred > y_true:
            observation_type = C.STALE
            observation_time -= 1
        else:
            observation_type = C.VALID_TTL
        hits = max(0, observation_time-1)
        observation_time = datetime.timedelta(seconds=max(0, observation_time))
        return observation_time, observation_type, hits
