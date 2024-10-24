import numpy as np

import cachai.utils.constants as C


class TTLSimulator:

    def __init__(self, iterations=1_000):
        self._iterations = iterations
        self._target_params = [
            (50, 5),
            (200, 10),
            (400, 30),
        ]
        # means = np.linspace(10, 500, 10).astype(int)
        # std = np.arange(1, len(means) + 1)**2
        # self._target_params = np.array([means, std]).T

    def update_target_params(self, progress):
        target_params = []
        for param in self._target_params:
            mean = float(
                round(param[0]*np.sin(progress*2*np.pi/2)/(param[0]/2) + param[0], 2)
            )
            std = param[1]
            target_params.append((mean, std))
        self._target_params = target_params

    def feedback(self, y_true, y_pred):
        observation_time = int(min(y_true, y_pred)[0])
        hits = max(0, observation_time-1)
        observation_type = None
        if y_pred[0] < y_true[0]:
            observation_type = C.MISS
        elif y_pred[0] > y_true[0]:
            observation_type = C.STALE
        else:
            observation_type = C.VALID_TTL
        return observation_time, observation_type, hits

    def generate(self):
        # get target
        target_param_index = np.random.randint(0, len(self._target_params))
        target_params = self._target_params[target_param_index]
        y = np.random.normal(target_params[0], target_params[1], 1)

        # generate features from target
        num_features = 1
        correlation = 0.8
        cov_matrix = np.eye(num_features) * (1 - correlation) + np.ones((num_features, num_features)) * correlation
        features = np.random.multivariate_normal(np.ones(num_features) * y, cov_matrix)
        X = features.reshape(1, -1)
        # X = np.full((1, num_features), fill_value=target)
        return X, y
