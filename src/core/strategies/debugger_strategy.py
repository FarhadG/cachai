import numpy as np
from pydantic import BaseModel
from river import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, TransformerMixin

import src.utils.models as M
import src.utils.constants as C
from src.core.strategies.base_strategy import BaseStrategy
from src.utils.scalers import CustomStandardScaler, StandardScaler


class _DebuggerStrategy(BaseStrategy):

    class Params(BaseModel):
        offset: int = 0

    def __init__(self, params, output_dir):
        super().__init__(params, output_dir)

    def predict(self, key, info={}):
        return info[C.Y_TRUE] + self._params.offset

    def update(self, observation_time, observation_type, key, stored_value, y_feedback, info):
        pass


class DebuggerStrategy(BaseStrategy):

    class Params(BaseModel):
        offset: int = 0,
        tune_hyperparams: bool = False,
        hyperparams: dict = {
            # "learning_rate": "constant",
            # "eta0": 0.01,
            # "alpha": 1e-4,
            # "penalty": "l1",
            # "tol": 1e-3,
            # "max_iter": 1,
            # "warm_start": True
        }

    def __init__(self, params, output_dir):
        super().__init__(output_dir)
        self._params = params
        n_features = 1
        # dummy_X = np.zeros((1, n_features))
        dummy_X = np.zeros((1, n_features))
        dummy_y = np.array([0])
        self._scaler = CustomStandardScaler()
        self._scaler.partial_fit(dummy_X)
        self._model = SGDRegressor(**self._params.hyperparams)
        self._model.partial_fit(dummy_X, dummy_y)

    def predict(self, key, info):
        X_scaled = self._scaler.fit_transform(info[C.X])
        prediction = self._model.predict(X_scaled)[0]
        return np.round(prediction)

    def update(self, observation_time, observation_type, key, stored_value, y_feedback, info={}):
        X_scaled = self._scaler.fit_transform(info[C.X])
        y = info[C.Y_TRUE]
        self._model.partial_fit(X_scaled, [y])
