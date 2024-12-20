import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, TransformerMixin

import src.utils.models as M
import src.utils.constants as C
from src.core.strategies.base_strategy import BaseStrategy


class _DebuggerStrategy(BaseStrategy):

    class Params(BaseModel):
        offset: int = 0

    def __init__(self, params, output_dir):
        super().__init__(params, output_dir)

    def predict(self, key, X, info={}):
        return info[C.Y_TRUE] + self._params.offset

    def update(self, observation_time, observation_type, key, stored_value, y_feedback, info):
        pass


class DebuggerStrategy(BaseStrategy):

    class Params(BaseModel):
        offset: int = 0

    def __init__(self, params, output_dir):
        super().__init__(params, output_dir)
        n_features = 1
        dummy_X = np.zeros((1, n_features))
        dummy_y = np.array([1])
        self._scaler = IncrementalStandardScaler()
        self._scaler.partial_fit(dummy_X)
        # TODO: hyper-parameter tuning
        base_params = {
            "learning_rate": "constant",
            "eta0": 0.01,
            "alpha": 1e-4,
            "penalty": "l1",
            "tol": 1e-3,
            "max_iter": 1,
            "warm_start": True
        }
        self._model = SGDRegressor(**base_params)
        self._model.partial_fit(dummy_X, dummy_y)

    def predict(self, key, X, info={}):
        X_scaled = self._scaler.transform(X)
        prediction = self._model.predict([X_scaled])[0]
        return prediction

    def update(self, observation_time, observation_type, key, stored_value, y_feedback, info={}):
        X = self._scaler.transform(info[C.X])
        y = info[C.Y_TRUE]
        self._model.partial_fit(X, y)


class IncrementalStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0

    def partial_fit(self, X):
        if self.mean_ is None:
            self.mean_ = np.mean(X, axis=0)
            self.var_ = np.var(X, axis=0)
            self.n_samples_seen_ = X.shape[0]
        else:
            n_samples = X.shape[0]
            new_mean = np.mean(X, axis=0)
            new_var = np.var(X, axis=0)

            total_samples = self.n_samples_seen_ + n_samples
            updated_mean = (self.n_samples_seen_ * self.mean_ + n_samples * new_mean) / total_samples

            self.var_ = (self.n_samples_seen_ * (self.var_ + (self.mean_ - updated_mean) ** 2) +
                         n_samples * (new_var + (new_mean - updated_mean) ** 2)) / total_samples
            self.mean_ = updated_mean
            self.n_samples_seen_ = total_samples
        return self

    def transform(self, X):
        if self.mean_ is None or self.var_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        epsilon = 1e-9  # Avoid division by zero
        return (X - self.mean_) / (np.sqrt(self.var_) + epsilon)

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)
