import numpy as np
from pydantic import BaseModel
from river import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, TransformerMixin

import src.utils.models as M
import src.utils.constants as C
from src.core.strategies.base_strategy import BaseStrategy
from src.utils.scalers import CustomStandardScaler, StandardScaler


class DebuggerStrategy(BaseStrategy):

    class Params(BaseModel):
        offset: int = 0

    def __init__(self, params, output_dir):
        super().__init__(output_dir)
        self._params = params or self.Params()

    def predict(self, key, info={}):
        return info[C.Y_TRUE] + self._params.offset

    def update(self, observation_time, observation_type, key, stored_value, y_feedback, info):
        pass


class RegressionDebuggerStrategy(BaseStrategy):

    # TODO: make these params as models dumps or keep them as pydantic models
    class Params(BaseModel):
        offset: int = 0
        learning_rate: str = 'constant'
        eta0: float = 9e-2
        alpha: float = 6e-3
        penalty: str = 'elasticnet'
        tol: float = 1e-4
        max_iter: int = 339

    def __init__(self, params, output_dir):
        super().__init__(output_dir)
        self._params = params or self.Params()
        n_features = 1
        # dummy_X = np.zeros((1, n_features))
        dummy_X = np.zeros((1, n_features))
        dummy_y = np.array([0])
        self._scaler = CustomStandardScaler()
        self._scaler.partial_fit(dummy_X)
        self._model = SGDRegressor(
            learning_rate=self._params.learning_rate,
            eta0=self._params.eta0,
            alpha=self._params.alpha,
            penalty=self._params.penalty,
            tol=self._params.tol,
            max_iter=self._params.max_iter,
            warm_start=True
        )
        self._model.partial_fit(dummy_X, dummy_y)

    def predict(self, key, info):
        X_scaled = self._scaler.fit_transform(info[C.X])
        prediction = self._model.predict(X_scaled)[0]
        return np.round(prediction)

    def update(self, observation_time, observation_type, key, stored_value, y_feedback, info={}):
        X_scaled = self._scaler.fit_transform(info[C.X])
        y = info[C.Y_TRUE]
        self._model.partial_fit(X_scaled, [y])

    @staticmethod
    def params_tuner(config, run_experiment):
        def update_params(config, params):
            config_clone = config.model_copy(deep=True)
            updated_params = RegressionDebuggerStrategy.Params(**params)
            config_clone.cachai_config.strategy_config.params = updated_params
            return config_clone

        def objective(trial):
            params = {
                'offset': 0,
                'learning_rate': trial.suggest_categorical('learning_rate', [
                    'constant', 'optimal', 'invscaling', 'adaptive'
                ]),
                'eta0': trial.suggest_float('eta0', 0.001, 0.1),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01),
                'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
                'tol': trial.suggest_categorical('tol', [1e-3, 1e-4]),
                'max_iter': trial.suggest_int('max_iter', 1, 10000),
            }
            updated_config = update_params(config, params)
            result = run_experiment(updated_config)
            return result[C.RMSE]
        return objective, update_params
