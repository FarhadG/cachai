import src.utils.models as M
import src.utils.constants as C


def debugger_strategy_tuner(config: M.ExperimentConfig, run_experiment):
    def update_params(config, params):
        config_clone = config.model_copy(deep=True)
        config_clone.cachai_config.strategy_config.params = M.DebuggerStrategy.Params(**params)
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
