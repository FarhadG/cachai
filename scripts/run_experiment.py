import optuna
import datetime
import random
import numpy as np

import src.utils.models as M
import src.utils.constants as C
from src.core.cachai import Cachai
from src.simulator.ttl_simulator import TTLSimulator
from src.utils.logger import create_logger
from src.utils.evaluation import evaluate_loss
from src.utils.file_system import standardize_path, write_config


def run_experiment(config: M.ExperimentConfig):
    experiment_name = config.experiment_name
    output_dir = f'{config.output_dir}/{standardize_path(experiment_name)}'
    write_config(config, output_dir)

    experiment_logger = create_logger(
        name=C.EXPERIMENT_LOGGER,
        output_dir=output_dir,
        schema=M.ExperimentLogSchema
    )

    cachai = Cachai(config.cachai_config, output_dir)
    simulator = TTLSimulator(config.simulator_config, output_dir)

    total_loss = {C.MSE: 0, C.RMSE: 0, C.MAE: 0, C.MBE: 0}

    # TODO: standardize between key/record value/payload naming
    for i in range(len(simulator.data)):
        key, X, y_true = simulator.data.iloc[i]
        # TODO: figure out how to pass the right info down for observe and predict
        info = {C.X: X, C.Y_TRUE: y_true}
        y_pred = cachai.predict(key, info)
        # write
        info[C.Y_PRED] = y_pred
        initial_time = datetime.timedelta(seconds=0)
        cachai.observe(initial_time, C.WRITE, key, info)
        # feedback
        observation_time, observation_type, hits = simulator.feedback(y_true, y_pred)
        for time in range(int(observation_time.total_seconds())):
            if random.random() < config.simulator_config.hit_rate:
                cachai.observe(time, C.HIT, key, info)
        cachai.observe(observation_time, observation_type, key, info)

        loss = evaluate_loss(np.array([y_true]), np.array([y_pred]))
        total_loss[C.MSE] += loss[C.MSE]
        total_loss[C.RMSE] += loss[C.RMSE]
        total_loss[C.MAE] += loss[C.MAE]
        total_loss[C.MBE] += loss[C.MBE]

        experiment_logger.log(M.ExperimentLogSchema(
            **loss,
            experiment_name=experiment_name,
            iteration=i,
            observation_time=observation_time,
            observation_type=observation_type,
            key=key,
            hits=hits,
            y_true=y_true,
            y_pred=y_pred
        ))

    return total_loss


def debugger_strategy_tuner(config: M.ExperimentConfig):

    def update_params(config, params):
        config_clone = config.model_copy(deep=True)
        config_clone.cachai_config.strategy_config.params = M.DebuggerStrategy.Params(**params)
        return config_clone

    def objective(trial: optuna.Trial):
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


def tune_hyperparameters(config: M.ExperimentConfig):
    objective, update_params = debugger_strategy_tuner(config)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=2)
    best_hyperparams = study.best_params
    best_score = study.best_value
    print(f'Best hyperparameters: {best_hyperparams} with loss {best_score}')
    config = update_params(config, best_hyperparams)
    return config, best_hyperparams, study.trials


def tune_experiment(config: M.ExperimentConfig):
    if config.cachai_config.strategy_config.tune_params:
        config, best_hyperparams, tuning_results = tune_hyperparameters(config)
    return run_experiment(config)
