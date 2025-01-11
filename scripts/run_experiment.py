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
from src.utils.strategy_helpers import strategy_from_config


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


def configure_experiment(config: M.ExperimentConfig):
    tune_params_trials = config.cachai_config.strategy_config.tune_params_trials
    if tune_params_trials > 0:
        def update_params(config, params):
            config_clone = config.model_copy(deep=True)
            config_clone.cachai_config.strategy_config.params = Strategy.Params(**params)
            return config_clone

        def objective(trial):
            hyperparams = Strategy.get_hyperparams(trial)
            updated_config = update_params(config, hyperparams)
            result = run_experiment(updated_config)
            return result[C.RMSE]

        Strategy = strategy_from_config(config.cachai_config.strategy_config)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=tune_params_trials)
        config = update_params(config, study.best_params)
        print(f'Best hyperparameters: {study.best_params} with loss {study.best_value}')
    return run_experiment(config)
