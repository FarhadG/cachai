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

    # TODO: standardize between key/record value/payload naming
    for i in range(len(simulator.data)):
        key, X, y_true = simulator.data.iloc[i]
        # TODO: figure out how to pass the right info down for observe and predict
        info = {C.Y_TRUE: y_true, C.X: X}
        y_pred = cachai.predict(key, X, info)
        info = {C.Y_TRUE: y_true, C.Y_PRED: y_pred, C.X: X}
        # write
        initial_time = datetime.timedelta(seconds=0)
        cachai.observe(initial_time, C.WRITE, key, info)
        # feedback
        observation_time, observation_type, hits = simulator.feedback(y_true, y_pred)
        for time in range(int(observation_time.total_seconds())):
            if random.random() < config.simulator_config.hit_rate:
                cachai.observe(time, C.HIT, key, info)
        cachai.observe(observation_time, observation_type, key, info)

        loss = evaluate_loss(np.array([y_true]), np.array([y_pred]))
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
