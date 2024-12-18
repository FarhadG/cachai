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

    for i in range(config.simulator_config.operations_count):
        # prediction
        key, X, y_true = simulator.generate()
        y_pred = cachai.predict(key, {C.X: X, C.Y_TRUE: y_true})
        # write
        initial_time = datetime.timedelta(seconds=0)
        cachai.observe(initial_time, C.ObservationType.WRITE.value, key, {C.Y_PRED: y_pred})
        # feedback
        observation_time, observation_type, hits = simulator.feedback(y_true, y_pred)
        for time in range(int(observation_time.total_seconds())):
            if random.random() < config.simulator_config.hit_rate:
                cachai.observe(time, C.ObservationType.HIT, key, {C.Y_PRED: y_pred})
        cachai.observe(observation_time, observation_type, key)

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
