import datetime
import random
import numpy as np

import cachai.utils.models as M
import cachai.utils.constants as C
from cachai.core.advisor import Advisor
from cachai.simulator.ttl_simulator import TTLSimulator
from cachai.utils.logger import create_logger
from cachai.utils.evaluation import evaluate_loss
from cachai.utils.file_system import standardize_path, write_config


def run_experiment(config: M.ExperimentConfig):
    experiment_name = config.experiment_name
    output_dir = f'{config.output_dir}/{standardize_path(experiment_name)}'
    write_config(config, output_dir)

    experiment_logger = create_logger(
        name=C.EXPERIMENT_LOGGER,
        output_dir=output_dir,
        schema=M.ExperimentLogSchema
    )

    advisor = Advisor(config.advisor_config, output_dir)
    simulator = TTLSimulator(config.simulator_config, output_dir)

    for i in range(config.simulator_config.operations_count):
        key, X, y_true = simulator.generate()
        y_pred = advisor.predict(X, key)
        initial_time = datetime.timedelta(seconds=0)
        advisor.observe(initial_time, C.ObservationType.WRITE.value, key, {C.Y_PRED: y_pred})

        observation_time, observation_type, hits = simulator.feedback(y_true, y_pred)
        for time in range(int(observation_time.total_seconds())):
            if random.random() < config.simulator_config.hit_rate:
                advisor.observe(time, C.ObservationType.HIT, key, {C.Y_PRED: y_pred})
        advisor.observe(observation_time, observation_type, key)

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
