import numpy as np

import cachai.utils.models as M
import cachai.utils.constants as C
from cachai.core.strategies.aggregate_strategy import IncrementStrategy
import config.base_config as BaseConfig

increment_experiments_configs = []

for i in range(1, 20, 5):
    increment_experiments_configs.append({
        'experiment_name': f'Linear {i}',
        'function_type': 'linear',
        'factor': i
    })
for i in np.arange(0, 2, 0.25):
    increment_experiments_configs.append({
        'experiment_name': f'Scalar {i}',
        'function_type': 'scalar',
        'factor': i
    })
for i in np.arange(0, 2, 0.25):
    increment_experiments_configs.append({
        'experiment_name': f'Power {i}',
        'function_type': 'power',
        'factor': i
    })
for i in np.arange(0, 2, 0.25):
    increment_experiments_configs.append({
        'experiment_name': f'Exponential {i}',
        'function_type': 'exponential',
        'factor': i
    })

increment_experiments = []
for config in increment_experiments_configs:
    experiment_name = f'Increment {config.get("experiment_name", config["function_type"])}'
    increment_experiments.append(
        M.ExperimentConfig(
            experiment_name=experiment_name,
            simulator_config=M.TTLSimulatorConfig(
                records_count=BaseConfig.records_count,
                operations_count=BaseConfig.operations_count,
                record_mean_range=BaseConfig.record_mean_range,
                record_var_range=BaseConfig.record_var_range
            ),
            advisor_config=M.AdvisorConfig(
                strategy_config=M.StrategyConfig(
                    name=C.INCREMENT_STRATEGY,
                    params=IncrementStrategy.Params(
                        function_type=config['function_type'],
                        factor=config['factor'],
                        per_key=True,
                    )
                )
            )
        )
    )
