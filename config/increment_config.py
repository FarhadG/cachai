import cachai.utils.models as M
import cachai.utils.constants as C
from cachai.core.strategies.aggregate_strategy import IncrementStrategy
import config.base_config as BaseConfig

increment_experiments_configs = [
    {'function_type': 'constant'},
    # {'function_type': 'linear'},
    # {'function_type': 'polynomial'},
    # {'function_type': 'exponential'},
]

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
                        per_key=True,
                    )
                )
            )
        )
    )
