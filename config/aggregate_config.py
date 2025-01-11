import src.utils.models as M
import src.utils.constants as C
from src.core.strategies.aggregate_strategy import AggregrateStrategy
import config.base_config as BaseConfig

aggregate_experiments_configs = [
    {'function_type': 'arithmetic_mean'},
    {'function_type': 'min'},
    {'function_type': 'max'},
    {'function_type': 'median'},
    {'function_type': 'mode'},
]
for i in range(0, 500, 100):
    aggregate_experiments_configs.append({
        'experiment_name': f'Constant {i}',
        'function_type': 'constant',
        'initial_value': i
    })
for alpha in [0.25, 0.5, 0.75, 0.95]:
    aggregate_experiments_configs.append({
        'experiment_name': f'EWMA {alpha}',
        'function_type': 'ewma',
        'ewma_alpha': alpha
    })
for threshold in [0.1, 0.25, 0.5, 0.75, 0.95]:
    aggregate_experiments_configs.append({
        'experiment_name': f'Update Risk {threshold}',
        'function_type': 'update_risk',
        'update_risk_threshold': threshold
    })


aggregate_experiments = []
for config in aggregate_experiments_configs:
    experiment_name = f'AggregrateStrategy {config.get("experiment_name", config["function_type"])}'
    aggregate_experiments.append(
        M.ExperimentConfig(
            experiment_name=experiment_name,
            simulator_config=M.TTLSimulatorConfig(
                records_count=BaseConfig.records_count,
                operations_count=BaseConfig.operations_count,
                record_mean_range=BaseConfig.record_mean_range,
                record_var_range=BaseConfig.record_var_range
            ),
            cachai_config=M.CachaiConfig(
                strategy_config=M.StrategyConfig(
                    name=C.AGGREGATE_STRATEGY,
                    params=AggregrateStrategy.Params(
                        function_type=config['function_type'],
                        per_key=True,
                        ewma_alpha=config.get('ewma_alpha', None),
                        update_risk_threshold=config.get('update_risk_threshold', None)
                    ),
                    tune_params_trials=BaseConfig.tune_params_trials
                )
            )
        )
    )
