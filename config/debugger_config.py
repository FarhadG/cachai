import src.utils.models as M
import src.utils.constants as C
import config.base_config as BaseConfig


debugger_experiments_configs = [
    {
        'experiment_name': C.DEBUGGER_STRATEGY,
        'strategy_config': M.StrategyConfig(
            name=C.DEBUGGER_STRATEGY
        )
    },
    {
        'experiment_name': C.DEBUGGER_REGRESSION_STRATEGY,
        'strategy_config': M.StrategyConfig(
            name=C.DEBUGGER_REGRESSION_STRATEGY,
            tune_params_trials=BaseConfig.tune_params_trials
        )
    },
]

debugger_experiments = []
for config in debugger_experiments_configs:
    debugger_experiments.append(
        M.ExperimentConfig(
            experiment_name=config['experiment_name'],
            simulator_config=M.TTLSimulatorConfig(
                debug=True,
                records_count=BaseConfig.records_count,
                operations_count=BaseConfig.operations_count,
                record_mean_range=BaseConfig.record_mean_range,
                record_var_range=BaseConfig.record_var_range
            ),
            cachai_config=M.CachaiConfig(
                strategy_config=config['strategy_config']
            )
        )
    )
