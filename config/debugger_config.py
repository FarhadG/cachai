import src.utils.models as M
import src.utils.constants as C
from src.core.strategies.debugger_strategy import DebuggerStrategy
import config.base_config as BaseConfig

debugger_experiments = [
    M.ExperimentConfig(
        experiment_name='Debugger Test',
        simulator_config=M.TTLSimulatorConfig(
            debug=True,
            records_count=BaseConfig.records_count,
            operations_count=BaseConfig.operations_count,
            record_mean_range=BaseConfig.record_mean_range,
            record_var_range=BaseConfig.record_var_range
        ),
        cachai_config=M.CachaiConfig(
            strategy_config=M.StrategyConfig(
                name=C.DEBUGGER_STRATEGY,
                params=DebuggerStrategy.Params(
                    offset=0
                )
            )
        )
    )
]
