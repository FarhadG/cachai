import cachai.utils.types as T
from cachai.strategies.heuristics_strategy import ConstantStrategy

constant_experiments = []
for i in range(0, 500, 50):
    constant_experiments.append(
        T.Experiment(
            experiment_name=f'Constant {i}',
            experiment_config=T.ExperimentConfig(
                iterations=1_000
            ),
            simulator_config=T.SimulatorConfig(
                type='TTLSimulator'
            ),
            strategy_config=T.StrategyConfig(
                type='ConstantStrategy',
                params=ConstantStrategy.Params(initial_ttl=i)
            )
        )
    )
