import cachai.utils.models as M
from cachai.strategies.heuristics_strategy import ConstantStrategy

constant_experiments = []
for i in range(0, 500, 50):
    constant_experiments.append(
        M.Experiment(
            experiment_name=f'Constant {i}',
            experiment_config=M.ExperimentConfig(
                iterations=1_000
            ),
            simulator_config=M.SimulatorConfig(
                type='TTLSimulator'
            ),
            strategy_config=M.StrategyConfig(
                type='ConstantStrategy',
                params=ConstantStrategy.Params(initial_ttl=i)
            )
        )
    )
