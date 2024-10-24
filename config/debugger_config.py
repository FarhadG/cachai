import cachai.utils.types as T

debugger_experiments = [
    T.Experiment(
        experiment_name='Debugger Test',
        experiment_config=T.ExperimentConfig(
            iterations=1_000,
            debug=True
        ),
        simulator_config=T.SimulatorConfig(
            type='TTLSimulator'
        ),
        strategy_config=T.StrategyConfig(
            type='DebuggerStrategy',
        )
    )
]
