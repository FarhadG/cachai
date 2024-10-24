import cachai.utils.models as M

debugger_experiments = [
    M.Experiment(
        experiment_name='Debugger Test',
        experiment_config=M.ExperimentConfig(
            iterations=1_000,
            debug=True
        ),
        simulator_config=M.SimulatorConfig(
            type='TTLSimulator'
        ),
        strategy_config=M.StrategyConfig(
            type='DebuggerStrategy',
        )
    )
]
