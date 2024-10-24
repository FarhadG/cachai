from typing import Literal
from pydantic import BaseModel

from cachai.strategies.heuristics_strategy import ConstantStrategy


class ExperimentConfig(BaseModel):
    iterations: int
    debug: bool = False


class SimulatorConfig(BaseModel):
    type: Literal['TTLSimulator']


class StrategyConfig(BaseModel):
    type: Literal[
        'DebuggerStrategy',
        'ConstantStrategy',
    ]
    params: None | ConstantStrategy.Params = None


class Experiment(BaseModel):
    experiment_name: str
    experiment_config: ExperimentConfig
    simulator_config: SimulatorConfig
    strategy_config: StrategyConfig
