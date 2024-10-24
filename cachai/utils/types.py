from pydantic import BaseModel
from typing import Literal, Type
from dataclasses import dataclass
from enum import Enum

import cachai.utils.constants as C
from cachai.strategies.heuristics_strategy import ConstantStrategy

"""
general types
"""


class ObservationType(Enum):
    HIT = C.HIT
    MISS = C.MISS
    STALE = C.STALE
    VALID_TTL = C.VALID_TTL


"""
experiment config definitions
"""


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
