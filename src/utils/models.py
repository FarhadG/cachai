import datetime
from collections import OrderedDict
from typing import Literal, Optional
from pydantic import BaseModel, create_model

import src.utils.constants as C
from src.core.strategies.debugger_strategy import DebuggerStrategy
from src.core.strategies.aggregate_strategy import AggregrateStrategy
from src.core.strategies.increment_strategy import IncrementStrategy

LossSchema = create_model('LossSchema', **OrderedDict([
    (C.MSE, (float, ...)),
    (C.RMSE, (float, ...)),
    (C.MAE, (float, ...)),
    (C.MBE, (float, ...))
]))

MetricsSchema = create_model('MetricsSchema', **OrderedDict([
    (C.CACHE_SERVE_RATE, (float, ...)),
    (C.CACHE_HIT_PRECISION, (float, ...)),
    (C.CACHE_HIT_ACCURACY, (float, ...)),
    (C.CACHE_STALE_RATE, (float, ...)),
    (C.CACHE_MISS_RATE, (float, ...)),
    (C.TOTAL_REQUESTS, (int, ...)),
    (C.CACHE_HIT_TOTAL, (int, ...)),
    (C.CACHE_STALE_TOTAL, (int, ...)),
    (C.CACHE_MISS_TOTAL, (int, ...)),
    (C.CACHE_SERVE_TOTAL, (int, ...))
]))


ExperimentLogSchema = create_model('ExperimentLogSchema', **OrderedDict([
    (C.EXPERIMENT_NAME, (str, ...)),
    (C.ITERATION, (int, ...)),
    (C.OBSERVATION_TIME, (datetime.timedelta, ...)),
    (C.OBSERVATION_TYPE, (str, ...)),
    (C.KEY, (str, ...)),
    (C.HITS, (int, ...)),
    (C.Y_TRUE, (float, ...)),
    (C.Y_PRED, (float, ...)),
    (C.RMSE, (float, ...)),
    (C.MAE, (float, ...)),
    (C.MBE, (float, ...)),
]))

CachaiLogSchema = create_model('CachaiLogSchema', **OrderedDict([
    (C.OBSERVATION_TIME, (datetime.timedelta, ...)),
    (C.OBSERVATION_TYPE, (str, ...)),
    (C.KEY, (str, ...)),
]))

StrategyLogSchema = create_model('StrategyLogSchema', **OrderedDict([
    (C.OBSERVATION_TIME, (datetime.timedelta, ...)),
    (C.OBSERVATION_TYPE, (str, ...)),
    (C.KEY, (str, ...)),
    (C.HITS, (int, ...)),
    (C.Y_TRUE, (float, ...)),
    (C.Y_PRED, (float, ...)),
    (C.Y_FEEDBACK, (float, ...)),
]))


class TTLSimulatorConfig(BaseModel):
    debug: bool = False
    hit_rate: float = 0.01
    operations_count: int
    records_count: int
    record_mean_range: tuple[int, int]
    record_var_range: tuple[int, int]


class StrategyConfig(BaseModel):
    # TODO: avoid hard coded Literal and use Strategy classes?
    name: Literal[
        'DebuggerStrategy',
        'AggregrateStrategy',
        'IncrementStrategy'
    ]
    params: Optional[
        DebuggerStrategy.Params |
        AggregrateStrategy.Params |
        IncrementStrategy.Params
    ] = None,
    tune_params: bool = False


class CachaiConfig(BaseModel):
    strategy_config: StrategyConfig


class ExperimentConfig(BaseModel):
    experiment_name: str
    output_dir: Optional[str] = 'results'
    cachai_config: CachaiConfig
    simulator_config: TTLSimulatorConfig
