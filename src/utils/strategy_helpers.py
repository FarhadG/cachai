import src.utils.constants as C
import src.utils.models as M
from src.core.strategies.debugger_strategy import DebuggerStrategy, RegressionDebuggerStrategy
from src.core.strategies.aggregate_strategy import AggregrateStrategy
from src.core.strategies.increment_strategy import IncrementStrategy


def strategy_from_config(config: M.StrategyConfig):
    strategy_name = config.name
    if strategy_name == C.DEBUGGER_STRATEGY:
        return DebuggerStrategy
    elif strategy_name == C.REGRESSION_DEBUGGER_STRATEGY:
        return RegressionDebuggerStrategy
    elif strategy_name == C.AGGREGATE_STRATEGY:
        return AggregrateStrategy
    elif strategy_name == C.INCREMENT_STRATEGY:
        return IncrementStrategy
    else:
        raise ValueError(f'Unknown strategy type: {strategy_name}')
