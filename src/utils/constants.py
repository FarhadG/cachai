from enum import Enum
from typing import Literal

# general
CACHAI = 'Cachai'
KEY = 'key'

# observation types
HIT = 'hit'
MISS = 'miss'
STALE = 'stale'
VALID_TTL = 'valid_ttl'
WRITE = 'write'
READ_WRITE = 'read_write'


# data columns
EXPERIMENT_NAME = 'experiment_name'
STRATEGY = 'strategy'
STRATEGY_NAME = 'strategy_name'
ITERATION = 'iteration'
OBSERVATION_TYPE = 'observation_type'
OBSERVATION_TIME = 'observation_time'
X = 'X'
Y_TRUE = 'y_true'
Y_PRED = 'y_pred'
Y_FEEDBACK = 'y_feedback'
HITS = 'hits'
DEBUG = 'debug'
TIMESTAMP = 'timestamp'
OPERATION = 'operation'
RECORD = 'record'
PAYLOAD = 'payload'
STORED_KEY_DATA = 'stored_key_data'
# logger names
CACHAI_LOGGER = 'cachai_logger'
EXPERIMENT_LOGGER = 'experiment_logger'
STRATEGY_LOGGER = 'strategy_logger'
# loss metrics
RMSE = 'rmse'
MAE = 'mae'
MBE = 'mbe'

# cache metrics
TOTAL_REQUESTS = 'total_requests'
CACHE_SERVE_RATE = 'cache_serve_rate'
CACHE_HIT_PRECISION = 'cache_hit_precision'
CACHE_HIT_ACCURACY = 'cache_hit_accuracy'
CACHE_STALE_RATE = 'cache_stale_rate'
CACHE_MISS_RATE = 'cache_miss_rate'
CACHE_HIT_TOTAL = 'cache_hit_total'
CACHE_STALE_TOTAL = 'cache_stale_total'
CACHE_MISS_TOTAL = 'cache_miss_total'
CACHE_SERVE_TOTAL = 'cache_serve_total'

# strategy names
DEBUGGER_STRATEGY = 'DebuggerStrategy'
AGGREGATE_STRATEGY = 'AggregrateStrategy'
INCREMENT_STRATEGY = 'IncrementStrategy'

# supported operations
CREATE = 'CREATE'
READ = 'READ'
UPDATE = 'UPDATE'
DELETE = 'DELETE'
DATA_CHANGED = 'DATA_CHANGED'
