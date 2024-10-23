from enum import Enum, auto

# observation types
HIT = 'hit'
MISS = 'miss'
STALE = 'stale'
VALID_TTL = 'valid_ttl'

# data columns
EXPERIMENT_NAME = 'experiment_name'
MODEL_NAME = 'model_name'
ITERATION = 'iteration'
OBSERVATION_TYPE = 'observation_type'
OBSERVATION_TIME = 'observation_time'
Y_TRUE = 'y_true'
Y_PRED = 'y_pred'
HITS = 'hits'

# metrics
RMSE = 'rmse'
MAE = 'mae'
MABE = 'mabe'
MSBE = 'msbe'
HIT_RATE = 'hit_rate'
MISS_RATE = 'miss_rate'
PRECISION = 'precision'
ACCURACY = 'accuracy'
RECALL = 'recall'
FALSE_POSITIVE_RATE = 'false_positive_rate'
F1_SCORE = 'f1_score'
HITS_TOTAL = 'hits_total'
HITS_MEAN = 'hits_mean'
LEN = 'len'
