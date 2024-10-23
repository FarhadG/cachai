import numpy as np
import pandas as pd
from sklearn.calibration import check_consistent_length
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import cachai.utils.constants as C

# OTHERS:
# Byte Hit Rate (BHR): The ratio of the number of bytes served from the cache to the total number of bytes requested.
# - BHR = (Total Bytes - Bytes Served) / Total Bytes
# Request Rate (RR): The rate at which requests are being made to the cache.
# - RR = TR / Time
# Latency Metrics:
# - Average Latency: The average time taken to serve a request from the cache.
# - 95th Percentile Latency: The time within which 95% of the requests are served.
# Cache Utilization (CU): The percentage of cache storage being used.
# - CU = (Used Cache Size / Total Cache Size) * 100
# Cache Hit Latency: The time taken to serve a request that results in a cache hit. Use: To ensure that cache hits are served efficiently.
# Miss Rate Decay: Definition: The rate at which the miss rate decreases over time as the cache gets populated. Use: To understand how quickly the cache becomes effective.


def mean_bias_error(y_true, y_pred, fn, *, weights=None):
    check_consistent_length(y_true, y_pred, weights)
    differences = y_pred - y_true
    return np.average(fn(differences), weights=weights)


def mean_squared_bias_error(y_true, y_pred, *, weights=None):
    return mean_bias_error(y_true, y_pred, np.square, weights=weights)


def mean_absolute_bias_error(y_true, y_pred, *, weights=None):
    return mean_bias_error(y_true, y_pred, np.abs, weights=weights)


def evaluate(df):
    """
    - True Positives (TP) as valid cache hits (True Hits).
    - False Positives (FP) as stale cache hits (False Hits).
    - False Negatives (FN) as cache misses (Misses).
    - True Negatives (TN) as valid cache misses (True Misses).
    - Total requests (TR) as the total number of requests.

    - Hit Rate = (TP + FP) / TR — How many of the requests were served from cache
    - Miss Rate = (FN + TN) / TR — How many of the requests were not served from cache
    - Precision = TP / (TP + FP) — How accurate were the cache hits
    - Accuracy = (TP + TN) / TR — How many of the requests were served correctly
    - Recall = TP / (TP + FN) — How many of the cache hits were found
    - FP Rate = FP / (FP + TP) — How many of the cache hits were stale
    - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """

    metrics = []
    y_true = df[C.Y_TRUE]
    y_pred = df[C.Y_PRED]
    hits = df[C.HITS]
    hits_total = hits.sum()
    hits_mean = hits.mean()
    total_requests = len(df)
    true_positive = len(df[df[C.OBSERVATION_TYPE] == C.HIT])
    true_negative = len(df[df[C.OBSERVATION_TYPE] == C.VALID_TTL])
    false_positive = len(df[df[C.OBSERVATION_TYPE] == C.STALE])
    false_negative = len(df[df[C.OBSERVATION_TYPE] == C.MISS])

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mabe = mean_absolute_bias_error(y_true, y_pred)
    msbe = mean_squared_bias_error(y_true, y_pred)
    cache_hits = true_positive + false_positive
    not_cache_hits = true_negative + false_negative
    hit_rate = cache_hits / total_requests
    miss_rate = not_cache_hits / total_requests
    accuracy = (true_positive + true_negative) / total_requests
    precision = (true_positive / cache_hits) if cache_hits else 0
    recall = (true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else 0
    false_positive_rate = (false_positive / cache_hits) if cache_hits else 0
    f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) else 0

    metrics.append([
        rmse, mae, mabe, msbe,
        hit_rate, miss_rate, precision, accuracy, recall,
        false_positive_rate, f1_score, hits_total, hits_mean, len(df)
    ])
    return pd.DataFrame(metrics, columns=[
        C.RMSE, C.MAE, C.MABE, C.MSBE,
        C.HIT_RATE, C.MISS_RATE, C.PRECISION, C.ACCURACY, C.RECALL,
        C.FALSE_POSITIVE_RATE, C.F1_SCORE, C.HITS_TOTAL, C.HITS_MEAN, C.LEN
    ])


def evaluate_group(df, groupby=[C.EXPERIMENT_NAME, C.MODEL_NAME]):
    return df.groupby(groupby).apply(evaluate, include_groups=False)
