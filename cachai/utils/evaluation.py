from collections import OrderedDict
import numpy as np
import pandas as pd
from pydantic import create_model
from sklearn.calibration import check_consistent_length
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import cachai.utils.constants as C
import cachai.utils.models as M


def evaluate_cache_metrics(df):
    """
    - True Positives (TP) as valid cache hits (True Hits).
    - False Positives (FP) as stale cache hits (False Hits).
    - False Negatives (FN) as cache misses (Misses).
    - True Negatives (TN) as valid cache misses (True Misses).
    - Total requests (TR) as the total number of requests.

    - Cache Served Rate = (TP + FP) / TR — How many of the requests were served from cache
    - Cache Not Served Rate = (FN + TN) / TR — How many of the requests were not served from cache
    - Precision = TP / (TP + FP) — How accurate were the cache hits
    - Accuracy = (TP + TN) / TR — How many of the requests were served correctly
    - Recall = TP / (TP + FN) — How many of the cache hits were found
    - False Positive Rate = FP / (FP + TP) — How many of the cache hits were stale
    - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """

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
    cache_stale_total = (df[C.OBSERVATION_TYPE] == C.STALE).sum()
    cache_miss_total = (df[C.OBSERVATION_TYPE] == C.MISS).sum()
    cache_valid_ttl_total = (df[C.OBSERVATION_TYPE] == C.VALID_TTL).sum()
    cache_hit_total = (df[C.OBSERVATION_TYPE] == C.HIT).sum()

    total_requests = cache_hit_total + cache_stale_total + cache_miss_total + cache_valid_ttl_total
    cache_serve_total = cache_hit_total + cache_stale_total
    cache_serve_rate = cache_serve_total / total_requests
    cache_hit_precision = (cache_hit_total + cache_valid_ttl_total) / cache_serve_total if cache_serve_total else 0
    cache_hit_accuracy = (cache_hit_total + cache_valid_ttl_total) / total_requests
    cache_stale_rate = cache_stale_total / cache_serve_total if cache_serve_total else 0
    cache_miss_rate = cache_miss_total / total_requests

    return M.MetricsSchema(
        cache_serve_rate=cache_serve_rate,
        cache_hit_precision=cache_hit_precision,
        cache_hit_accuracy=cache_hit_accuracy,
        cache_stale_rate=cache_stale_rate,
        cache_miss_rate=cache_miss_rate,
        total_requests=total_requests,
        cache_hit_total=cache_hit_total,
        cache_stale_total=cache_stale_total,
        cache_miss_total=cache_miss_total,
        cache_serve_total=cache_serve_total
    ).model_dump()


def mean_bias_error(y_true, y_pred, weights=None):
    check_consistent_length(y_true, y_pred, weights)
    differences = y_pred - y_true
    return np.average(np.sign(differences), weights=weights)


def evaluate_loss(y_true, y_pred, round_to=3):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mbe = mean_bias_error(y_true, y_pred)
    return M.LossSchema(
        rmse=round(rmse, round_to),
        mae=round(mae, round_to),
        mbe=round(mbe, round_to)
    ).model_dump()


def evaluate_experiment(df):
    output = []
    y_true = df[C.Y_TRUE].to_numpy().astype(float)
    y_pred = df[C.Y_PRED].to_numpy().astype(float)
    loss = evaluate_loss(y_true, y_pred)
    output.append(loss)
    return pd.DataFrame(output, columns=M.LossSchema.__fields__.keys())


def evaluate_experiment_group(df, groupby=[C.EXPERIMENT_NAME]):
    return df \
        .groupby(groupby) \
        .apply(evaluate_experiment) \
        .sort_values(by=C.EXPERIMENT_NAME) \
        .reset_index(level=0)


def evaluate_advisor(df):
    output = []
    cache_metrics = evaluate_cache_metrics(df)
    output.append(cache_metrics)
    return pd.DataFrame(output, columns=M.MetricsSchema.__fields__.keys())


def evaluate_advisor_group(df, groupby=[C.EXPERIMENT_NAME]):
    return df \
        .groupby(groupby) \
        .apply(evaluate_advisor) \
        .sort_values(by=C.EXPERIMENT_NAME) \
        .reset_index(level=0)