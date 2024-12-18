from typing import Any, Callable, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass


def periodic(
    x,
    amplitude=1,
    periods_count=1,
    h_shift=0,
    v_shift=0
):
    a = amplitude/2
    h_shift_from_zero = np.pi/2+h_shift
    v_shift_from_zero = a+v_shift
    return a*np.sin((periods_count*x)-h_shift_from_zero)+v_shift_from_zero


def spike(
    x,
    amplitude=1,
    h_shift=0,
    v_shift=0,
    spread=1
):
    a = amplitude/2
    return a*np.exp(-((x-h_shift)/spread)**2)+v_shift


def noise(
    x,
    amplitude=1
):
    return amplitude*np.random.normal(0, 1, len(x))


def generate_traffic(
    start,
    end,
    freq='min',  # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    count=None,
    periodic_params={'amplitude': 200, 'periods_count': 2, 'v_shift': 100},
    spike_params={'amplitude': 2000, 'h_shift': np.pi*1.5, 'v_shift': 0, 'spread': 0.01},
    noise_params={'amplitude': 10},
):
    date_range = pd.date_range(start=start, end=end, freq=freq).strftime('%Y-%m-%d %H:%M:%S')
    x = np.linspace(0, 2*np.pi, len(date_range))
    traffic = sum([
        periodic(x, **periodic_params),
        spike(x, **spike_params),
        noise(x, **noise_params)
    ])
    traffic[traffic < 0] = 0
    # if count is provided, we normalize the traffic volume to that count
    if count != None:
        traffic = (traffic/sum(traffic)*count).astype(int)
        total_traffic_count = sum(traffic)
        # this ensures that after the normalization, the traffic volume is exactly the count
        if total_traffic_count != count:
            missing_count = np.abs(total_traffic_count-count)
            random_index = np.random.choice(len(traffic), missing_count)
            delta = 1 if total_traffic_count < count else -1
            for i in random_index:
                traffic[i] += delta
    return np.repeat(date_range, traffic.astype(int))


@dataclass
class Operation:
    type: str
    sample_record: Optional[Callable] = None


def sample_operations(
    operations=[],
    count=1,
    probs=None
):
    operations_len = len(operations)
    probs = probs if probs is not None else [1/operations_len]*operations_len
    assert sum(probs) == 1, 'Probabilities must sum to 1'
    indices = np.random.choice(operations_len, count, p=probs)
    return np.array([operations[i] for i in indices])
