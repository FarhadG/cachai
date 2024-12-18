import numpy as np


def uniform(
    data,
    count=1
):
    n = len(data)
    domain = np.arange(n)
    indices = np.random.choice(domain, count)
    return data[indices]


def normal(
    data,
    count=1,
    mean=0.5,
    std=0.1
):
    n = len(data)
    raw_indices = np.random.normal(n*mean, n*std, count)
    indices = np.remainder(raw_indices, n).astype(int)
    indices = np.where(indices < 0, indices + n, indices)
    return data[indices]


def zipf(
    data,
    count=1,
    alpha=1.5,
    return_probs=False
):
    n = len(data)
    ranks = np.arange(1, n+1)
    if alpha >= 0:
        probs = ranks**-alpha
    else:
        probs = ranks[::-1]**alpha
    probs /= probs.sum()
    indices = np.random.choice(ranks-1, count, p=probs)
    samples = data[indices]
    return (samples, probs) if return_probs else samples


def periodic_zipf(
    data,
    progress=1.0,
    periods_count=1,
    count=1,
    alpha=1.5,
    return_probs=False
):
    alpha_progress = round(alpha*np.cos(progress*periods_count*np.pi), 2)
    return zipf(data, alpha=alpha_progress, count=count, return_probs=return_probs)
