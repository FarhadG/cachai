import os
from typing import List
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, wait

import cachai.utils.models as M
from cachai.utils.timer import timer


def run_parallel(func, experiments, max_workers=1):
    with ProcessPoolExecutor(max_workers) as executor:
        futures = [executor.submit(func, experiment) for experiment in experiments]
    wait(futures)


def run_sequential(func, experiments):
    for experiment in experiments:
        func(experiment)


class Runner():
    def __init__(
        self,
        parallel=False,
        max_workers=cpu_count() - 1,
        disable_pydevd=True
    ):
        self._max_workers = max_workers
        self._parallel = parallel
        if disable_pydevd:
            os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

    @timer
    def run(self, func, experiments):
        # TODO: test different configurations for CPU/IO-bound tasks
        if self._parallel:
            run_parallel(func, experiments, self._max_workers)
        else:
            run_sequential(func, experiments)
