from time import time
from functools import wraps


def _print_timer(name, start, end):
    print(f"{name} took {end - start:.2f} seconds to run")


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        func_name = func.__name__
        class_name = args[0].__class__.__name__
        name = f"{class_name}.{func_name}" if class_name else func_name
        _print_timer(name, start, end)
        return result
    return wrapper


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        end = time()
        _print_timer(self.name, self.start, end)
