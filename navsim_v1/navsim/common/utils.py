import functools
from collections import deque
import time

def mean_time_every_5_calls(name: str = None):
    """
    Decorator that measures a function's wall time and logs the mean of the last 5 calls.
    Prints once every 5 calls.
    """
    def decorator(func):
        times = deque(maxlen=5)
        call_count = 0
        label = name or getattr(func, "__qualname__", getattr(func, "__name__", "function"))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal call_count
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                times.append(duration)
                call_count += 1
                if call_count % 5 == 0 and len(times) == 5:
                    mean_duration = sum(times) / 5.0
                    print(f"[perf] {label}: mean of last 5 calls = {mean_duration:.6f}s")
        return wrapper
    return decorator
