"""Wall-clock timing decorator with JIT-aware GPU synchronization.

Provides a ``@timer(name)`` decorator that records per-function wall-clock
time into a global ``timers`` dictionary.  The first invocation of each
decorated function is excluded from the statistics because it typically
includes JAX's JIT compilation overhead.
"""

from collections.abc import Callable
from functools import wraps
from time import perf_counter_ns
from typing import ParamSpec, TypeVar

import jax

from .parameters import params

P = ParamSpec("P")
R = TypeVar("R")

timers: dict[str, dict[str, float | int]] = {}

ns_to_s: float = 10 ** (-9)


def timer(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that records wall-clock time for *func* under *name*.

    When ``params.debug.time_functions`` is ``True``, every call is timed
    after a ``jax.block_until_ready`` barrier (necessary for accurate GPU
    timing).  The very first call is recorded with zero time and zero hits
    to exclude JIT compilation cost.

    Parameters
    ----------
    name:
        Key under which the timing statistics are stored in ``timers``.
    """

    def decorator_timer(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper_timer(*args: P.args, **kwargs: P.kwargs) -> R:
            if params.debug.time_functions:
                start = perf_counter_ns()
                value = func(*args, **kwargs)
                jax.block_until_ready(value)
                stop = perf_counter_ns()
                dt = (stop - start) * ns_to_s
                if name in timers:
                    timers[name]["avg-t"] = (
                        timers[name]["avg-t"] * timers[name]["hits"] + dt
                    ) / (timers[name]["hits"] + 1)
                    timers[name]["hits"] += 1
                    timers[name]["tot-t"] = (
                        timers[name]["avg-t"] * timers[name]["hits"]
                    )
                else:
                    timers[name] = {}
                    # Ignore the first hit,
                    # likely to be subject to JIT compilation
                    timers[name]["avg-t"] = 0
                    timers[name]["hits"] = 0
            else:
                value = func(*args, **kwargs)
            return value

        return wrapper_timer

    return decorator_timer
