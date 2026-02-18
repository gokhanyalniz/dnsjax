from functools import wraps
from time import perf_counter_ns

import jax

from parameters import params

timers = {}

ns_to_s = 10 ** (-9)


def timer(name):
    def decorator_timer(func):
        @wraps(func)
        def wrapper_timer(*args, **kwargs):
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
