"""Optional per-stage wall-clock timing of a query (enabled by the environment variable USGLOC_TIMING=1).

``start()`` opens a query; ``tick(name)`` adds the time since the previous tick to ``TIMES[name]`` after a CUDA
synchronize, so asynchronous kernels are charged to the stage that launched them. Disabled (the default), both are
no-ops: no synchronize, no change to the pipeline's results or speed.
"""
import os
import time

import torch

ON = os.environ.get("USGLOC_TIMING") == "1"
TIMES: dict = {}
_last = [None]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def start():
    if ON:
        _sync()
        TIMES.clear()
        _last[0] = time.perf_counter()


def tick(name):
    if ON and _last[0] is not None:
        _sync()
        now = time.perf_counter()
        TIMES[name] = TIMES.get(name, 0.0) + now - _last[0]
        _last[0] = now
