"""Local no-op Trackio shim for AReaL smoke runs.

AReaL imports trackio at module import time even when Trackio logging is
disabled. The current node5 vLLM stack needs huggingface-hub 0.x, while recent
trackio imports require hub 1.x symbols. Keep this shim in the experiment
working directory so PacMan smoke tests can use disabled logging without
changing the global package set again.
"""


def init(*args, **kwargs):
    return None


def log(*args, **kwargs):
    return None


def finish(*args, **kwargs):
    return None
