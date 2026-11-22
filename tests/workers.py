"""Readers for the worker processes of `sift index` in tests, which end or hang the process on a file so named, as a
reader's native code can. In a module of their own, without the fixtures' imports: a spawned process imports it."""

import os
import time

from sift import indexing

_loaded = indexing._loaded  # sift's own, before a test puts one of these in its place


def ending(path, fields):
    """As indexing._loaded reads a file; one named poison ends the process."""
    if "poison" in os.path.basename(path):
        os._exit(70)
    return _loaded(path, fields)


def hanging(path, fields):
    """As indexing._loaded reads a file; one named hangs comes to no end, for two minutes."""
    if "hangs" in os.path.basename(path):
        time.sleep(120)
    return _loaded(path, fields)


RELEASE = "release"  # the file in a task's folder at which the tasks held there go on


def marking(folder, n, held_until=None):
    """Leave a file named `n` in `folder`: a task that tells it ran. With `held_until`, a time, every task but the
    first then holds its process until a file named RELEASE is there, or that time has come."""
    open(os.path.join(folder, str(n)), "w").close()
    while n and held_until and time.time() < held_until and not os.path.exists(os.path.join(folder, RELEASE)):
        time.sleep(0.01)
    return n
