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


def marking(folder, n):
    """Leave a file named `n` in `folder`: a task that tells it ran."""
    open(os.path.join(folder, str(n)), "w").close()
    return n
