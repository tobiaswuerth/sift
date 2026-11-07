"""A server's work for its clients, off its event loop: one call at a time on a worker thread, so that the loop stays
free to answer, to accept connections, and to hear that a client cancelled a call or went away. A cancelled call
stops: its SQLite statements end at once (store.ABORT), and it calls the models no more. Writes run to their end.

The models run on the main thread, which holds the loop, whoever calls them (OnMainThread): PyTorch crashes in worker
threads on Python 3.15 (Windows, macOS)."""

import functools
import logging
import threading

import anyio
import anyio.from_thread
import anyio.to_thread

from . import store

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """A call to the models after the call it belongs to was cancelled."""


class _Call:
    def __init__(self):
        self.cancelled = False


_turn = threading.Lock()  # held by the call that runs: one at a time
_state = threading.Lock()  # guards _running, so that a cancellation never stops the call after the cancelled one
_running = None


async def run(fn, *args, abortable=True, **kwargs):
    """Run fn(*args, **kwargs) on a worker thread once the calls before it are done, and return what it returns. If
    the awaiting task is cancelled, it returns at once; the call itself stops (SQLite statements interrupted, models
    refused) unless not `abortable`, and one that hasn't started yet never does."""
    call = _Call()

    def work():
        global _running
        with _turn:
            with _state:
                if call.cancelled:
                    return None
                _running = call
                store.ABORT.clear()
            try:
                return fn(*args, **kwargs)
            finally:
                with _state:
                    _running = None
                    store.ABORT.clear()

    try:
        return await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        with _state:
            call.cancelled = abortable or _running is not call  # a write that hasn't started yet needn't start
            if abortable and _running is call:
                store.ABORT.set()
                log.info("a call was cancelled: its work stops")
        raise


class OnMainThread:
    """Models whose methods run on the main thread, wherever they are called: from a worker thread of `run`, through
    the event loop. Called for a cancelled call, they raise Cancelled."""

    def __init__(self, models):
        self._models = models._models if isinstance(models, OnMainThread) else models

    def __getattr__(self, name):
        value = getattr(self._models, name)
        if not callable(value):
            return value

        @functools.wraps(value)
        def on_main_thread(*args, **kwargs):
            if store.ABORT.is_set():
                raise Cancelled("the call was cancelled")
            if threading.current_thread() is threading.main_thread():
                return value(*args, **kwargs)
            return anyio.from_thread.run_sync(functools.partial(value, *args, **kwargs))

        return on_main_thread
