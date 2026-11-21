"""A server's work for its clients, off its event loop: one call at a time on a worker thread, so that the loop stays
free to answer, to accept connections, and to hear that a client cancelled a call or went away. A cancelled call
stops: its SQLite statements end at once (store.ABORT), and it calls the models no more. Writes run to their end.

The models of this machine run on the main thread, which holds the loop, whoever calls them (OnMainThread): PyTorch
crashes in worker threads on Python 3.15 (Windows, macOS). A worker's are asked in a thread of their own: the loop
stays free while a request waits for its answer, and a cancelled call stops waiting."""

import functools
import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future, wait
from typing import TYPE_CHECKING, Any, TypeVar

import anyio
import anyio.from_thread
import anyio.to_thread

from . import store

if TYPE_CHECKING:
    from .models import Models

T = TypeVar("T")

log = logging.getLogger(__name__)
ABORT_CHECK = 0.05  # seconds between looks at store.ABORT while a worker's answer is awaited


class Cancelled(Exception):
    """A call to the models after the call it belongs to was cancelled."""


class _Call:
    def __init__(self) -> None:
        self.cancelled = False


_turn = threading.Lock()  # held by the call that runs: one at a time
_state = threading.Lock()  # guards _running, so that a cancellation never stops the call after the cancelled one
_running: _Call | None = None


async def run(fn: Callable[..., T], *args: Any, abortable: bool = True, **kwargs: Any) -> T:
    """Run fn(*args, **kwargs) on a worker thread once the calls before it are done, and return what it returns. If
    the awaiting task is cancelled, it returns at once; the call itself stops (SQLite statements interrupted, models
    refused) unless not `abortable`, and one that hasn't started yet never does."""
    call = _Call()

    def work() -> T:
        global _running
        with _turn:
            with _state:
                if call.cancelled:  # its caller has gone, and what this raises reaches no one
                    raise Cancelled("the call was cancelled")
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


class OnMainThread(Models if TYPE_CHECKING else object):  # every method of the models, through __getattr__
    """Models whose methods run on the main thread, wherever they are called, if they must (`main_thread`): from a
    worker thread of `run`, through the event loop, which waits meanwhile. A worker's run in a thread of their own,
    waited for until the call is cancelled (see _abortable). Called for a cancelled call, they raise Cancelled."""

    def __init__(self, models: "Models") -> None:
        self._models: Models = models._models if isinstance(models, OnMainThread) else models

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._models, name)
        if not callable(value):
            return value

        @functools.wraps(value)
        def on_main_thread(*args: Any, **kwargs: Any) -> Any:
            if store.ABORT.is_set():
                raise Cancelled("the call was cancelled")
            call = functools.partial(value, *args, **kwargs)
            if not self._models.main_thread:
                return _abortable(call)
            if threading.current_thread() is threading.main_thread():
                return call()
            return anyio.from_thread.run_sync(call)

        return on_main_thread


class _Requests:
    """A worker's requests, made one after another in a thread of their own: one cancelled before its turn is never
    made. A daemon, so that a request left waiting holds up no exit."""

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[tuple[Future[Any], Callable[[], Any]]] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(self, call: Callable[[], T]) -> Future[T]:
        """call(), in its turn: a Future of its answer."""
        answer: Future[T] = Future()
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._serve, name="sift-worker-requests", daemon=True)
                self._thread.start()
        self._queue.put((answer, call))
        return answer

    def _serve(self) -> None:
        while True:
            answer, call = self._queue.get()
            if not answer.set_running_or_notify_cancel():  # cancelled before its turn
                continue
            if store.ABORT.is_set():  # its call was cancelled, and hasn't looked yet: the next call clears it first
                answer.set_exception(Cancelled("the call was cancelled"))
                continue
            try:
                answer.set_result(call())
            except BaseException as error:  # handed to the caller, if it still waits
                answer.set_exception(error)


_requests = _Requests()


def _abortable(call: Callable[[], T]) -> T:
    """call() as a worker's request (see _Requests), its answer awaited until the call it belongs to is cancelled:
    then Cancelled at once, and the request dropped if not made yet, its answer if it was. A worker's request can wait
    minutes for one (remote.TIMEOUT), and meanwhile the server takes no other call."""
    answer = _requests.submit(call)
    while not wait([answer], ABORT_CHECK).done:
        if store.ABORT.is_set():
            answer.cancel()
            raise Cancelled("the call was cancelled; a request to the worker on its way ends by itself")
    return answer.result()
