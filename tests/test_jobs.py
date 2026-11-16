"""A server's work off its event loop (sift.jobs): one call at a time on a worker thread, a cancelled call stopped, the
models on the main thread. And the store's side of stopping: statements end while store.ABORT is set."""

import sqlite3
import threading
import time
from functools import partial

import anyio
import pytest

from sift import jobs, store

# Endless for a test: tens of seconds. Truly endless, a regression would hang the test run rather than fail it.
ENDLESS = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n LIMIT 300000000) SELECT count(*) FROM n"
LONG = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n LIMIT 300000) SELECT count(*) FROM n"


@pytest.fixture
def db(tmp_path):
    return store.database(tmp_path / "jobs.sqlite", create=True)


@pytest.fixture(autouse=True)
def no_abort():
    yield
    store.ABORT.clear()


async def until(event):
    """Wait for a threading.Event without blocking the loop."""
    with anyio.fail_after(5):
        while not event.is_set():
            await anyio.sleep(0.001)


def test_statements_stop_while_abort_is_set(db):
    store.ABORT.set()
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        db.execute(ENDLESS).fetchone()
    store.ABORT.clear()
    assert db.execute(LONG).fetchone() == (300000,)


def test_a_call_runs_on_a_worker_thread_while_the_loop_runs_on():
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait(5)
        return threading.current_thread()

    async def main():
        async with anyio.create_task_group() as tasks:
            threads = []
            tasks.start_soon(lambda: _append(threads, jobs.run(work)))
            await until(started)  # the loop goes on while the call runs
            release.set()
        return threads[0]

    assert anyio.run(main) is not threading.main_thread()


async def _append(into, awaitable):
    into.append(await awaitable)


def test_calls_run_one_at_a_time():
    inside, most, count = [0], [0], threading.Lock()

    def work():
        with count:
            inside[0] += 1
            most[0] = max(most[0], inside[0])
        time.sleep(0.05)
        with count:
            inside[0] -= 1

    async def main():
        async with anyio.create_task_group() as tasks:
            for _ in range(3):
                tasks.start_soon(jobs.run, work)

    anyio.run(main)
    assert most[0] == 1


def test_a_cancelled_call_stops_and_the_next_one_runs(db):
    began, errors = threading.Event(), []

    def endless():
        began.set()
        try:
            db.execute(ENDLESS).fetchone()
        except sqlite3.OperationalError as e:
            errors.append(e)

    async def main():
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(jobs.run, endless)
            await until(began)
            tasks.cancel_scope.cancel()  # the client cancelled, or went away
        with anyio.fail_after(5):  # the next call waits for the cancelled one: it must stop
            return await jobs.run(lambda: db.execute(LONG).fetchone())

    assert anyio.run(main) == (300000,)  # ABORT no longer set: the next call's statements run
    assert "interrupted" in str(errors[0])


def test_a_call_cancelled_before_its_turn_never_runs():
    release, ran = threading.Event(), []

    async def main():
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(jobs.run, lambda: release.wait(5))  # holds the worker
            await anyio.sleep(0.05)
            with anyio.CancelScope() as waiting:
                tasks.start_soon(_cancel_soon, waiting)
                await jobs.run(lambda: ran.append(True))
            release.set()

    anyio.run(main)
    assert ran == []


async def _cancel_soon(scope):
    await anyio.sleep(0.05)
    scope.cancel()


def test_writes_run_to_their_end_when_cancelled(db):
    db.execute("CREATE TABLE t(x)")
    began, done = threading.Event(), threading.Event()

    def write():
        began.set()
        time.sleep(0.1)  # the cancellation comes in meanwhile
        db.execute(f"INSERT INTO t {LONG.replace('count(*)', 'i')}")
        db.commit()
        done.set()

    async def main():
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(partial(jobs.run, write, abortable=False))
            await until(began)
            tasks.cancel_scope.cancel()
        with anyio.fail_after(5):
            await jobs.run(lambda: None)  # one call at a time: this one waits for the write

    anyio.run(main)
    assert done.is_set()
    assert db.execute("SELECT count(*) FROM t").fetchone() == (300000,)


class Models:
    name, main_thread = "fake", True

    def __init__(self):
        self.threads = []

    def encode(self, texts):
        self.threads.append(threading.current_thread())
        return len(texts)


class Worker:
    """Models on another machine, which answer once something on this one happens."""

    name, main_thread = "worker", False

    def __init__(self):
        self.answered = threading.Event()

    def encode(self, texts):
        return self.answered.wait(5)


def test_models_called_on_any_thread_leave_the_loop_free():
    # A request to a worker held the event loop on the main thread: up to 10 minutes of a server answering nothing,
    # not even a cancellation, when the worker's machine went to sleep.
    models, answers = jobs.OnMainThread(Worker()), []

    async def ask():
        answers.append(await jobs.run(lambda: models.encode(["a"])))

    async def main():
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask)
            await anyio.sleep(0.05)
            models._models.answered.set()  # only while the loop is free

    anyio.run(main)
    assert answers == [True]  # not timed out


def test_a_call_cancelled_while_a_worker_is_asked_holds_up_no_other():
    # The request went on, and with it the server's one call at a time: up to ten minutes before the next call, even
    # one that needs no models, though the client had cancelled.
    worker = Worker()  # answers only once told to: not within this test
    models = jobs.OnMainThread(worker)

    async def main():
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(partial(jobs.run, lambda: models.encode(["a"])))
            await anyio.sleep(0.05)
            tasks.cancel_scope.cancel()
        with anyio.fail_after(1):
            return await jobs.run(lambda: "next")

    try:
        assert anyio.run(main) == "next"
    finally:
        worker.answered.set()  # the request left behind ends


def test_requests_of_calls_cancelled_before_they_were_sent_never_are():
    # Each waited in a thread of its own for the one before, then went to the worker anyway: the next call waited for
    # all of them, and the worker's GPU computed answers no one read.
    sent, answer = [], threading.Event()

    class Slow:
        name, main_thread = "worker", False

        def encode(self, texts):
            sent.append(texts[0])
            answer.wait(5)
            return texts[0]

    models = jobs.OnMainThread(Slow())

    async def main():
        for text in "abc":
            with anyio.move_on_after(0.05):
                await jobs.run(lambda text=text: models.encode([text]))
        answer.set()
        with anyio.fail_after(2):
            return await jobs.run(lambda: models.encode(["d"]))

    assert anyio.run(main) == "d"
    assert sent == ["a", "d"]  # a was on its way; b and c never left


def test_a_request_of_a_call_cancelled_before_it_looked_is_not_made():
    # Cancelled while the request before was on its way, the call looks for it only now and then: not in time to drop
    # its own request, which the worker would have computed for no one.
    made = []
    store.ABORT.set()
    with pytest.raises(jobs.Cancelled):
        jobs._Requests().submit(lambda: made.append("a")).result(timeout=5)
    assert made == []


def test_a_slow_workers_answer_is_awaited(monkeypatch):
    # Many looks at whether the call was cancelled, none of them ending the wait.
    monkeypatch.setattr(jobs, "ABORT_CHECK", 0.001)

    class Slow:
        name, main_thread = "worker", False

        def encode(self, texts):
            time.sleep(0.05)
            return texts

    models = jobs.OnMainThread(Slow())
    assert anyio.run(jobs.run, lambda: models.encode(["a"])) == ["a"]


def test_a_workers_answer_and_error_come_back_from_its_thread():
    class Answering:
        name, main_thread = "worker", False

        def encode(self, texts):
            if not texts:
                raise ConnectionError("embedding remote https://pc:8766 is unreachable")
            return threading.current_thread()

    models = jobs.OnMainThread(Answering())
    assert anyio.run(jobs.run, lambda: models.encode(["a"])) is not threading.main_thread()
    with pytest.raises(ConnectionError, match="unreachable"):
        anyio.run(jobs.run, lambda: models.encode([]))


def test_models_called_on_any_thread_still_stop_when_cancelled():
    models = jobs.OnMainThread(Worker())
    store.ABORT.set()
    with pytest.raises(jobs.Cancelled):
        models.encode(["a"])


def test_the_models_run_on_the_main_thread_whoever_calls():
    models = jobs.OnMainThread(Models())
    assert models.name == "fake"
    assert anyio.run(jobs.run, lambda: models.encode(["a", "b"])) == 2
    assert models.encode(["c"]) == 1  # outside a server too
    assert models._models.threads == [threading.main_thread()] * 2


def test_a_cancelled_call_runs_the_models_no_more():
    models = jobs.OnMainThread(Models())
    store.ABORT.set()
    with pytest.raises(jobs.Cancelled):
        models.encode(["a"])
    assert models._models.threads == []


def test_models_are_wrapped_once():
    models = jobs.OnMainThread(Models())
    assert jobs.OnMainThread(models)._models is models._models
