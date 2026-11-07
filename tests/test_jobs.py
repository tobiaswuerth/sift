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
    name = "fake"

    def __init__(self):
        self.threads = []

    def encode(self, texts):
        self.threads.append(threading.current_thread())
        return len(texts)


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
