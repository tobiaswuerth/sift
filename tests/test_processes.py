"""The worker processes sift starts: how many the machine gives, and the pools of them."""

import os

import pytest

from sift import processes

from . import workers


def test_cpus_respects_what_the_platform_allows(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 6, raising=False)  # Python 3.13+
    assert processes.cpus() == 6
    monkeypatch.delattr(os, "process_cpu_count")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)  # Linux
    assert processes.cpus() == 3
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: None)  # unknown
    assert processes.cpus() == 1


@pytest.mark.parametrize("windows, most", [(True, 61), (False, 200)])
def test_a_pool_has_no_more_processes_than_the_system_lets_it(monkeypatch, windows, most):
    # Windows waits for 64 handles at once, and Python's pool for 61 processes at most: with 63 CPU threads or more,
    # `sift index` ended in "max_workers must be <= 61" there, as did the keyword build and the upgrade of an old index.
    made = []
    monkeypatch.setattr(processes, "WINDOWS", windows)
    monkeypatch.setattr(processes, "ProcessPoolExecutor", lambda n, mp_context: made.append((n, mp_context)))
    processes.pool(200)
    processes.pool(2)
    # spawned, not forked, on every platform: forking a process with threads (PyTorch starts some) can deadlock
    assert [(n, context.get_start_method()) for n, context in made] == [(most, "spawn"), (2, "spawn")]


def test_a_pool_of_more_processes_than_any_system_has_works():
    with processes.running(200) as pool:
        assert pool.submit(os.getpid).result() != os.getpid()


def test_a_pool_left_by_an_error_runs_nothing_it_has_not_started(tmp_path):
    # A pool left by an error ran everything handed to it before the error showed: a keyword build that could not
    # write its first part read every other part first, for minutes or hours.
    started = []

    def left_by_an_error():
        with processes.running(1) as pool:
            started.extend(pool.submit(workers.marking, str(tmp_path), n) for n in range(50))
            started[0].result()
            raise RuntimeError("the first part could not be written")

    with pytest.raises(RuntimeError, match="the first part"):
        left_by_an_error()
    assert 1 <= len(list(tmp_path.iterdir())) < 10  # those running by then, and the few a pool holds ready
    assert started[-1].cancelled()
