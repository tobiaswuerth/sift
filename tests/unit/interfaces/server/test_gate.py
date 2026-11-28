import threading

import pytest

from sift.api.errors import JobStoppedError, Problem, StoreUnavailableError
from sift.api.events import Event, Noted
from sift.interfaces.server.gate import WAITING_FOR_ALONE, WAITING_FOR_READS, Gate
from tests.waits import crossed, joined, until

ROUNDS = 300


def test_no_read_runs_while_a_command_holds_the_home_alone_and_none_waits_for_good() -> None:
    gate, guard, said = Gate(), threading.Lock(), list[Event]()
    inside, overlaps, done = [0, 0], list[tuple[int, int]](), list[str]()
    barrier = threading.Barrier(10)

    def entered(at: int) -> None:
        with guard:
            inside[at] += 1
            if inside[1] > 1 or (inside[0] and inside[1]):
                overlaps.append((inside[0], inside[1]))

    def left(at: int) -> None:
        with guard:
            inside[at] -= 1

    def reading() -> None:
        crossed(barrier, 30)
        for _ in range(ROUNDS):
            with gate.shared(said.append):
                entered(0)
                left(0)
        done.append("read")

    def writing() -> None:
        crossed(barrier, 30)
        for _ in range(ROUNDS // 10):
            with gate.alone(said.append):
                entered(1)
                left(1)
        done.append("written")

    threads = [threading.Thread(target=reading) for _ in range(8)] + [
        threading.Thread(target=writing) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        joined(thread, 60)
    assert (overlaps, sorted(done)) == ([], ["read"] * 8 + ["written"] * 2)
    assert {event.text for event in said if isinstance(event, Noted)} <= {WAITING_FOR_ALONE, WAITING_FOR_READS}
    assert (gate.readers, gate.held_alone) == (0, False)


def test_a_command_that_waits_too_long_for_reads_is_refused_and_lets_them_go_on() -> None:
    gate, said = Gate(), list[Event]()
    with gate.shared(said.append):
        with (
            pytest.raises(
                StoreUnavailableError, match=r"searches still read this home's stores after 0\.05 s"
            ) as refused,
            gate.alone(said.append, patience=0.05),
        ):
            pass
        with gate.shared(said.append):
            pass
    assert refused.value.code is Problem.BUSY
    assert said == [Noted(WAITING_FOR_READS)]
    with gate.alone(said.append):
        assert gate.held_alone is True
    assert said == [Noted(WAITING_FOR_READS)]


def test_a_command_stopped_as_it_waits_for_reads_leaves_the_home_to_reads_and_commands() -> None:
    gate, said = Gate(), list[Event]()

    def stopped(_event: Event) -> None:
        raise JobStoppedError("the job was stopped", "run it again")

    with gate.shared(said.append):
        with pytest.raises(JobStoppedError), gate.alone(stopped):
            pass
        assert (gate.readers, gate.held_alone) == (1, False)
        with gate.shared(said.append):
            pass
    with gate.alone(said.append):
        assert gate.held_alone is True
    assert (gate.readers, gate.held_alone, said) == (0, False, [])


def test_a_read_that_comes_while_a_command_holds_the_home_alone_is_told_so_and_runs_after_it() -> None:
    gate, said, read = Gate(), list[Event](), threading.Event()

    def reading() -> None:
        with gate.shared(said.append):
            read.set()

    thread = threading.Thread(target=reading)
    with gate.alone(said.append):
        thread.start()
        assert until(lambda: said == [Noted(WAITING_FOR_ALONE)])
        assert not read.is_set()
    joined(thread, 30)
    assert (read.is_set(), gate.readers, gate.held_alone) == (True, 0, False)


def test_a_command_stopped_while_it_waits_for_reads_stops_waiting_and_leaves_the_home_free() -> None:
    gate, said = Gate(), list[Event]()

    def stopped() -> None:
        raise JobStoppedError("the job was stopped", "run it again")

    waiting = gate.alone(said.append, patience=30, alive=stopped)
    with gate.shared(said.append), pytest.raises(JobStoppedError), waiting:
        raise AssertionError
    with gate.alone(said.append):
        assert gate.held_alone is True
    assert said == [Noted(WAITING_FOR_READS)]


def test_a_command_looks_more_than_once_for_the_reads_to_end_before_it_is_refused() -> None:
    gate, looked, said = Gate(), list[int](), list[Event]()
    with gate.shared(said.append):
        waiting = gate.alone(said.append, patience=0.5, alive=lambda: looked.append(1))
        with pytest.raises(StoreUnavailableError) as refused, waiting:
            raise AssertionError
    assert (refused.value.code, looked, said) == (Problem.BUSY, [1, 1], [Noted(WAITING_FOR_READS)])
