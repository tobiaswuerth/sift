import threading

import pytest

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.events import Event, Noted
from sift.interfaces.server.gate import WAITING_FOR_ALONE, WAITING_FOR_READS, Gate
from tests.waits import crossed, joined

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
        assert gate.held_alone
    assert said == [Noted(WAITING_FOR_READS)]
