import itertools
import random
import tempfile
from pathlib import Path
from typing import NamedTuple

from hypothesis import given
from hypothesis import strategies as st

from sift.domain.plan.segments import Policy
from sift.storage.database import Batches, Connection, Count, reading, rows, writing
from sift.storage.keyword_beside import built, left
from sift.storage.keyword_lists import segments
from sift.storage.keyword_meta import beside, drop_beside
from sift.storage.keywords import remade, remake_started, settle, totals
from tests.examples import few
from tests.keyword_indexes import MERGING, NOTES, TWO, checked, generation_of, rounds, settled, staled
from tests.keyword_orders import current_of, due, embedded, placed, placed_in
from tests.keyword_stores import SMALL, Round, applied, opened_store

SIZES = {
    "served": "SELECT chunks, average FROM keyword_served ORDER BY first",
    "built": "SELECT chunks, average FROM keyword_built ORDER BY first",
}


class Built(NamedTuple):
    chunks: int
    average: float


def begun(connection: Connection) -> bool:
    with reading(connection) as transaction:
        return beside(transaction) is not None


def places_waiting(connection: Connection) -> int:
    with reading(connection) as transaction:
        (found,) = rows(transaction, Count, "SELECT count(*) FROM keyword_places")
    return found.n


def test_an_order_of_a_current_index_is_built_beside_it_which_serves_as_before_until_the_whole_is_built() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        before = placed(connection)
        ids = [each.id for each in before]
        top = max(each.place for each in before)
        assert placed_in(connection, [*ids[::-1][:-2], 10**15]) == len(ids)
        assert left(connection) == len(ids)
        seen, as_served = [0], generation_of(connection)
        for done in built(connection, TWO):
            assert done > seen[-1]
            seen.append(done)
            served = (current_of(connection), placed(connection) == before, begun(connection))
            assert served == (True, done < len(ids), done < len(ids))
            with reading(connection) as transaction:
                assert all((each.last <= top) == (done < len(ids)) for each in segments(transaction))
            assert (generation_of(connection) == as_served) == (done < len(ids))
            checked(connection)
        assert seen[-1] == len(ids)
        assert (current_of(connection), begun(connection), left(connection)) == (True, False, 0)
        held = placed(connection)
        assert [each.id for each in held] == [*ids[::-1][:-2], *ids[:2]]
        assert [each.place for each in held] == list(range(top + 1, top + 1 + len(ids)))
        assert (places_waiting(connection), due(connection)) == (0, False)
        checked(connection)
        applied(connection, [{"x.md": ("", "X", "four")}])
        assert placed(connection)[-1].place == top + len(ids) + 1
        checked(connection)


def test_what_is_built_beside_is_scored_for_the_average_length_of_the_whole_and_counts_nothing_twice() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        with reading(connection) as transaction:
            whole = totals(transaction)
        placed_in(connection, [each.id for each in placed(connection)][::-1])
        next(built(connection, TWO))
        with reading(connection) as transaction:
            assert totals(transaction) == whole
            made = rows(transaction, Built, SIZES["built"])
        assert made == [Built(2, whole.tokens / whole.chunks)]


def test_a_change_written_while_an_order_is_built_beside_drops_what_was_built_and_the_order_is_due_again() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        before = placed(connection)
        placed_in(connection, [each.id for each in before][::-1])
        next(built(connection, TWO))
        assert not due(connection)
        as_served = generation_of(connection)
        with writing(connection) as transaction:
            drop_beside(transaction)
        assert (generation_of(connection), begun(connection)) == (as_served, False)
        applied(connection, [{"n1.md": None, "x.md": ("", "X", "four")}])
        assert (begun(connection), places_waiting(connection), left(connection)) == (False, 0, 0)
        assert list(built(connection, TWO)) == []
        kept = {each.id: each.place for each in before}
        assert all(kept.get(each.id, each.place) == each.place for each in placed(connection))
        assert due(connection)
        checked(connection)


def test_an_order_begun_is_taken_up_where_it_stopped_and_no_other_is_begun_meanwhile() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        assert placed_in(connection, ids[::-1]) == len(ids)
        assert next(built(connection, TWO)) == 2
        assert placed_in(connection, ids) == 0
        embedded(connection, ids)
        assert not due(connection)
        assert left(connection) == len(ids) - 2
        assert list(built(connection, TWO))[-1] == len(ids) - 2
        assert [each.id for each in placed(connection)] == ids[::-1]
        checked(connection)
        assert due(connection)


def test_an_index_of_another_version_due_an_order_is_placed_anew_where_it_is_and_remade_once() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        top = max(each.place for each in placed(connection))
        staled(connection)
        assert placed_in(connection, ids[::-1]) == len(ids)
        assert (current_of(connection), begun(connection), left(connection)) == (False, False, 0)
        assert [each.place for each in placed(connection)] == list(range(top + 1, top + 1 + len(ids)))
        assert remake_started(connection) == len(ids)
        assert list(remade(connection, TWO))[-1] == len(ids)
        assert [each.id for each in placed(connection)] == ids[::-1]
        assert not due(connection)
        checked(connection)


def test_an_index_that_became_one_of_another_version_drops_what_was_built_beside_it_when_it_is_remade() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        before = placed(connection)
        placed_in(connection, [each.id for each in before][::-1])
        next(built(connection, TWO))
        staled(connection)
        assert (left(connection), list(built(connection, TWO))) == (0, [])
        assert begun(connection)
        assert remake_started(connection) == len(before)
        assert (begun(connection), places_waiting(connection)) == (False, 0)
        list(remade(connection))
        assert placed(connection) == before
        assert due(connection)
        checked(connection)


def test_segments_are_merged_within_the_index_served_and_within_the_one_built_never_across_the_two() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        rounds: list[Round] = [{f"n{n}.md": ("", f"Changed {n}", "four " * n)} for n in range(0, 7, 2)]
        applied(connection, [NOTES, *rounds], MERGING)
        before = placed(connection)
        ids = [each.id for each in before]
        assert placed_in(connection, ids[::-1]) == len(ids)
        for done in built(connection, MERGING):
            list(settle(connection, MERGING))
            assert (placed(connection) == before) == (done < len(ids))
            checked(connection)
        assert [each.id for each in placed(connection)] == ids[::-1]
        list(settle(connection, MERGING))
        assert sum(each.chunks for each in settled(connection, MERGING.merging)) == len(ids)
        checked(connection)


def sizes(connection: Connection, held: str) -> list[int]:
    with reading(connection) as transaction:
        return [each.chunks for each in rows(transaction, Built, SIZES[held])]


def test_a_merge_of_the_index_served_leaves_the_segments_built_beside_it_alone() -> None:
    four = Batches(chunks_a_segment=4, merging=Policy(factor=2, floor=1, span=2.0))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        placed_in(connection, ids[::-1])
        assert next(built(connection, four)) == 4
        assert (sizes(connection, "served"), sizes(connection, "built")) == ([len(ids)], [4])
        assert list(settle(connection, four)) == []
        assert (sizes(connection, "served"), sizes(connection, "built")) == ([len(ids)], [4])
        list(built(connection, four))
        assert [each.id for each in placed(connection)] == ids[::-1]
        checked(connection)


def test_the_segments_built_beside_are_merged_as_they_are_built() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        with reading(connection) as transaction:
            whole = totals(transaction)
        placed_in(connection, [each.id for each in placed(connection)][::-1])
        steps = built(connection, MERGING)
        assert (next(steps), sizes(connection, "built")) == (2, [2])
        assert (next(steps), sizes(connection, "built")) == (4, [4])
        with reading(connection) as transaction:
            assert rows(transaction, Built, SIZES["built"]) == [Built(4, whole.tokens / whole.chunks)]
        checked(connection)


@few(40)
@given(rounds, rounds, st.randoms(use_true_random=False), st.integers(0, 4))
def test_an_index_holds_its_live_chunks_at_every_step_of_an_order_whatever_is_written_meanwhile(
    first: list[Round], later: list[Round], shuffled: random.Random, steps: int
) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, first, MERGING)
        ids = [each.id for each in placed(connection)]
        shuffled.shuffle(ids)
        placed_in(connection, ids)
        for _ in itertools.islice(built(connection, MERGING), steps):
            checked(connection)
        applied(connection, later, MERGING)
        checked(connection)
        list(built(connection, MERGING))
        checked(connection)
        assert not begun(connection)
        assert places_waiting(connection) == 0
