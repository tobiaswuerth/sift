import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.plan.changes import Changes, FileOutcome, FileSeen, changes, state_read, state_unread

keys = st.sampled_from("abcdef")
digests = st.sampled_from("xy")
helds = st.dictionaries(keys, digests, max_size=6)
reads = st.lists(st.tuples(keys, digests), max_size=6, unique_by=lambda each: each[0])


def test_what_was_read_is_new_changed_or_unchanged_and_what_is_not_is_gone() -> None:
    held = {"a": "1", "b": "2", "c": "3"}
    assert changes(held, [("d", "4"), ("b", "2"), ("a", "9")]) == Changes(new=("d",), changed=("a",), gone=("c",))
    assert changes({}, []) == Changes((), (), ())


@given(helds, reads)
def test_changes_part_the_keys_read_and_held_each_into_one_place_in_order(
    held: dict[str, str], read: list[tuple[str, str]]
) -> None:
    found = changes(held, read)
    read_keys = [key for key, _ in read]
    unchanged = [key for key, digest in read if held.get(key) == digest]
    assert sorted([*found.new, *found.changed, *unchanged]) == sorted(read_keys)
    assert set(found.gone) == held.keys() - set(read_keys)
    assert set(found.new) == set(read_keys) - held.keys()
    assert list(found.new) == [key for key in read_keys if key in found.new]
    assert list(found.changed) == [key for key in read_keys if key in found.changed]
    assert list(found.gone) == [key for key in held if key in found.gone]


def seen(
    *, refusal: str | None = None, held: bool = True, listed: bool = True, gone: bool = False, because: str = ""
) -> FileSeen:
    return FileSeen(refusal, held=held, listed=listed, gone=gone, left_out_because=because)


@pytest.mark.parametrize(
    "found, state",
    [
        (seen(refusal="Access is denied"), FileOutcome("unreadable", "Access is denied", system=True)),
        (seen(held=False, listed=False, gone=True), FileOutcome("missing")),
        (seen(held=False, listed=False, because="a reason"), FileOutcome("left out", "a reason")),
        (seen(), None),
        (seen(held=False), None),
        (seen(listed=False, gone=True), None),
    ],
)
def test_a_file_asked_for_is_said_before_it_is_read_when_there_is_nothing_to_read(
    found: FileSeen, state: FileOutcome | None
) -> None:
    assert state_unread(found) == state


@pytest.mark.parametrize(
    "found, as_it_is, changed, state",
    [
        (seen(listed=False, gone=True), True, True, FileOutcome("removed")),
        (seen(listed=False, because="a reason"), True, True, FileOutcome("removed", "a reason")),
        (seen(), False, True, FileOutcome("unreadable", "a problem")),
        (seen(held=False), True, True, FileOutcome("added", "a problem")),
        (seen(), True, True, FileOutcome("changed", "a problem")),
        (seen(), True, False, FileOutcome("unchanged", "a problem")),
    ],
)
def test_a_file_read_is_said_added_changed_unchanged_unreadable_or_removed(
    found: FileSeen, *, as_it_is: bool, changed: bool, state: FileOutcome
) -> None:
    assert state_read(found, held_as_it_is=as_it_is, problem="a problem", changed=changed) == state
