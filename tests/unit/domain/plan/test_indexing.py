from hypothesis import given
from hypothesis import strategies as st

from sift.domain.plan.indexing import (
    MOST_TEXT_BYTES,
    MOST_TIME,
    Found,
    IndexPlan,
    Stat,
    batches,
    index_plan,
    stored_time,
)

A, B = Stat(1, 10), Stat(2, 20)


def test_new_and_changed_files_are_read_and_unchanged_ones_counted() -> None:
    plan = index_plan(Found({"new.md": A, "changed.md": B, "same.md": A}), {"changed.md": A, "same.md": A})
    assert plan == IndexPlan(
        read=("changed.md", "new.md"), too_large=(), drop=(), kept=(), left_as_indexed=(), unchanged=1
    )


def test_a_file_touched_but_of_the_same_time_and_size_is_not_read() -> None:
    assert index_plan(Found({"a.md": A}), {"a.md": A}).read == ()


def test_files_gone_are_dropped() -> None:
    assert index_plan(Found({"a.md": A}), {"a.md": A, "gone.md": B}).drop == ("gone.md",)


def test_files_in_a_folder_that_could_not_be_listed_are_kept() -> None:
    known = {"robots/brush.md": A, "robots/deep/x.md": A, "robotsx.md": A, "rl.md": A}
    plan = index_plan(Found({"rl.md": A}, unlisted_folders=("robots/",)), known)
    assert (plan.kept, plan.drop) == (("robots/brush.md", "robots/deep/x.md"), ("robotsx.md",))


def test_a_root_that_could_not_be_listed_keeps_everything() -> None:
    plan = index_plan(Found({}, unlisted_folders=("",)), {"a.md": A, "b/c.md": B})
    assert (plan.kept, plan.drop, plan.left_as_indexed) == (("a.md", "b/c.md"), (), ())


def test_a_file_that_could_not_be_stated_is_kept() -> None:
    plan = index_plan(Found({"b.md": A}, unlisted_files=("a.md", "new.md")), {"a.md": A, "b.md": A})
    assert (plan.kept, plan.drop, plan.read) == (("a.md",), (), ())


def test_a_folder_left_without_any_file_keeps_its_documents() -> None:
    plan = index_plan(Found({}), {"a.md": A, "b.md": B})
    assert (plan.drop, plan.left_as_indexed) == ((), ("a.md", "b.md"))


def test_a_folder_of_links_alone_left_out_is_no_folder_emptied() -> None:
    plan = index_plan(Found({}, outside=("link.md",)), {"link.md": A})
    assert (plan.drop, plan.left_as_indexed) == (("link.md",), ())


def test_a_file_larger_than_a_document_may_be_is_left_out_unread() -> None:
    huge, full = Stat(1, MOST_TEXT_BYTES + 1), Stat(1, MOST_TEXT_BYTES)
    plan = index_plan(Found({"huge.txt": huge, "full.txt": full}), {})
    assert (plan.read, plan.too_large) == (("full.txt",), ("huge.txt",))
    assert index_plan(Found({"huge.txt": huge}), {"huge.txt": huge}).too_large == ()


def test_a_time_past_what_the_store_counts_to_is_held_at_its_end() -> None:
    assert [stored_time(t) for t in (-(2**64), -5, 0, 2**70)] == [-MOST_TIME, -5, 0, MOST_TIME]


paths = st.text(alphabet="ab/.", min_size=1, max_size=6)
stats = st.builds(Stat, st.integers(0, 3), st.integers(0, 3))
listings = st.dictionaries(paths, stats, max_size=8)


@given(
    listings,
    listings,
    st.lists(st.sampled_from(["", "a/", "b/", "a"]), max_size=2),
    st.lists(paths, max_size=2),
)
def test_a_plan_reads_drops_and_keeps_what_it_should(
    seen: dict[str, Stat], known: dict[str, Stat], folders: list[str], unstated: list[str]
) -> None:
    plan = index_plan(Found(seen, tuple(folders), tuple(unstated)), known)
    unseen = known.keys() - seen.keys()
    assert set(plan.read) | {path for path in seen if known.get(path) == seen[path]} == seen.keys()
    assert len(plan.read) + plan.unchanged == len(seen)
    assert set(plan.drop) | set(plan.kept) | set(plan.left_as_indexed) == unseen
    assert not set(plan.drop) & set(plan.kept)
    assert not plan.left_as_indexed or not seen
    assert list(plan.read) == sorted(plan.read)


def test_files_are_read_in_batches_of_so_many_files_and_bytes() -> None:
    stats = {"a": Stat(0, 5), "b": Stat(0, 5), "c": Stat(0, 5), "big": Stat(0, 50), "d": Stat(0, 1)}
    assert batches(["a", "b", "c", "big", "d"], stats, 2, 12) == (("a", "b"), ("c",), ("big",), ("d",))
    assert batches([], stats, 2, 12) == ()


@given(st.lists(st.integers(0, 20), max_size=30), st.integers(1, 5), st.integers(1, 30))
def test_a_batch_holds_at_most_so_many_files_and_bytes_unless_one_file_is_more(
    sizes: list[int], most_files: int, most_bytes: int
) -> None:
    stats = {str(n): Stat(0, size) for n, size in enumerate(sizes)}
    found = batches(list(stats), stats, most_files, most_bytes)
    assert [path for batch in found for path in batch] == list(stats)
    for batch in found:
        assert 0 < len(batch) <= most_files
        assert len(batch) == 1 or sum(stats[path].size for path in batch) <= most_bytes
