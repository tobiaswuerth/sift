from typing import SupportsIndex, final, override

from hypothesis import given
from hypothesis import strategies as st

from sift.domain.plan.indexing import (
    MOST_TEXT_BYTES,
    MOST_TIME,
    RACY,
    Found,
    IndexPlan,
    Stat,
    batches,
    index_plan,
    recorded,
    stored_time,
    within,
)

A, B = Stat(1, 10), Stat(2, 20)


def test_new_and_changed_files_are_read_and_unchanged_ones_counted() -> None:
    plan = index_plan(Found({"new.md": A, "changed.md": B, "same.md": A}), {"changed.md": A, "same.md": A})
    assert plan == IndexPlan(
        read=("changed.md", "new.md"),
        too_large=(),
        drop=(),
        kept=(),
        left_as_indexed=(),
        emptied=(),
        forget=(),
        unchanged=1,
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
    assert (plan.drop, plan.left_as_indexed, plan.emptied) == ((), ("a.md", "b.md"), ("",))
    assert index_plan(Found({}), {}).emptied == ()


def test_a_folder_inside_that_is_there_but_empty_keeps_its_documents_as_a_drive_not_mounted_looks_so() -> None:
    known = {"archive/2020/a.md": A, "archive/b.md": A, "archived.md": A, "gone/c.md": A, "rl.md": A}
    plan = index_plan(Found({"rl.md": A}, empty_folders=("archive/", "new/")), known)
    assert (plan.left_as_indexed, plan.emptied) == (("archive/2020/a.md", "archive/b.md"), ("archive/",))
    assert plan.drop == ("archived.md", "gone/c.md")


def test_a_file_that_could_not_be_read_is_read_again_though_it_looks_unchanged() -> None:
    marks = {"locked.md": True, "huge.md": False}
    files = {"locked.md": A, "huge.md": A, "same.md": A}
    plan = index_plan(Found(files), files, unreadable=marks)
    assert (plan.read, plan.unchanged, plan.forget) == (("locked.md",), 2, ())


def test_the_mark_of_an_unreadable_file_is_forgotten_once_the_file_is_gone_for_good() -> None:
    marked = ("gone.md", "new-gone.md", "kept/a.md", "unstated.md", "there.md", "empty/x.md")
    marks: dict[str, bool] = dict.fromkeys(marked, True)
    known = {"gone.md": A, "kept/a.md": A, "there.md": A, "empty/x.md": A}
    found = Found(
        {"there.md": A}, unlisted_folders=("kept/",), unlisted_files=("unstated.md",), empty_folders=("empty/",)
    )
    plan = index_plan(found, known, unreadable=marks)
    assert plan.forget == ("gone.md", "new-gone.md")
    assert plan.drop == ("gone.md",)


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


folders_there = st.lists(st.sampled_from(["", "a/", "b/", "a"]), max_size=2).map(tuple)
founds = st.builds(
    Found,
    listings,
    folders_there,
    st.lists(paths, max_size=2).map(tuple),
    st.just(tuple[str, ...]()),
    st.lists(st.sampled_from(["a/", "b/", "a"]), max_size=2).map(tuple),
)


@given(founds, listings, st.dictionaries(paths, st.booleans()))
def test_a_plan_reads_drops_and_keeps_what_it_should(
    found: Found, known: dict[str, Stat], marks: dict[str, bool]
) -> None:
    seen, unlisted, unstated = found.files, found.unlisted_folders, found.unlisted_files
    plan = index_plan(found, known, unreadable=marks)
    unseen = known.keys() - seen.keys()
    drop, kept, left = set(plan.drop), set(plan.kept), set(plan.left_as_indexed)
    assert len(plan.read) + plan.unchanged == len(seen)
    assert set(plan.read) == {path for path in seen if known.get(path) != seen[path] or marks.get(path)}
    assert list(plan.read) == sorted(plan.read)
    assert drop | kept | left == unseen
    assert len(drop) + len(kept) + len(left) == len(unseen)
    assert not [path for path in drop if path in unstated or any(path.startswith(folder) for folder in unlisted)]
    assert not [path for path in drop for folder in plan.emptied if path.startswith(folder)]
    assert bool(left) == bool(plan.emptied)
    assert not set(plan.forget) & (seen.keys() | kept | left | set(unstated))
    assert drop & marks.keys() <= set(plan.forget) <= marks.keys()


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


def test_a_file_changed_within_the_clock_s_grain_of_the_scan_is_recorded_to_be_read_again() -> None:
    scanned = 100 * 10**9
    fresh, settled = Stat(scanned - 10**9, 5), Stat(scanned - 3 * 10**9, 7)
    assert recorded({"fresh.md": fresh, "settled.md": settled}, scanned) == {
        "fresh.md": Stat(RACY, 5),
        "settled.md": settled,
    }
    assert index_plan(Found({"fresh.md": fresh}), {"fresh.md": Stat(RACY, 5)}).read == ("fresh.md",)


def _in_any_before(paths: frozenset[str], folders: tuple[str, ...]) -> frozenset[str]:
    return frozenset(path for path in paths for folder in folders if folder == "" or path.startswith(folder))


@given(
    st.lists(st.lists(st.sampled_from("abc"), min_size=1, max_size=4).map("/".join), max_size=30),
    st.lists(
        st.lists(st.sampled_from("abc"), max_size=3).map(lambda parts: "".join(f"{part}/" for part in parts)),
        max_size=8,
    ),
)
def test_the_files_within_folders_are_those_found_by_their_prefix(paths: list[str], folders: list[str]) -> None:
    files = frozenset(f"{path}.md" for path in paths)
    assert within(files, tuple(folders)) == _in_any_before(files, tuple(folders))


def test_a_plan_of_many_empty_folders_and_many_gone_files_costs_their_depth() -> None:
    looked: list[int] = []

    @final
    class Counted(str):
        __slots__ = ()

        @override
        def find(self, sub: str, start: SupportsIndex | None = None, end: SupportsIndex | None = None, /) -> int:
            looked.append(1)
            return super().find(sub, start, end)

        @override
        def startswith(
            self, prefix: str | tuple[str, ...], start: SupportsIndex | None = None, end: SupportsIndex | None = None, /
        ) -> bool:
            looked.append(1)
            return super().startswith(prefix, start, end)

    empty = tuple(f"f{n}/" for n in range(200))
    known: dict[str, Stat] = {Counted(f"g{n}/x{n}.md"): A for n in range(2_000)}
    plan = index_plan(Found({"a.md": A}, empty_folders=empty), known)
    assert (len(plan.drop), plan.emptied) == (2_000, ())
    walks, folders_a_path = 3, 2
    assert 0 < len(looked) <= walks * folders_a_path * len(known)
