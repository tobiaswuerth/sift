import dataclasses
import itertools
from pathlib import Path

from sift.api.collections import AddCollection
from sift.api.search import SearchKeywords
from sift.app.collections import add
from sift.app.search import search
from sift.app.searching import Watching
from sift.retrieval.keyword.search import Tuning
from tests.files import write
from tests.homes import indexed


def test_a_search_out_of_time_looks_no_further_for_more_works(tmp_path: Path) -> None:
    for n in range(4):
        write(tmp_path / "c" / f"copy{n}.md", "---\ndoi: 10.1000/halo\n---\n# Copy\n\nhalo halo halo\n")
    for n in range(3):
        write(tmp_path / "c" / f"other{n}.md", f"---\ndoi: 10.1000/other{n}\n---\n# Other {n}\n\nhalo\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "c")))
    indexed(home)
    request = SearchKeywords("halo", k=3, compact=True)
    assert len(search(home, request).hits) == 3
    times = itertools.chain([0.0], itertools.repeat(10.0))
    cut = search(home, dataclasses.replace(request, budget=1.0), watching=Watching(clock=lambda: next(times)))
    assert (cut.partial, [hit.id for hit in cut.hits]) == (True, ["c/copy0.md"])


def many_notes(tmp_path: Path) -> Path:
    for n in range(300):
        write(tmp_path / "many" / f"n{n:03}.md", f"# Note {n}\n\nA note, number {n}.\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "many")))
    indexed(home)
    return home


def test_a_search_out_of_time_says_it_is_partial_and_keeps_what_it_found(tmp_path: Path) -> None:
    home = many_notes(tmp_path)
    ticks = itertools.count()
    request = SearchKeywords("note", k=500, budget=0.5)
    cut = search(home, request, watching=Watching(clock=lambda: float(next(ticks))), tuning=Tuning(first_batch=1))
    assert cut.partial
    assert 0 < len(cut.hits) < 300
    whole = search(home, dataclasses.replace(request, budget=None), tuning=Tuning(first_batch=1))
    assert (whole.partial, len(whole.hits)) == (False, 300)


def test_with_time_to_spare_a_budget_changes_nothing(tmp_path: Path) -> None:
    home = many_notes(tmp_path)
    request = SearchKeywords("note", k=40)
    spared = search(
        home,
        dataclasses.replace(request, budget=60.0),
        watching=Watching(clock=lambda: 0.0),
        tuning=Tuning(first_batch=1),
    )
    assert spared == search(home, request, tuning=Tuning(first_batch=1))
    assert not spared.partial


def test_a_search_out_of_time_while_its_prefix_is_read_is_partial(tmp_path: Path) -> None:
    for n in range(50):
        write(tmp_path / "w" / f"n{n}.md", f"# Note\n\nw{n}\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "w")))
    indexed(home)
    request = SearchKeywords("w*", k=50, compact=True)
    assert len(search(home, request).hits) == 50
    times = itertools.chain([0.0], itertools.repeat(10.0))
    cut = search(home, dataclasses.replace(request, budget=1.0), watching=Watching(clock=lambda: next(times)))
    assert (cut.partial, cut.hits) == (True, ())
    assert len(search(home, request).hits) == 50
