from pathlib import Path

from sift.api.collections import AddCollection
from sift.api.indexing import IndexCollections
from sift.app.collections import add
from sift.app.indexing import index
from sift.storage.catalog import CATALOG, set_active
from sift.storage.watching import HomeWatch
from tests.files import write
from tests.homes import indexed, quiet


def home_of(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    for name in ("a", "b"):
        write(tmp_path / name / "alice.md", f"# Alice\n\nAlice keeps notes in {name}.\n")
        add(home, AddCollection(str(tmp_path / name)))
    indexed(home)
    return home


def test_what_is_seen_stays_until_another_connection_commits_to_a_store_or_the_catalog(tmp_path: Path) -> None:
    home = home_of(tmp_path)
    watch = HomeWatch(home)
    first = watch.seen()
    assert (watch.seen(), len(first)) == (first, 3)
    write(tmp_path / "b" / "bob.md", "# Bob\n\nBob too.\n")
    index(home, IndexCollections(("b",)), quiet)
    after_index = watch.seen()
    assert (after_index[:2], after_index[2] != first[2]) == (first[:2], True)
    set_active(home, ["a"], active=False)
    after_deactivate = watch.seen()
    assert (len(after_deactivate), after_deactivate[0] != first[0]) == (2, True)
    watch.let_go()


def test_a_home_without_a_catalog_is_seen_alike_until_one_is_made(tmp_path: Path) -> None:
    home = tmp_path / "home"
    watch = HomeWatch(home)
    assert watch.seen() == watch.seen() == (None,)
    write(tmp_path / "a" / "alice.md", "# Alice\n\nAlice keeps notes.\n")
    add(home, AddCollection(str(tmp_path / "a")))
    assert watch.seen()[0] is not None
    watch.let_go()


def test_a_store_that_cannot_be_read_is_never_seen_alike_and_one_gone_is_seen_gone(tmp_path: Path) -> None:
    home = home_of(tmp_path)
    (tmp_path / "home" / CATALOG).with_name("collections").joinpath("a.sqlite").write_bytes(b"no database")
    watch = HomeWatch(home)
    assert watch.seen() != watch.seen()
    watch.let_go()
    store = (tmp_path / "home" / CATALOG).with_name("collections").joinpath("b.sqlite")
    store.unlink()
    assert watch.seen()[2] is None
    store.mkdir()
    assert watch.seen()[2] != watch.seen()[2]
    watch.let_go()
    assert watch.held == {}
