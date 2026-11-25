import contextlib
import dataclasses
import sqlite3
import stat
from pathlib import Path

import pytest

from sift.api.collections import AddCollection, Counted, ShowStatus
from sift.api.errors import InvalidRequestError, NotFoundError, Problem
from sift.api.events import Event, Phase, PhaseDone, PhaseStarted
from sift.api.ids import CollectionId
from sift.api.indexing import IndexCollections, Skipped
from sift.api.notices import Notice
from sift.api.search import Hit, SearchKeywords, SearchResult
from sift.app.collections import add, status
from sift.app.documents import documents
from sift.app.indexing import index
from sift.app.search import search
from sift.app.searching import EMPTY_QUERY
from sift.storage.keywords import REMAKE_REMEDY, TAKEN_OVER_REMEDY
from sift.storage.layout import LAYOUT
from sift.storage.store import MISSING_REMEDY, create_store
from sift.storage.store_files import delete_store
from sift.storage.unusable import DAMAGED_REMEDY
from tests.damage import changed, damage_table
from tests.files import write
from tests.homes import asked, indexed

LONG = "\n\n".join(["Pruning skips what cannot win. " * 20] * 8)


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "pruning.md", f"# Block-Max Pruning\n\nRanges are scored best bound first.\n\n## More\n\n{LONG}\n")
    write(root / "halo.md", "---\ntags: [quantum, cosmology]\n---\n# Halo\n\nDark matter halo models.\n")
    write(root / "matter.md", "# Matter\n\nDark matter, and matter again.\n")
    for n in range(8):
        write(root / f"filler{n}.md", f"# Filler {n}\n\nNothing to see here, note {n}.\n")
    return root


@pytest.fixture
def home(tmp_path: Path, notes: Path) -> Path:
    home = tmp_path / "home"
    add(home, AddCollection(str(notes)))
    indexed(home)
    return home


def ids(home: Path, request: SearchKeywords) -> list[str]:
    return [hit.id for hit in search(home, request).hits]


def test_a_store_that_may_only_be_read_is_searched_and_left_as_it_was(home: Path) -> None:
    store = home / "collections" / "notes.sqlite"
    before = store.stat().st_mtime_ns
    store.chmod(stat.S_IREAD)
    try:
        assert ids(home, SearchKeywords("halo")) == ["notes/halo.md"]
    finally:
        store.chmod(stat.S_IREAD | stat.S_IWRITE)
    assert store.stat().st_mtime_ns == before


def test_a_search_finds_each_document_by_its_best_passage_with_where_it_is(home: Path) -> None:
    result = search(home, SearchKeywords("halo"))
    assert result == SearchResult(
        (
            Hit(
                "notes/halo.md",
                "notes",
                "halo.md",
                "Halo",
                4,
                6,
                3.959956,
                "# Halo\n\nDark matter halo models.",
            ),
        ),
        1,
        (),
        scored_by="keywords",
    )


def test_the_lines_of_a_hit_are_its_passage(home: Path) -> None:
    (hit,) = search(home, SearchKeywords('"block max"')).hits
    (shown,) = documents(home, asked(hit.id, lines=(hit.first_line, hit.last_line))).found
    assert hit.text is not None
    assert shown.text.strip() == hit.text.strip()


def test_documents_are_ranked_by_bm25_ties_by_their_order(home: Path) -> None:
    assert ids(home, SearchKeywords("matter")) == ["notes/matter.md", "notes/halo.md"]
    assert ids(home, SearchKeywords("matter", k=1)) == ["notes/matter.md"]


def test_every_passage_of_a_document_is_found_when_asked(home: Path) -> None:
    assert ids(home, SearchKeywords("pruning")) == ["notes/pruning.md"]
    assert len(ids(home, SearchKeywords("pruning", all_passages=True))) > 1


def test_compact_hits_leave_their_text_out(home: Path) -> None:
    (hit,) = search(home, SearchKeywords("halo", compact=True)).hits
    assert hit.text is None
    assert hit.lines == "4-6"


def test_front_matter_values_are_searched_but_the_title_field_counts_as_the_title(home: Path, notes: Path) -> None:
    assert ids(home, SearchKeywords("quantum")) == ["notes/halo.md"]
    write(notes / "halo.md", "---\ntags: [classical]\n---\n# Halo\n\nDark matter halo models.\n")
    indexed(home)
    assert ids(home, SearchKeywords("quantum")) == []
    assert ids(home, SearchKeywords("classical")) == ["notes/halo.md"]


def test_words_no_passage_holds_together_find_passages_with_any_of_them(home: Path) -> None:
    assert sorted(ids(home, SearchKeywords("cosmology pruning"))) == ["notes/halo.md", "notes/pruning.md"]
    assert ids(home, SearchKeywords("cosmology AND pruning")) == []
    assert ids(home, SearchKeywords("zzz")) == []


@pytest.mark.parametrize("query", ["\N{EN DASH}", "-- ,"])
def test_a_query_of_no_word_finds_nothing(home: Path, query: str) -> None:
    assert search(home, SearchKeywords(query)).hits == ()


def test_a_nul_parts_words(home: Path) -> None:
    assert ids(home, SearchKeywords("dark\N{NULL}halo")) == ["notes/halo.md"]


def test_a_query_outside_the_grammar_says_how_to_write_one(home: Path) -> None:
    with pytest.raises(InvalidRequestError, match="a quote is not closed"):
        search(home, SearchKeywords('halo "dark'))


def test_k_counts_results_from_one_to_ten_thousand_and_is_held_to_the_most_there_are(home: Path) -> None:
    with pytest.raises(InvalidRequestError, match=r"^0 is outside 1 to 10,000$"):
        search(home, SearchKeywords("halo", k=0))
    with pytest.raises(InvalidRequestError, match="10,001 is outside 1 to 10,000"):
        search(home, SearchKeywords("halo", k=10_001))
    assert ids(home, SearchKeywords("note", k=10_000)) == [f"notes/filler{n}.md" for n in range(8)]


def test_a_query_outside_the_grammar_is_refused_before_any_store_is_opened(home: Path) -> None:
    store = home / "collections" / "notes.sqlite"
    delete_store(store)
    create_store(store, CollectionId(1), LAYOUT[:1])
    with pytest.raises(InvalidRequestError, match="a quote is not closed"):
        search(home, SearchKeywords('halo "dark'))
    with contextlib.closing(sqlite3.connect(store)) as database:
        assert database.execute("PRAGMA user_version").fetchone() == (1,)


def test_one_collection_is_searched_when_named(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nhalo\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    assert set(ids(home, SearchKeywords("halo"))) == {"notes/halo.md", "more/halo.md"}
    assert ids(home, SearchKeywords("halo", collections=("more",))) == ["more/halo.md"]
    with pytest.raises(NotFoundError):
        search(home, SearchKeywords("halo", collections=("nope",)))


def test_a_collection_never_indexed_keeps_no_other_from_being_searched(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "later" / "halo.md", "# Halo later\n\nhalo\n")
    add(home, AddCollection(str(tmp_path / "later")))
    result = search(home, SearchKeywords("halo"))
    assert ([hit.id for hit in result.hits], result.searched, result.unsearched) == (["notes/halo.md"], 2, ())


def test_without_any_collection_a_search_searches_none(tmp_path: Path) -> None:
    assert search(tmp_path / "nowhere", SearchKeywords("halo")) == SearchResult((), 0, (), scored_by="keywords")


def test_changes_and_removals_are_followed(home: Path, notes: Path) -> None:
    (notes / "halo.md").unlink()
    write(notes / "new.md", "# New\n\nA halo of light.\n")
    indexed(home)
    assert ids(home, SearchKeywords("halo")) == ["notes/new.md"]


def test_a_document_about_a_word_outranks_one_that_names_it_in_its_title(home: Path, notes: Path) -> None:
    write(notes / "titled.md", "# Entropy\n\n" + "Something else entirely, at length. " * 30 + "\n")
    write(notes / "about.md", "# Thermodynamics\n\nEntropy grows. Entropy is a state function. Entropy.\n")
    indexed(home)
    assert ids(home, SearchKeywords("entropy"))[0] == "notes/about.md"


def test_a_store_that_cannot_be_read_is_passed_over_and_said(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nhalo\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    (home / "collections" / "more.sqlite").unlink()
    result = search(home, SearchKeywords("halo"))
    assert [hit.id for hit in result.hits] == ["notes/halo.md"]
    assert result.searched == 1
    (missing,) = result.unsearched
    assert dataclasses.replace(missing, message="") == Notice(Problem.STORE_MISSING, "", MISSING_REMEDY, "more")


@pytest.mark.parametrize("table", ["keyword_postings", "chunks", "keyword_segments", "sources"])
def test_a_store_damaged_where_a_search_reads_it_is_passed_over_and_the_others_searched(
    home: Path, tmp_path: Path, table: str
) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nhalo\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    damage_table(home / "collections" / "notes.sqlite", table)
    result = search(home, SearchKeywords("halo"))
    assert [hit.id for hit in result.hits] == ["more/halo.md"]
    assert [(each.collection, each.remedy) for each in result.unsearched] == [("notes", DAMAGED_REMEDY)]


def _more(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nhalo\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)


@pytest.mark.parametrize(
    "sql, query",
    [
        ("UPDATE keyword_postings SET postings = x'02'", "halo"),
        ("UPDATE keyword_postings SET positions = x'0001'", '"halo models"'),
        ("DELETE FROM keyword_lengths", "halo"),
        ("DELETE FROM chunks WHERE document = (SELECT id FROM documents WHERE title = 'Halo')", "halo"),
    ],
)
def test_a_keyword_index_whose_data_is_damaged_is_passed_over_and_the_others_searched(
    home: Path, tmp_path: Path, sql: str, query: str
) -> None:
    _more(home, tmp_path)
    changed(home / "collections" / "notes.sqlite", sql)
    result = search(home, SearchKeywords(query))
    (said,) = result.unsearched
    assert (said.collection, said.remedy) == ("notes", DAMAGED_REMEDY)
    assert "is damaged" in said.message
    assert result.searched == 1


def test_a_store_lost_while_searched_weighs_on_no_score(home: Path, tmp_path: Path) -> None:
    _more(home, tmp_path)
    alone = search(home, SearchKeywords("halo", collections=("more",))).hits
    changed(home / "collections" / "notes.sqlite", "DELETE FROM keyword_lengths")
    result = search(home, SearchKeywords("halo"))
    assert result.hits == alone
    assert [each.collection for each in result.unsearched] == ["notes"]


def test_a_store_lost_in_the_looser_search_weighs_on_no_score(home: Path, tmp_path: Path) -> None:
    _more(home, tmp_path)
    alone = search(home, SearchKeywords("halo pruning", collections=("more",))).hits
    assert search(home, SearchKeywords("halo AND pruning")).hits == ()
    changed(
        home / "collections" / "notes.sqlite",
        "DELETE FROM chunks WHERE document = (SELECT id FROM documents WHERE title = 'Halo')",
    )
    result = search(home, SearchKeywords("halo pruning"))
    assert result.hits == alone != ()
    assert [each.collection for each in result.unsearched] == ["notes"]


def test_a_store_lost_while_its_passages_are_read_weighs_on_no_score(home: Path, tmp_path: Path) -> None:
    _more(home, tmp_path)
    alone = search(home, SearchKeywords("halo", collections=("more",))).hits
    damage_table(home / "collections" / "notes.sqlite", "sources")
    result = search(home, SearchKeywords("halo"))
    assert result.hits == alone
    assert [each.collection for each in result.unsearched] == ["notes"]


def test_two_stores_lost_in_one_search_leave_the_third_to_answer(home: Path, tmp_path: Path) -> None:
    _more(home, tmp_path)
    write(tmp_path / "third" / "halo.md", "# Halo third\n\nhalo\n")
    add(home, AddCollection(str(tmp_path / "third")))
    indexed(home)
    alone = search(home, SearchKeywords("halo", collections=("more",))).hits
    changed(home / "collections" / "notes.sqlite", "DELETE FROM keyword_lengths")
    damage_table(home / "collections" / "third.sqlite", "sources")
    result = search(home, SearchKeywords("halo"))
    assert result.hits == alone
    assert sorted(str(each.collection) for each in result.unsearched) == ["notes", "third"]
    assert result.searched == 1


def test_a_keyword_index_of_another_version_is_passed_over_until_indexed_again(home: Path) -> None:
    changed(home / "collections" / "notes.sqlite", "UPDATE meta SET value = 'older' WHERE key = 'keyword_fingerprint'")
    (said,) = search(home, SearchKeywords("halo")).unsearched
    assert (said.collection, said.remedy) == ("notes", REMAKE_REMEDY)
    indexed(home)
    assert ids(home, SearchKeywords("halo")) == ["notes/halo.md"]


@pytest.mark.parametrize(
    "marked, said, remedy",
    [
        (
            "UPDATE meta SET value = 'remaking ' || value WHERE key = 'keyword_fingerprint'",
            "the store's keyword index is not yet wholly remade",
            REMAKE_REMEDY,
        ),
        (
            "UPDATE meta SET value = 'remaking other' WHERE key = 'keyword_fingerprint'",
            "another sift began remaking this keyword index for itself and has not finished",
            TAKEN_OVER_REMEDY,
        ),
    ],
)
def test_a_keyword_index_being_remade_is_passed_over_saying_by_whom(
    home: Path, marked: str, said: str, remedy: str
) -> None:
    changed(home / "collections" / "notes.sqlite", marked)
    (unsearched,) = search(home, SearchKeywords("halo")).unsearched
    assert (unsearched.message, unsearched.remedy) == (said, remedy)
    indexed(home)
    assert ids(home, SearchKeywords("halo")) == ["notes/halo.md"]


def test_an_index_of_another_version_is_remade_by_indexing_even_with_its_folder_gone(home: Path, notes: Path) -> None:
    changed(home / "collections" / "notes.sqlite", "UPDATE meta SET value = 'older' WHERE key = 'keyword_fingerprint'")
    notes.rename(notes.with_name("gone"))
    said: list[Event] = []
    (outcome,) = index(home, IndexCollections(), said.append).collections
    assert isinstance(outcome.outcome, Skipped)
    (shown,) = status(home, ShowStatus()).collections
    assert isinstance(shown.contents, Counted)
    chunks = shown.contents.chunks
    assert said[1] == PhaseStarted(Phase.REMAKE, chunks)
    assert isinstance(said[-2], PhaseDone)
    assert said[-2].count == chunks
    assert ids(home, SearchKeywords("halo")) == ["notes/halo.md"]


def test_an_empty_collection_of_another_version_is_current_once_indexed(home: Path, tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    add(home, AddCollection(str(tmp_path / "empty")))
    indexed(home)
    changed(home / "collections" / "empty.sqlite", "UPDATE meta SET value = 'older' WHERE key = 'keyword_fingerprint'")
    indexed(home)
    assert search(home, SearchKeywords("halo", collections=("empty",))) == SearchResult((), 1, (), scored_by="keywords")


@pytest.mark.parametrize("query, said", [("", "query is empty"), (" ", EMPTY_QUERY), ("\t\n", EMPTY_QUERY)])
def test_an_empty_query_is_refused(home: Path, query: str, said: str) -> None:
    with pytest.raises(InvalidRequestError, match=said):
        search(home, SearchKeywords(query))
