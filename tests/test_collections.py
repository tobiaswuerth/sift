"""Several collections, each in a store of its own: searched as one, some of them inactive, a store missing."""

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from sift import cli, embed, indexing, store
from sift.catalog import Catalog
from sift.search import Index
from sift.server import _collections

from .conftest import write


def ids(hits):
    return [h["id"] for h in hits]


@pytest.fixture
def three(tmp_path, home, fake_models):
    """Three collections, indexed and embedded: papers, recipes and diary, in stores of their own."""
    write(
        tmp_path / "papers" / "robots.md", "---\nyear: 2024\n---\n# Painting robots\n\nA robot paints brush strokes.\n"
    )
    write(tmp_path / "papers" / "rl.md", "---\nyear: 2019\n---\n# Reward shaping\n\nReward shaping speeds learning.\n")
    write(tmp_path / "recipes" / "soup.md", "---\nyear: 2021\n---\n# Tomato soup\n\nA soup a robot could cook.\n")
    write(tmp_path / "recipes" / "bread.md", "---\nyear: old\n---\n# Bread\n\nFlour, water, salt.\n")
    write(tmp_path / "diary" / "monday.md", "# Monday\n\nThe robot vacuum broke again.\n")
    catalog = Catalog(home)
    for name in ("papers", "recipes", "diary"):
        catalog.add(tmp_path / name, name)
    indexing.index(catalog)
    embed.embed_pending(catalog)
    return catalog


def test_searches_cover_every_collection(three):
    index = Index(three)
    assert sorted(ids(index.search("robot"))) == ["diary/monday.md", "papers/robots.md", "recipes/soup.md"]
    assert ids(index.vsearch("robot paints brush strokes", k=1)) == ["papers/robots.md"]
    assert len(index.vsearch("robot", k=10)) == 5
    assert "recipes/soup.md" in ids(index.query("a robot that cooks soup", k=3))
    assert ids(index.search("robot", collection="recipes")) == ["recipes/soup.md"]


def test_the_best_of_every_collection_come_first(three):
    # Each store gives its best; together they are ordered as one.
    index = Index(three)
    hits = index.search("robot", k=2)
    assert len(hits) == 2
    assert [h["score"] for h in hits] == sorted((h["score"] for h in index.search("robot")), reverse=True)[:2]


def test_browsing_every_collection_by_a_field_is_in_order(three):
    # Numbers before text, as SQLite orders them, across the stores; those without the field last.
    index = Index(three)
    assert ids(index.browse(sort="year")) == [
        "papers/rl.md",
        "recipes/soup.md",
        "papers/robots.md",
        "recipes/bread.md",
        "diary/monday.md",
    ]
    assert ids(index.browse(sort="-year")) == [
        "recipes/bread.md",
        "papers/robots.md",
        "recipes/soup.md",
        "papers/rl.md",
        "diary/monday.md",
    ]
    assert ids(index.browse()) == [
        "diary/monday.md",
        "papers/rl.md",
        "papers/robots.md",
        "recipes/bread.md",
        "recipes/soup.md",
    ]
    assert ids(index.browse(k=2, sort="year")) == ["papers/rl.md", "recipes/soup.md"]


def test_similar_finds_documents_of_other_collections(three):
    assert "recipes/soup.md" in ids(Index(three).similar("papers/robots.md", k=4))


# --- inactive collections ------------------------------------------------------


def test_an_inactive_collection_is_passed_over(three):
    index = Index(three)
    index.vsearch("robot")  # its codes in memory, as in a running server
    three.set_active(["diary"], False)
    assert sorted(ids(index.search("robot"))) == ["papers/robots.md", "recipes/soup.md"]
    assert "diary/monday.md" not in ids(index.vsearch("robot vacuum", k=5))
    assert index.vectors.keys() == {three.get("papers").id, three.get("recipes").id}  # its codes let go
    three.set_active(["diary"], True)
    assert ids(index.vsearch("robot vacuum broke", k=1)) == ["diary/monday.md"]


def test_an_inactive_collection_named_is_an_error_that_says_how_to_search_it(three):
    three.set_active(["diary"], False)
    index = Index(three)
    for call in (
        lambda: index.search("robot", collection="diary"),
        lambda: index.get("diary/monday.md"),
        lambda: index.folders("diary"),
        lambda: index.unresolved("diary"),
    ):
        with pytest.raises(LookupError, match=r"collection 'diary' is inactive: `sift activate diary`"):
            call()


def test_an_inactive_collection_is_left_out_of_index_and_embed_unless_named(three, tmp_path, capsys):
    three.set_active(["diary"], False)
    write(tmp_path / "diary" / "tuesday.md", "# Tuesday\n\nNothing broke.\n")
    write(tmp_path / "papers" / "new.md", "# New\n\nA new paper.\n")
    capsys.readouterr()
    indexing.index(three)
    assert "diary: inactive, skipped" in capsys.readouterr().err
    assert three.store("diary").execute("SELECT count(*) FROM docs").fetchone() == (1,)
    embed.embed_pending(three)
    assert "embedding 1 chunk with" in capsys.readouterr().err  # the paper's
    indexing.index(three, ["diary"])
    embed.embed_pending(three, collections=["diary"])
    assert three.store("diary").execute("SELECT count(*) FROM vectors").fetchone() == (2,)


def test_a_file_of_an_inactive_collection_can_be_reindexed(three, tmp_path):
    # Its store stays up to date for when it is active again.
    three.set_active(["diary"], False)
    write(tmp_path / "diary" / "tuesday.md", "# Tuesday\n\nNothing broke.\n")
    assert Index(three).reindex(["diary/tuesday.md"])[0]["status"] == "added"
    three.set_active(["diary"], True)
    assert ids(Index(three).search("nothing broke")) == ["diary/tuesday.md"]


def test_the_status_shows_whether_a_collection_is_active_and_where_its_index_lies(three, home):
    three.set_active(["diary"], False)
    status = Index(three).status()
    diary, papers, _ = status["collections"]
    assert diary == {
        "name": "diary",
        "path": three.get("diary").path,
        "active": False,
        "store": str(home / "collections" / "diary.sqlite"),
        "size": diary["size"],
    }
    assert diary["size"] > 0
    assert (papers["active"], papers["store"], papers["docs"]) == (True, str(home / "collections" / "papers.sqlite"), 2)


def test_a_server_tells_agents_of_the_active_collections_only(three):
    three.set_active(["diary"], False)
    told = _collections(Index(three))
    assert "papers (" in told
    assert "recipes (" in told
    assert "diary" not in told
    three.set_active(["papers", "recipes"], False)
    assert _collections(Index(three)) == "No collections are active."


def test_a_server_tells_agents_the_document_counts_kept(three):
    # Not counted anew at each start: that took a large collection seconds to minutes.
    db = three.store("papers")
    db.execute("UPDATE counts SET n = 1234567 WHERE name = 'docs'")
    db.commit()
    assert "papers (" in _collections(Index(three))
    assert "1,234,567 documents" in _collections(Index(three))


# --- a store missing -----------------------------------------------------------


def test_a_missing_store_is_passed_over_and_said(three, capsys):
    # Its drive isn't connected: the others are searched still, and the status says which is missing.
    stored = three.get("diary").store
    three.release(three.get("diary").id)
    os.replace(stored, stored + ".away")
    index = Index(three)
    assert sorted(ids(index.search("robot"))) == ["papers/robots.md", "recipes/soup.md"]
    assert ids(index.vsearch("robot vacuum", k=1)) != ["diary/monday.md"]
    err = capsys.readouterr().err
    assert err.count(f"diary: its index is missing: {stored}") == 1  # said once
    assert index.status()["collections"][0] == {
        "name": "diary",
        "path": three.get("diary").path,
        "active": True,
        "store": stored,
        "missing": True,
    }
    with pytest.raises(LookupError, match=r"diary: its index is missing: .* is its drive connected"):
        index.search("robot", collection="diary")
    assert not os.path.exists(stored)  # never made anew
    os.replace(stored + ".away", stored)
    assert "diary/monday.md" in ids(index.search("robot"))


def unusable(path, how):
    """Make the store at `path` one sift can't open: damaged, or updated by a newer sift."""
    if how == "damaged":
        Path(path).write_bytes(b"not a database, " * 1000)
        return
    db = sqlite3.connect(path)
    db.execute(f"PRAGMA user_version = {store.SCHEMA_VERSION + 1}")
    db.close()


@pytest.mark.parametrize("how, said", [("damaged", "file is not a database"), ("newer", "made by a newer sift")])
def test_a_store_that_cant_be_opened_is_passed_over_and_said(three, tmp_path, capsys, how, said):
    # Damaged, or updated by a newer sift, one store stopped `sift index` and `sift embed` at it, kept `sift serve`
    # from starting, and failed every search of all collections and the status, which would have said why.
    stored = three.get("diary").store
    three.release(three.get("diary").id)
    unusable(stored, how)
    write(tmp_path / "papers" / "new.md", "# New\n\nA new robot paper.\n")
    indexing.index(three)  # diary comes first: the others after it are indexed
    embed.embed_pending(three)
    index = Index(three)
    assert sorted(ids(index.search("robot"))) == ["papers/new.md", "papers/robots.md", "recipes/soup.md"]
    assert ids(index.vsearch("a new robot paper", k=1)) == ["papers/new.md"]
    (diary,) = [c for c in index.status()["collections"] if c["name"] == "diary"]
    assert said in diary["error"]
    assert "- diary (" not in _collections(Index(three))  # what a server tells agents
    assert said in capsys.readouterr().err
    with pytest.raises(LookupError, match=said):
        index.search("robot", collection="diary")


def test_a_store_that_fails_to_open_is_passed_over_and_said(three, tmp_path, capsys, monkeypatch):
    # Locked by another program, a backup or a virus scanner, or out of reach: "disk I/O error" on opening one store
    # failed every search of all collections, the status, a server's start, and `sift index` and `sift embed`.
    stored, real = three.get("diary").store, store.connect
    three.release(three.get("diary").id)

    def connect(path, cid, create=False):
        if str(path) == stored:
            raise sqlite3.OperationalError("disk I/O error")
        return real(path, cid, create)

    monkeypatch.setattr(store, "connect", connect)
    write(tmp_path / "papers" / "new.md", "# New\n\nA new robot paper.\n")
    indexing.index(three)
    embed.embed_pending(three)
    index = Index(three)
    assert sorted(ids(index.search("robot"))) == ["papers/new.md", "papers/robots.md", "recipes/soup.md"]
    (diary,) = [c for c in index.status()["collections"] if c["name"] == "diary"]
    assert diary["error"] == "diary: its index can't be opened: disk I/O error"
    assert "- diary (" not in _collections(Index(three))
    assert "diary: its index can't be opened: disk I/O error" in capsys.readouterr().err


def test_documents_of_a_collection_removed_meanwhile_are_left_out(three):
    index = Index(three)
    ranked = index._keyword("robot", 10, index.active().values(), "1", [])
    three.remove("diary")
    assert "diary/monday.md" not in ids(index._hits(ranked))


def test_a_model_of_another_size_is_passed_over_and_said(three, capsys):
    # A collection embedded with another model can't be searched with this one's vectors.
    diary = three.store("diary")
    chunks = [i for (i,) in diary.execute("SELECT id FROM chunks")]
    store.put_vectors(diary, chunks, np.full((len(chunks), 200), 0.5, np.float16))  # codes of 4 words, not 1
    store.touch(diary)
    diary.commit()
    capsys.readouterr()
    assert "diary/monday.md" not in ids(Index(three).vsearch("robot vacuum broke", k=5))
    assert "diary was embedded with another model" in capsys.readouterr().err


# --- the command line ----------------------------------------------------------


def run(capsys, *argv):
    cli.main(list(argv))
    return capsys.readouterr()


def test_collections_are_deactivated_and_activated_from_the_command_line(three, capsys):
    assert "diary, recipes are inactive" in run(capsys, "deactivate", "diary", "recipes").err
    assert "diary was deactivated already" in run(capsys, "deactivate", "diary").err
    status = json.loads(run(capsys, "status").out)
    assert {c["name"]: c["active"] for c in status["collections"]} == {
        "diary": False,
        "papers": True,
        "recipes": False,
    }
    assert "recipes is active" in run(capsys, "activate", "recipes").err
    with pytest.raises(SystemExit, match="no collection named 'nope'"):
        cli.main(["activate", "nope"])


def test_stores_are_moved_from_the_command_line(three, tmp_path, capsys):
    for c in three.collections():
        three.release(c.id)  # no other sift may have them open
    assert "moved the index of diary to" in run(capsys, "move", "diary", str(tmp_path / "away")).err
    assert three.get("diary").store == str(tmp_path / "away" / "diary.sqlite")
    run(capsys, "move", "papers", str(tmp_path / "away" / "data_papers.sqlite"))
    three.release(three.get("papers").id)
    assert three.get("papers").store == str(tmp_path / "away" / "data_papers.sqlite")
    run(capsys, "move", str(tmp_path / "all"))
    assert {c.store for c in three.collections()} == {
        str(tmp_path / "all" / name) for name in ("diary.sqlite", "data_papers.sqlite", "recipes.sqlite")
    }
    assert sorted(ids(Index(three).search("robot"))) == ["diary/monday.md", "papers/robots.md", "recipes/soup.md"]


def test_a_collection_is_added_with_its_store_elsewhere(home, tmp_path, capsys):
    write(tmp_path / "notes" / "a.md", "# A\n")
    err = run(capsys, "add", str(tmp_path / "notes"), "--store", str(tmp_path / "big" / "data_notes.sqlite")).err
    assert f"its index at {tmp_path / 'big' / 'data_notes.sqlite'}" in err
    run(capsys, "index")
    assert ids(Index(Catalog(home)).search("A")) == ["notes/a.md"]


def test_a_collection_not_embedded_yet_is_searched_by_keywords_only(three, tmp_path):
    write(tmp_path / "fresh" / "a.md", "# Robot fresh\n\nNot embedded yet.\n")
    three.add(tmp_path / "fresh", "fresh")
    indexing.index(three)
    index = Index(three)
    assert "fresh/a.md" in ids(index.search("robot"))
    assert "fresh/a.md" not in ids(index.vsearch("robot fresh", k=10))


def test_indexing_passes_over_a_collection_whose_store_is_missing(three, capsys):
    stored = three.get("diary").store
    three.release(three.get("diary").id)
    os.replace(stored, stored + ".away")
    capsys.readouterr()
    indexing.index(three)
    assert f"diary: its index is missing: {stored}" in capsys.readouterr().err
    assert not os.path.exists(stored)


def test_files_of_several_collections_are_reindexed_at_once(three, tmp_path):
    write(tmp_path / "papers" / "rl.md", "# Reward shaping\n\nShaped rewards, changed.\n")
    write(tmp_path / "recipes" / "soup.md", "# Tomato soup\n\nWith basil now.\n")
    found = Index(three).reindex(["recipes/soup.md", "papers/rl.md", "diary/monday.md"])
    assert [(f["id"], f["status"]) for f in found] == [
        ("recipes/soup.md", "changed"),
        ("papers/rl.md", "changed"),
        ("diary/monday.md", "unchanged"),
    ]
    assert ids(Index(three).search("basil")) == ["recipes/soup.md"]


def test_with_every_collection_inactive_a_search_says_how_to_activate_one(three):
    three.set_active(["diary", "papers", "recipes"], False)
    index = Index(three)
    assert index.search("robot") == []
    with pytest.raises(LookupError, match="no collection is active: `sift activate NAME`"):
        index.vsearch("robot")


def test_a_note_whose_collection_is_removed_meanwhile_is_a_clear_error(three, monkeypatch):
    # Another sift removed the collection between finding the note and reading its links.
    index = Index(three)
    found = index._note("diary/monday.md")
    three.remove("diary")
    monkeypatch.setattr(Index, "_note", lambda self, id: found)
    with pytest.raises(LookupError, match="its collection was removed meanwhile"):
        index.links("diary/monday.md")


def test_embedding_passes_over_a_collection_whose_store_is_missing_unless_named(three, tmp_path, capsys):
    write(tmp_path / "papers" / "new.md", "# New\n\nA new paper.\n")
    indexing.index(three)
    stored = three.get("diary").store
    three.release(three.get("diary").id)
    os.replace(stored, stored + ".away")
    capsys.readouterr()
    embed.embed_pending(three)
    err = capsys.readouterr().err
    assert f"diary: its index is missing: {stored}" in err
    assert "embedding 1 chunk with" in err  # the new paper's
    with pytest.raises(LookupError, match="diary: its index is missing"):
        embed.embed_pending(three, collections=["diary"])


# --- keyword scores over several collections -------------------------------------


@pytest.fixture
def split_and_whole(tmp_path, fake_models):
    """The same files as two collections, one small, and as one: (two collections, one)."""
    texts = {
        "big/a.md": "# Caching\n\nA cache in front of the database. Recall the hit rate.\n",
        "big/b.md": "# Queues\n\nA queue between services absorbs bursts.\n",
        "big/c.md": "# Logging\n\nStructured logs for every request, and a benchmark of their cost.\n",
        "big/d.md": "# Retries\n\nRetries with backoff and jitter.\n",
        "big/e.md": "# Testing\n\nUnit tests first, then integration tests.\n",
        "small/bench.md": "# Benchmark\n\nRecall at ten, in a benchmark of recall and precision.\n",
        # tens of thousands of tokens: FTS5 keeps their counts in varints of two and three bytes
        "big/long.md": "# Notes\n\n" + "\n\n".join(f"Note {i}: a cache, a recall of tests." for i in range(3000)),
    }
    for path, text in texts.items():
        write(tmp_path / "all" / path, text)
    two = Catalog(tmp_path / "two")
    two.add(tmp_path / "all" / "big", "big")
    two.add(tmp_path / "all" / "small", "small")
    one = Catalog(tmp_path / "one")
    one.add(tmp_path / "all", "all")
    for catalog in (two, one):
        indexing.index(catalog)
    return two, one


def scores(hits):
    """Each hit's score, by its file's path in the folder `all` and its lines: the same of two collections and one."""
    return {(h["id"].removeprefix("all/"), h["lines"]): h["score"] for h in hits}


@pytest.mark.parametrize(
    "query",
    [
        "benchmark recall",
        "recall",
        '"hit rate" OR benchmark',
        "recall NOT cache",
        "(recall OR queue) AND a",
        "bench*",
        "bench *",
        '"bench" *',
        "tests",
    ],
)
def test_keyword_scores_over_several_collections_are_those_of_one_index(split_and_whole, query):
    # Each collection's word counts alone would score a small collection's terms as common: a page about benchmarks,
    # alone in its collection, would sink below every passage of the others.
    two, one = split_and_whole
    for all_passages in (False, True):
        apart = Index(two).search(query, all_passages=all_passages)
        together = Index(one).search(query, all_passages=all_passages)
        assert scores(apart) == pytest.approx(scores(together), abs=1e-3)
        assert [h["score"] for h in apart] == sorted((h["score"] for h in apart), reverse=True)


def test_a_small_collections_page_ranks_as_in_one_index(split_and_whole):
    two, _ = split_and_whole
    assert ids(Index(two).search("benchmark recall"))[0] == "small/bench.md"


def test_a_query_that_places_phrases_keeps_each_collections_scores(split_and_whole):
    # NEAR, column filters and the like aren't taken apart into phrases: such hits are ranked by their own
    # collection's word counts.
    two, _ = split_and_whole
    assert ids(Index(two).search("title: benchmark")) == ["small/bench.md"]
    assert set(ids(Index(two).search("NEAR(benchmark recall)"))) == {"small/bench.md"}


def test_a_documents_best_passage_is_that_of_one_index(tmp_path, fake_models):
    # Its own collection's word counts can pick another passage as a document's best: in a collection of one
    # document, a word in every passage counts for nothing there, and the passage of a rarer word wins.
    filler = " ".join(f"word{i}" for i in range(400))
    texts = {
        "small/doc.md": f"# Doc\n\n## Alpha\n\nalpha alpha alpha alpha {filler}\n\n## Beta\n\nbeta alpha {filler}\n",
        **{f"big/{i}.md": f"# Note {i}\n\nbeta and gamma, {i}.\n" for i in range(8)},
    }
    for path, text in texts.items():
        write(tmp_path / "all" / path, text)
    two, one = Catalog(tmp_path / "two"), Catalog(tmp_path / "one")
    two.add(tmp_path / "all" / "big", "big")
    two.add(tmp_path / "all" / "small", "small")
    one.add(tmp_path / "all", "all")
    for catalog in (two, one):
        indexing.index(catalog)
    assert two.store("small").execute("SELECT count(*) FROM chunks").fetchone()[0] > 1

    def best(catalog, prefix):
        hit = next(h for h in Index(catalog).search("alpha OR beta", k=20) if h["id"].startswith(prefix))
        return hit["lines"], hit["score"]

    assert best(two, "small/doc.md") == pytest.approx(best(one, "all/small/doc.md"), abs=1e-3)
