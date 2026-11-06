"""Several collections, each in a store of its own: searched as one, some of them inactive, a store missing."""

import json
import os

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
