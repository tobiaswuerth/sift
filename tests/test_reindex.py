"""Indexing files again on demand: a note written a moment ago is found by the next search, its links resolve, and a
running server follows without loading every vector again."""

import os

import numpy as np
import pytest

from sift import indexing, readers, store
from sift.search import Index

from .conftest import write

NEW = "vault/06_Topics/Wartungsvertrag 2027.md"
OFFER = "vault/03_Interactions/2026-10-14 Offerte.md"


def ids(hits):
    return [h["id"] for h in hits]


@pytest.fixture
def live(linked, fake_models):
    """The vault, embedded, and searched once, so its vectors are in memory as in a running server: (index, vault)."""
    from sift import embed

    db, vault, _ = linked
    embed.embed_pending(db)
    index = Index(db)
    index.vsearch("anything")
    return index, vault


def test_a_new_note_is_found_at_once(live, monkeypatch):
    index, vault = live
    write(vault / "06_Topics" / "Wartungsvertrag 2027.md", "## Summary\n\nMaintenance of the press line, from 2027.\n")
    reads = []
    monkeypatch.setattr(
        readers, "load", lambda path, fields=None, real=readers.load: reads.append(path) or real(path, fields)
    )
    assert index.reindex([NEW]) == [{"id": NEW, "status": "added", "chunks": 1, "embedded": 1}]
    assert len(reads) == 1  # the offer linking it isn't read again
    assert ids(index.search("press line")) == [NEW]
    monkeypatch.setattr(index.models, "vectors", lambda *args: pytest.fail("all vectors loaded again"))
    assert ids(index.vsearch("maintenance of the press line", k=1)) == [NEW]
    assert index.links(OFFER)["out"][1]["id"] == NEW


def test_a_changed_note(live):
    index, vault = live
    hans = vault / "01_People" / "Hans Müller.md"
    write(hans, hans.read_text(encoding="utf-8").replace("Zürich", "Winterthur"))
    assert index.reindex([str(hans)]) == [
        {"id": "vault/01_People/Hans Müller.md", "status": "changed", "chunks": 1, "embedded": 1}
    ]
    assert ids(index.search("Winterthur")) == ["vault/01_People/Hans Müller.md"]
    assert index.search("Zürich") == []
    assert ids(index.vsearch("Winterthur", k=1)) == ["vault/01_People/Hans Müller.md"]
    assert index.reindex([str(hans)])[0]["status"] == "unchanged"


def test_a_deleted_note(live):
    index, vault = live
    (vault / "02_Organizations" / "Acme AG.md").unlink()
    assert index.reindex(["vault/02_Organizations/Acme AG.md"]) == [
        {"id": "vault/02_Organizations/Acme AG.md", "status": "removed"}
    ]
    assert "vault/02_Organizations/Acme AG.md" not in ids(index.vsearch("industrial customer"))
    assert index.links("vault/01_People/Hans Müller.md")["out"][0]["unresolved"]
    assert index.reindex(["vault/02_Organizations/Acme AG.md", "vault/Nowhere.md"]) == [
        {"id": "vault/02_Organizations/Acme AG.md", "status": "not found"},
        {"id": "vault/Nowhere.md", "status": "not found"},
    ]


def test_vectors_in_memory_follow_every_change(live):
    # What a running server holds must equal what it would load anew.
    index, vault = live
    write(vault / "a.md", "Alpha note.\n")
    write(vault / "b.md", "Beta note.\n")
    index.reindex(["vault/a.md", "vault/b.md"])
    write(vault / "a.md", "Alpha note, changed.\n")
    (vault / "b.md").unlink()
    index.reindex(["vault/a.md", "vault/b.md"])
    held = dict(zip(index.vectors.ids.tolist(), index.vectors.scores(np.ones(64, np.float16) / 8).tolist()))
    index.version = None
    index.load()
    loaded = dict(zip(index.vectors.ids.tolist(), index.vectors.scores(np.ones(64, np.float16) / 8).tolist()))
    assert held == pytest.approx(loaded)


def held_and_loaded(index):
    """The vectors a running server holds, and those it would load anew: {chunk id: score} each, and the ids."""
    query = np.ones(64, np.float16) / 8
    held = (index.vectors.ids.tolist(), index.vectors.scores(query).tolist())
    fresh = Index(index.db, index.models)
    fresh.load()
    return held, (fresh.vectors.ids.tolist(), fresh.vectors.scores(query).tolist())


def test_a_note_rewritten_under_its_old_chunk_ids(live):
    # The newest note's chunks hold the highest ids, which SQLite hands out again once they are deleted: its new
    # passage came under the old id, the server kept the old vector next to the new one, and the old wording still
    # found the note.
    index, vault = live
    write(vault / "Beat Keller.md", "Segelt Regatten auf dem Bodensee.\n")
    index.reindex(["vault/Beat Keller.md"])
    write(vault / "Beat Keller.md", "Arbeitet bei Acme AG im Einkauf.\n")
    assert index.reindex(["vault/Beat Keller.md"])[0]["status"] == "changed"
    (held_ids, held), (loaded_ids, loaded) = held_and_loaded(index)
    assert held_ids == loaded_ids
    assert held == pytest.approx(loaded)
    assert "vault/Beat Keller.md" not in ids(index.vsearch("Regatten Bodensee", k=1))


def test_a_note_rewritten_under_its_old_chunk_ids_while_no_model_answers(live, monkeypatch):
    # Not embedded now, the new passage has no vector yet: the old one, under the same id, must go all the same.
    index, vault = live
    write(vault / "Beat Keller.md", "Segelt Regatten auf dem Bodensee.\n")
    index.reindex(["vault/Beat Keller.md"])

    def unreachable(texts, query=False):
        raise ConnectionError("embedding remote https://workstation:8766 does not answer")

    monkeypatch.setattr(index.models, "encode", unreachable)
    write(vault / "Beat Keller.md", "Arbeitet bei Acme AG im Einkauf.\n")
    assert index.reindex(["vault/Beat Keller.md"])[0]["embedded"] == 0
    monkeypatch.undo()
    (held_ids, _), (loaded_ids, _) = held_and_loaded(index)
    assert held_ids == loaded_ids


def test_vectors_in_memory_match_a_fresh_load_through_many_changes(live):
    # Notes written, extended, rewritten and deleted in random order, as an agent works: after each reindex the
    # server must hold what it would load anew.
    import random

    index, vault = live
    rng, words = random.Random(1), "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    for step in range(40):
        name = f"n{rng.randrange(6)}.md"
        if rng.random() < 0.2:
            (vault / name).unlink(missing_ok=True)
        else:
            parts = rng.randrange(1, 4)  # passages: a section of about 1,800 characters each
            write(
                vault / name,
                "".join(f"## Part {p}\n\n" + " ".join(rng.choices(words, k=300)) + "\n\n" for p in range(parts)),
            )
        index.reindex([f"vault/{name}"])
        (held_ids, held), (loaded_ids, loaded) = held_and_loaded(index)
        assert held_ids == loaded_ids, f"step {step}"
        assert held == pytest.approx(loaded), f"step {step}"


def test_vectors_changed_elsewhere_are_loaded_anew(live, monkeypatch):
    # Another process changed the index meanwhile: patching the vectors in memory would miss its changes.
    index, vault = live
    store.touch(index.db)
    index.db.commit()
    write(vault / "a.md", "Alpha note.\n")
    index.reindex(["vault/a.md"])
    loads, real = [], index.models.vectors
    monkeypatch.setattr(index.models, "vectors", lambda *args: loads.append(1) or real(*args))
    assert ids(index.vsearch("alpha note", k=1)) == ["vault/a.md"]
    assert loads == [1]


def test_a_json_lines_file(live):
    index, vault = live
    write(vault / "log.jsonl", '{"id": "1", "text": "First entry."}\n{"id": "2", "text": "Second entry."}\n')
    assert index.reindex(["vault/log.jsonl"]) == [
        {"id": "vault/log.jsonl", "status": "added", "chunks": 2, "embedded": 2}
    ]
    write(vault / "log.jsonl", '{"id": "1", "text": "First entry."}\n')
    assert index.reindex(["vault/log.jsonl"])[0] == {
        "id": "vault/log.jsonl",
        "status": "changed",
        "chunks": 1,
        "embedded": 1,
    }
    assert ids(index.browse(path="log/*")) == ["vault/log/1"]


def test_files_that_give_no_documents(live, monkeypatch):
    index, vault = live
    write(vault / "empty.jsonl", "not json\n")
    assert index.reindex(["vault/empty.jsonl"]) == [
        {"id": "vault/empty.jsonl", "status": "skipped", "note": "1 line skipped (line 1: not JSON)"}
    ]
    write(vault / "locked.md", "x")
    monkeypatch.setattr(readers, "load", lambda path, fields=None: None)
    assert index.reindex(["vault/locked.md"]) == [{"id": "vault/locked.md", "status": "unreadable"}]


def test_a_keyword_index_needs_no_models(linked, monkeypatch):
    # Without `sift embed` ever run, reindex embeds nothing, and loads no model.
    db, vault, _ = linked
    index = Index(db)
    monkeypatch.setattr(Index, "models", property(lambda self: pytest.fail("loaded a model")))
    write(vault / "a.md", "Alpha note.\n")
    assert index.reindex(["vault/a.md"]) == [{"id": "vault/a.md", "status": "added", "chunks": 1, "embedded": 0}]


def test_without_the_index_s_model_nothing_is_embedded(live):
    index, vault = live
    store.set_meta(index.db, "model", "another/model")
    index.db.commit()
    write(vault / "a.md", "Alpha note.\n")
    (found,) = index.reindex(["vault/a.md"])
    assert (found["status"], found["embedded"]) == ("added", 0)
    assert "index was embedded with another/model" in found["note"]


def test_without_a_worker_the_index_still_follows(live, monkeypatch):
    index, vault = live

    def unreachable(texts, query=False):
        raise ConnectionError("embedding remote https://workstation:8766 does not answer")

    monkeypatch.setattr(index.models, "encode", unreachable)
    write(vault / "a.md", "Alpha note.\n")
    (found,) = index.reindex(["vault/a.md"])
    assert (found["status"], found["embedded"]) == ("added", 0)
    assert found["note"] == "not embedded: embedding remote https://workstation:8766 does not answer"
    assert ids(index.search("alpha")) == ["vault/a.md"]  # keyword search finds it


def test_a_passage_the_model_breaks_stays_to_embed(live, monkeypatch):
    # Now and then a model overflows into NaN: that vector is never stored, and the passage stays to embed.
    index, vault = live
    real = index.models.encode
    monkeypatch.setattr(index.models, "encode", lambda texts, query=False: real(texts, query) * np.nan)
    write(vault / "a.md", "Alpha note.\n")
    (found,) = index.reindex(["vault/a.md"])
    assert (found["chunks"], found["embedded"]) == (1, 0)
    assert "could not be embedded" in found["note"]
    assert ids(index.search("alpha")) == ["vault/a.md"]


def test_the_same_file_twice_counts_once(live):
    index, vault = live
    write(vault / "a.md", "Alpha note.\n")
    found = index.reindex(["vault/a.md", str(vault / "a.md")])
    assert [f["status"] for f in found] == ["added"]


def test_a_path_in_another_case_names_the_file_as_it_is(live):
    # Windows and macOS find files in any case; indexed under another spelling, the file would be there twice.
    index, vault = live
    assert index.reindex(["vault/01_PEOPLE/hans müller.md"])[0] == {
        "id": "vault/01_People/Hans Müller.md",
        "status": "unchanged",
        "chunks": 1,
        "embedded": 1,
    }
    (vault / "02_Organizations" / "Acme AG.md").unlink()
    assert index.reindex(["vault/02_organizations/acme ag.md"])[0]["status"] == "removed"


@pytest.mark.parametrize(
    "path, error",
    [
        ("nope/a.md", "no collection named 'nope'"),
        ("vault", "a file of the collection"),
        ("vault/../outside.md", "outside the collection"),
        ("vault/.obsidian/workspace.md", "sift indexes no files in .obsidian/"),
        ("vault/node_modules/x.md", "sift indexes no files in node_modules/"),
        ("vault/picture.png", "sift doesn't index .png files"),
        ("vault/notes", "sift doesn't index files without an extension"),
    ],
)
def test_reindex_refuses_what_sift_index_would_not_read(live, path, error):
    index = live[0]
    with pytest.raises((LookupError, ValueError), match=error):
        index.reindex(["vault/a.md", path])


def test_reindex_refuses_what_siftignore_leaves_out(live):
    index, vault = live
    write(vault / ".siftignore", "Private/\n*.tmp.md\n")
    write(vault / "Private" / "keep.md", "x")
    for path in ("vault/Private/keep.md", "vault/a.tmp.md"):
        with pytest.raises(ValueError, match=r"\.siftignore leaves it out"):
            index.reindex([path])


def test_reindex_refuses_files_outside_every_collection(live, tmp_path):
    index, vault = live
    write(tmp_path / "elsewhere" / "a.md", "x")
    with pytest.raises(ValueError, match="is in no collection"):
        index.reindex([str(tmp_path / "elsewhere" / "a.md")])
    with pytest.raises(ValueError, match="is in no collection"):
        index.reindex([str(vault) + "-other/a.md"])  # a folder whose name starts like the collection's


def test_a_file_in_no_collection_says_which_there_are_and_how_to_add_one(live, tmp_path):
    # An agent told only "is in no collection" tried other spellings of the path until it gave up.
    index, vault = live
    outside = tmp_path / "archive" / "2026" / "x.md"
    write(outside, "x")
    with pytest.raises(ValueError, match="is in no collection") as refused:
        index.reindex([str(outside)])
    message = str(refused.value)
    assert message.startswith(f"{outside} is in no collection. The collections and their folders: ")
    assert f"vault: {vault}" in message
    assert f"work: {tmp_path / 'work'}" in message
    assert message.endswith(f"To index files there, add their folder as a collection: `sift add {outside.parent}`")


def test_a_file_in_no_collection_nearest_first(home, tmp_path):
    db = store.connect(home)
    for name in ("a", "b"):
        (tmp_path / name / "notes").mkdir(parents=True)
    store.add_collection(db, tmp_path / "a", "far")
    store.add_collection(db, tmp_path / "b" / "notes", "near")
    write(tmp_path / "b" / "x.md", "x")
    with pytest.raises(ValueError, match=r"folders: near: .*; far: "):
        Index(db).reindex([str(tmp_path / "b" / "x.md")])
    for i in range(4):
        (tmp_path / f"c{i}").mkdir()
        store.add_collection(db, tmp_path / f"c{i}")
    with pytest.raises(ValueError, match=r"folders: near: [^;]*(; [^;]*){4}; \.\.\.\. To index"):  # five, and more
        Index(db).reindex([str(tmp_path / "b" / "x.md")])


def test_a_file_while_no_collection_is_registered(home, tmp_path):
    write(tmp_path / "x.md", "x")
    with pytest.raises(ValueError, match=r"is in no collection: none is registered yet\. To index files there, add"):
        Index(store.connect(home)).reindex([str(tmp_path / "x.md")])


def test_an_unknown_collection_names_those_there_are(live):
    with pytest.raises(LookupError, match="no collection named 'nope'; the collections are vault, work"):
        live[0].reindex(["nope/a.md"])
    with pytest.raises(LookupError, match="no collection named 'nope'; the collections are vault, work"):
        live[0].search("x", collection="nope")


def test_reindex_refuses_files_behind_a_linked_folder(live, tmp_path, monkeypatch):
    # `sift index` doesn't follow links to folders, which may lead anywhere.
    index, vault = live
    write(tmp_path / "secret" / "a.md", "x")
    try:
        os.symlink(tmp_path / "secret", vault / "linked", target_is_directory=True)
    except OSError:  # Windows lets only administrators make symbolic links: pretend
        write(vault / "linked" / "a.md", "x")
        monkeypatch.setattr(indexing.os.path, "islink", lambda path: path.endswith("linked"))
    with pytest.raises(ValueError, match="linked/ is linked from elsewhere"):
        index.reindex(["vault/linked/a.md"])


def test_reindex_refuses_a_collection_whose_folder_is_gone(live):
    # An unplugged drive must not look like every file deleted.
    index, vault = live
    vault.rename(vault.with_name("moved"))
    with pytest.raises(ValueError, match="vault: folder not found"):
        index.reindex(["vault/01_People/Hans Müller.md"])


def test_reindex_refuses_names_sqlite_cannot_store(live):
    with pytest.raises(ValueError, match="unsupported file name"):
        live[0].reindex(["vault/r\udce9sum\udce9.md"])


def test_an_interrupted_reindex_changes_nothing(live, monkeypatch):
    index, vault = live
    write(vault / "a.md", "Alpha note.\n")
    (vault / "02_Organizations" / "Acme AG.md").unlink()

    def failing(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(indexing, "_store", failing)
    with pytest.raises(KeyboardInterrupt):
        index.reindex(["vault/02_Organizations/Acme AG.md", "vault/a.md"])
    assert "vault/02_Organizations/Acme AG.md" in ids(index.browse(collection="vault"))
    assert "vault/a.md" not in ids(index.browse(collection="vault"))


def test_an_interrupted_embedding_stores_no_vectors(live, monkeypatch):
    index, vault = live
    write(vault / "a.md", "Alpha note.\n")
    vectors = index.db.execute("SELECT count(*) FROM vectors").fetchone()

    def failing(db):
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "touch", failing)
    with pytest.raises(KeyboardInterrupt):
        index.reindex(["vault/a.md"])
    assert index.db.execute("SELECT count(*) FROM vectors").fetchone() == vectors


@pytest.mark.parametrize("paths, error", [([], "at least one"), ("vault/a.md", "a list"), ([1], "a list")])
def test_reindex_takes_a_list_of_paths(live, paths, error):
    with pytest.raises(ValueError, match=error):
        live[0].reindex(paths)


def test_reindex_takes_at_most_so_many_paths(live, monkeypatch):
    monkeypatch.setattr(indexing, "MOST_FILES", 2)
    with pytest.raises(ValueError, match="at most 2 files at a time"):
        live[0].reindex(["vault/a.md", "vault/b.md", "vault/c.md"])
