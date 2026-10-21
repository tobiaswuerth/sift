"""Links between Markdown notes in the index: stored as written, resolved when asked, so they stay right as notes come
and go; links out of a note and back to it, and the targets no note answers to yet."""

import json
import os

import pytest

from sift import indexing, store
from sift.search import Index

from .conftest import keywords_agree, write

HANS, ANNA, ACME = (
    "vault/01_People/Hans Müller.md",
    "vault/01_People/Anna Keller.md",
    "vault/02_Organizations/Acme AG.md",
)
OFFER = "vault/03_Interactions/2026-10-14 Offerte.md"


def out(index, id, **options):
    return index.links(id, "out", **options)["out"]


def into(index, id, **options):
    return index.links(id, "in", **options)["in"]


def resolved(links):
    """Each link out as (target, where it leads): an id with how, "ambiguous" with the candidates, or "unresolved"."""
    return [
        (
            lk["target"],
            (lk["id"], lk["via"])
            if "id" in lk
            else ("ambiguous", lk["ambiguous"])
            if "ambiguous" in lk
            else "unresolved",
        )
        for lk in links
    ]


def note_vault(tmp_path, home, files, **fields):
    """A collection "v" of these notes, {path: text}, indexed."""
    for path, text in files.items():
        write(tmp_path / "v" / path, text)
    db = store.connect(home)
    store.add_collection(db, tmp_path / "v", "v", fields or None)
    indexing.index(db)
    return db, tmp_path / "v"


# --- the vault ---------------------------------------------------------------


def test_a_note_in_the_vault(linked):
    hans = Index(linked[0]).document(HANS, fields=["*"])
    assert hans["title"] == "Hans Müller"  # not "Summary": the vault takes titles from file names
    assert hans["fields"]["aliases"] == ["Hans Mueller", "H. Müller"]


def test_links_out_of_a_note(linked):
    index = Index(linked[0])
    assert out(index, OFFER) == [
        {"target": "Hans Mueller", "line": 3, "kind": "wikilink", "id": HANS, "title": "Hans Müller", "via": "alias"},
        {"target": "Wartungsvertrag 2027", "line": 3, "kind": "wikilink", "unresolved": True},
    ]  # and neither code example
    assert out(index, HANS) == [
        {
            "target": "Acme AG",
            "line": 3,
            "kind": "wikilink",
            "field": "org",
            "id": ACME,
            "title": "Acme AG",
            "via": "name",
        },
        {"target": "Peter Frei", "line": 4, "kind": "wikilink", "field": "reports_to", "unresolved": True},
    ]
    assert resolved(out(index, ACME)) == [("01_People/Hans Müller", (HANS, "path"))]
    assert out(index, ACME)[0]["fragment"] == "Kontakt"


def test_links_into_a_note(linked):
    index = Index(linked[0])
    assert into(index, ACME) == [
        {"id": HANS, "title": "Hans Müller", "links": [{"line": 3, "field": "org", "text": 'org: "[[Acme AG]]"'}]}
    ]
    assert into(index, HANS) == [
        {
            "id": ANNA,
            "title": "Anna Keller",
            "links": [{"line": 2, "field": "colleague_of", "text": 'colleague_of: ["[[Hans Müller|Hans]]"]'}],
        },
        {
            "id": ACME,
            "title": "Acme AG",
            "links": [{"line": 4, "text": "Key contact: [[01_People/Hans Müller#Kontakt]]"}],
        },
        {
            "id": OFFER,
            "title": "2026-10-14 Offerte",
            "links": [{"line": 3, "text": "Call with [[Hans Mueller]] about [[Wartungsvertrag 2027]]."}],
        },
    ]


def test_a_long_line_shows_around_its_link(tmp_path, home):
    # A paragraph is one line in Markdown: all of it, for each link, would fill an agent's context.
    before, after = "words " * 200, " more" * 200
    db, _ = note_vault(tmp_path, home, {"Hans.md": "", "a.md": f"{before}[[Hans]]{after}", "b.md": f"[[Hans]]{after}"})
    (a,), (b,) = (entry["links"] for entry in into(Index(db), "v/Hans.md"))
    assert (a["text"][0], a["text"][-1], len(a["text"])) == ("…", "…", 302)
    assert "[[Hans]]" in a["text"]
    assert (b["text"][:13], b["text"][-1], len(b["text"])) == ("[[Hans]] more", "…", 301)


def test_a_line_shows_once_however_often_it_links(tmp_path, home):
    db, _ = note_vault(tmp_path, home, {"Hans.md": "", "a.md": "[[Hans]] and [[Hans|him]]\n\n[[hans]]"})
    assert into(Index(db), "v/Hans.md")[0]["links"] == [
        {"line": 1, "text": "[[Hans]] and [[Hans|him]]"},
        {"line": 3, "text": "[[hans]]"},
    ]


def test_links_both_ways(linked):
    found = Index(linked[0]).links(ACME, "both")
    assert (found["id"], found["title"]) == (ACME, "Acme AG")
    assert [lk["id"] for lk in found["out"]] == [HANS]
    assert [lk["id"] for lk in found["in"]] == [HANS]
    assert set(Index(linked[0]).links(ACME)) == {"id", "title", "out"}  # out by default


def test_links_through_one_field(linked):
    index = Index(linked[0])
    assert [lk["id"] for lk in into(index, HANS, field="colleague_of")] == [ANNA]
    assert into(index, HANS, field="org") == []
    assert resolved(out(index, HANS, field="org")) == [("Acme AG", (ACME, "name"))]


def test_links_two_steps_away(linked):
    index = Index(linked[0])
    # Anna links Hans, who links Acme AG and Peter Frei, who has no note yet.
    assert out(index, ANNA, depth=2) == [
        {
            "target": "Hans Müller",
            "line": 2,
            "kind": "wikilink",
            "field": "colleague_of",
            "shown": "Hans",
            "id": HANS,
            "title": "Hans Müller",
            "via": "name",
        },
        {
            "target": "Acme AG",
            "line": 3,
            "kind": "wikilink",
            "field": "org",
            "id": ACME,
            "title": "Acme AG",
            "via": "name",
            "through": HANS,
        },
    ]
    # Into Acme AG come Hans and, through Hans, Anna and the offer; Acme AG itself, which links Hans, is left out.
    found = into(index, ACME, depth=2)
    assert [(lk["id"], lk.get("through")) for lk in found] == [(HANS, None), (ANNA, HANS), (OFFER, HANS)]
    assert found[1]["links"] == [{"line": 2, "field": "colleague_of", "text": 'colleague_of: ["[[Hans Müller|Hans]]"]'}]


def test_at_most_k_links_each_way(linked):
    index = Index(linked[0])
    assert len(into(index, HANS, k=2)) == 2
    assert len(out(index, HANS, k=1)) == 1


@pytest.mark.parametrize(
    "options, error",
    [
        ({"direction": "up"}, "direction must be"),
        ({"depth": 3}, "depth must be 1 or 2"),
        ({"depth": "2"}, "depth must be 1 or 2"),
        ({"field": 1}, "field must be"),
        ({"k": 0}, "k must be"),
    ],
)
def test_links_reject_bad_options(linked, options, error):
    with pytest.raises(ValueError, match=error):
        Index(linked[0]).links(HANS, **options)


def test_links_of_unknown_documents(linked):
    with pytest.raises(LookupError, match=r"unknown document 'vault/Nobody\.md'"):
        Index(linked[0]).links("vault/Nobody.md")
    with pytest.raises(LookupError, match="unknown document 'Wartungsvertrag 2027'"):
        Index(linked[0]).links("Wartungsvertrag 2027")  # linked, but not written yet


def test_a_note_named_as_links_name_it(linked, tmp_path):
    db = linked[0]
    index = Index(db)
    assert index.links("Acme AG", "in")["id"] == ACME
    assert index.links("[[Hans Mueller]]")["id"] == HANS  # by its alias
    assert [h["id"] for h in index.mentions("Hans Müller")] == ["work/contract.md"]
    write(tmp_path / "other" / "Acme AG.md", "Another Acme.\n")
    store.add_collection(db, tmp_path / "other", "other")
    indexing.index(db)
    with pytest.raises(LookupError, match=r"'Acme AG' matches other/Acme AG\.md, vault/02_Organizations/Acme AG\.md"):
        index.links("Acme AG")


def test_unresolved_targets(linked):
    index = Index(linked[0])
    assert index.unresolved("vault") == [
        {"target": "Peter Frei", "docs": 1, "ids": [HANS]},
        {"target": "Wartungsvertrag 2027", "docs": 1, "ids": [OFFER]},
    ]
    assert index.unresolved("work") == []
    assert index.unresolved() == index.unresolved("vault")
    with pytest.raises(LookupError, match="no collection named 'nope'"):
        index.unresolved("nope")


def test_unresolved_targets_most_linked_first(tmp_path, home):
    db, _ = note_vault(
        tmp_path,
        home,
        {
            "a.md": "[[Missing]] [[Missing|again]] [[Other]]",
            "b.md": "[[missing.md]] and [[Elsewhere/Missing]] and [this](Missing.md)",
            "c.md": "[[Missing]]",
        },
    )
    assert Index(db).unresolved(k=2) == [
        {"target": "Missing", "docs": 3, "ids": ["v/a.md", "v/b.md", "v/c.md"]},
        {"target": "Elsewhere/Missing", "docs": 1, "ids": ["v/b.md"]},
    ]


def test_a_path_to_nowhere_is_unresolved_though_its_name_is_known(tmp_path, home):
    db, _ = note_vault(tmp_path, home, {"People/Hans.md": "", "a.md": "[[Other/Hans]] [x](Nope/Hans.md) [[Hans]]"})
    assert Index(db).unresolved() == [
        {"target": "Nope/Hans.md", "docs": 1, "ids": ["v/a.md"]},
        {"target": "Other/Hans", "docs": 1, "ids": ["v/a.md"]},
    ]


def test_status_counts_links(linked):
    collections = {c["name"]: c for c in Index(linked[0]).status()["collections"]}
    assert (collections["vault"]["links"], collections["vault"]["unresolved"]) == (6, 2)
    assert (collections["work"]["links"], collections["work"]["unresolved"]) == (0, 0)


# --- how a link resolves -----------------------------------------------------


def test_a_link_resolves_as_obsidian_resolves_it(tmp_path, home):
    db, _ = note_vault(
        tmp_path,
        home,
        {
            "People/Hans.md": "---\naliases: [Hansi, Johannes]\n---\n",
            "Archive/People/Hans.md": "",
            "Projects/Hans.md": "",
            "Projects/Plan.md": "[[Hans]] [[People/Hans]] [[Archive/People/Hans]] [[Johannes]] [[Plan]]",
            "Projects/Sub/Note.md": "[[Hans]] [[archive/people/hans.md]] [[Nope/Hans]] ![[notes.txt]] [[notes]]",
            "Other.md": "[[Hans]] [[Hansi]] [[Dup]]",
            "A/Dup.md": "",
            "B/Dup.md": "",
            "C/X.md": "---\naliases: Dup\n---\n",
            "notes.txt": "plain text",
        },
    )
    index = Index(db)
    hans = ["v/Archive/People/Hans.md", "v/People/Hans.md", "v/Projects/Hans.md"]
    assert resolved(out(index, "v/Projects/Plan.md")) == [
        ("Hans", ("v/Projects/Hans.md", "name")),  # of several of that name, the one in the note's folder
        ("People/Hans", ("v/People/Hans.md", "path")),  # a path from the vault's root first ...
        ("Archive/People/Hans", ("v/Archive/People/Hans.md", "path")),
        ("Johannes", ("v/People/Hans.md", "alias")),
        ("Plan", ("v/Projects/Plan.md", "name")),  # a link to itself
    ]
    assert resolved(out(index, "v/Projects/Sub/Note.md")) == [
        ("Hans", ("ambiguous", hans)),
        ("archive/people/hans.md", ("v/Archive/People/Hans.md", "path")),
        ("Nope/Hans", "unresolved"),
        ("notes.txt", ("v/notes.txt", "name")),
        ("notes", "unresolved"),  # a text file's name has its extension
    ]
    assert resolved(out(index, "v/Other.md")) == [
        ("Hans", ("ambiguous", hans)),
        ("Hansi", ("v/People/Hans.md", "alias")),
        ("Dup", ("ambiguous", ["v/A/Dup.md", "v/B/Dup.md"])),  # a name before an alias
    ]


def test_a_partial_path_must_be_unique(tmp_path, home):
    db, _ = note_vault(tmp_path, home, {"x/People/Hans.md": "", "y/People/Hans.md": "", "n.md": "[[People/Hans]]"})
    assert resolved(out(Index(db), "v/n.md")) == [
        ("People/Hans", ("ambiguous", ["v/x/People/Hans.md", "v/y/People/Hans.md"]))
    ]


def test_an_alias_of_several_notes_is_ambiguous(tmp_path, home):
    db, _ = note_vault(
        tmp_path, home, {"a.md": "---\nalias: X\n---\n", "b.md": "---\naliases: [X]\n---\n", "n.md": "[[x]]"}
    )
    assert resolved(out(Index(db), "v/n.md")) == [("x", ("ambiguous", ["v/a.md", "v/b.md"]))]


def test_ambiguous_links_name_a_few_candidates(tmp_path, home):
    db, _ = note_vault(tmp_path, home, {**{f"{i}/Same.md": "" for i in range(7)}, "n.md": "[[Same]]"})
    assert len(out(Index(db), "v/n.md")[0]["ambiguous"]) == 5


def test_markdown_links_are_relative_to_their_note(tmp_path, home):
    db, _ = note_vault(
        tmp_path,
        home,
        {
            "People/Hans.md": "",
            "Notes/Hans.md": "",
            "Notes/Plan.md": (
                "[a](../People/Hans.md) [b](Hans.md) [c](./Hans.md) [d](People/Hans.md) [e](<../People/Hans.md>) "
                "[f](../../outside.md) [g](./Missing.md) [h](Plan%20B.md)"
            ),
            "Notes/Plan B.md": "",
        },
    )
    assert resolved(out(Index(db), "v/Notes/Plan.md")) == [
        ("../People/Hans.md", ("v/People/Hans.md", "path")),
        ("Hans.md", ("v/Notes/Hans.md", "path")),
        ("./Hans.md", ("v/Notes/Hans.md", "path")),
        ("People/Hans.md", ("v/People/Hans.md", "path")),  # else from the vault's root, as Obsidian writes them too
        ("../People/Hans.md", ("v/People/Hans.md", "path")),
        ("../../outside.md", "unresolved"),
        ("./Missing.md", "unresolved"),
        ("Plan B.md", ("v/Notes/Plan B.md", "path")),
    ]


def test_names_compare_composed_and_in_any_case(tmp_path, home):
    # macOS hands out file names decomposed: ü as u and a combining diaeresis.
    db, _ = note_vault(tmp_path, home, {"Hans Müller.md": "", "n.md": "[[hans müller]] [[HANS MÜLLER.md]]"})
    assert [lk["via"] for lk in out(Index(db), "v/n.md")] == ["name", "name"]


def test_links_stay_in_their_collection(tmp_path, home, linked):
    db = linked[0]
    write(tmp_path / "other" / "Peter Frei.md", "A note of another collection.\n")
    store.add_collection(db, tmp_path / "other", "other")
    indexing.index(db)
    assert resolved(out(Index(db), HANS))[1] == ("Peter Frei", "unresolved")


def test_links_follow_notes_as_they_come_and_go(linked):
    # Without the linking notes being read again.
    db, vault, _ = linked
    offer = db.execute("SELECT id FROM docs WHERE path LIKE '03_Interactions/%'").fetchone()
    write(vault / "06_Topics" / "Wartungsvertrag 2027.md", "# Maintenance contract 2027\n")
    indexing.index(db)
    assert resolved(out(Index(db), OFFER))[1] == (
        "Wartungsvertrag 2027",
        ("vault/06_Topics/Wartungsvertrag 2027.md", "name"),
    )
    (vault / "02_Organizations" / "Acme AG.md").rename(vault / "02_Organizations" / "Acme Group.md")
    indexing.index(db)
    assert resolved(out(Index(db), HANS))[0] == ("Acme AG", "unresolved")
    assert db.execute("SELECT id FROM docs WHERE path LIKE '03_Interactions/%'").fetchone() == offer


# --- storing links -----------------------------------------------------------


def rows(db, table):
    return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_links_and_names_go_with_their_documents(linked):
    db, vault, _ = linked
    assert (rows(db, "links"), rows(db, "names")) == (6, 7)  # names: the 5 files, and Hans's 2 aliases
    links_before = db.execute("SELECT id FROM links ORDER BY id").fetchall()
    os.utime(vault / "01_People" / "Hans Müller.md", ns=(10**18, 10**18))  # touched, unchanged: its rows stay
    indexing.index(db)
    assert db.execute("SELECT id FROM links ORDER BY id").fetchall() == links_before
    (vault / "01_People" / "Hans Müller.md").unlink()
    indexing.index(db)
    assert (rows(db, "links"), rows(db, "names")) == (4, 4)
    store.remove_collection(db, "vault")
    assert (rows(db, "links"), rows(db, "names")) == (0, 1)


def test_records_have_no_names(tmp_path, home):
    db, _ = note_vault(tmp_path, home, {"r.jsonl": json.dumps({"id": "Hans", "text": "[[Hans]]"}) + "\n"})
    assert (rows(db, "links"), rows(db, "names")) == (0, 0)


# --- an index of sift 1.0 ----------------------------------------------------


def as_made_by_sift_1_0(db):
    """Turn an index into one sift 1.0 made: no links or names, and front-matter lists of unquoted items as text."""
    db.executescript("DROP TABLE links; DROP TABLE names;")
    (doc,) = db.execute("SELECT id FROM docs WHERE path = '01_People/Hans Müller.md'").fetchone()
    store.keywords(db, [doc], add=False)
    meta = json.loads(db.execute("SELECT meta FROM docs WHERE id = ?", (doc,)).fetchone()[0])
    meta["aliases"] = "[Hans Mueller, H. Müller]"
    db.execute("UPDATE docs SET meta = ? WHERE id = ?", (json.dumps(meta, ensure_ascii=False), doc))
    store.keywords(db, [doc])
    db.execute("PRAGMA user_version = 3")
    db.commit()


def test_an_index_of_sift_1_0_gets_its_links(linked, fake_models, home, capsys, monkeypatch):
    from sift import embed, readers

    db, vault, _ = linked
    write(vault / "page.html", "<html><body><h1>Price list</h1><p>Rates for 2027.</p></body></html>")
    write(vault / "Prices.md", "See ![[page.html]].\n")
    indexing.index(db)
    embed.embed_pending(db)
    before = Index(db).search("Acme")
    vectors = rows(db, "vectors")
    as_made_by_sift_1_0(db)
    db.close()
    monkeypatch.setattr(readers, "load", lambda *args: pytest.fail("a file was read again"))
    db = store.connect(home)
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    assert "reading links" in capsys.readouterr().err
    assert (rows(db, "links"), rows(db, "names"), rows(db, "vectors")) == (7, 9, vectors)  # nothing embedded again
    assert Index(db).document(HANS, fields=["aliases"])["fields"] == {"aliases": ["Hans Mueller", "H. Müller"]}
    assert keywords_agree(db)
    assert resolved(out(Index(db), OFFER))[0] == ("Hans Mueller", (HANS, "alias"))
    assert resolved(out(Index(db), "vault/Prices.md")) == [("page.html", ("vault/page.html", "name"))]
    assert Index(db).search("Acme") == before


def test_an_index_of_sift_1_0_gets_passages_for_notes_of_front_matter_alone(linked, home):
    db, vault, _ = linked
    write(vault / "01_People" / "Peter Frei.md", '---\norg: "[[Acme AG]]"\nemail: peter@acme.example\n---\n')
    indexing.index(db)
    (doc,) = db.execute("SELECT id FROM docs WHERE path = '01_People/Peter Frei.md'").fetchone()
    store.keywords(db, [doc], add=False)
    db.execute("DELETE FROM chunks WHERE doc = ?", (doc,))  # sift 1.0 gave it no passage
    as_made_by_sift_1_0(db)
    db.close()
    db = store.connect(home)
    assert "vault/01_People/Peter Frei.md" in [h["id"] for h in Index(db).search("Peter Frei")]
    assert keywords_agree(db)


def test_an_interrupted_upgrade_leaves_the_index_as_it_was(linked, home, monkeypatch):
    db = linked[0]
    as_made_by_sift_1_0(db)
    db.close()

    def interrupted(db):
        db.execute("INSERT INTO links(doc, collection, line, kind, target, key) VALUES(1, 1, 1, 'wikilink', 'x', 'x')")
        raise KeyboardInterrupt

    real = indexing.add_links
    monkeypatch.setattr(indexing, "add_links", interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.connect(home)
    monkeypatch.setattr(indexing, "add_links", real)
    db = store.connect(home)  # tries again, from the start
    assert rows(db, "links") == 6


def test_an_index_takes_its_links_once(linked, monkeypatch):
    # Two sifts that start at once both see an index of 1.0; the second must find the work done once it has the
    # database, and not add every link a second time.
    db = linked[0]
    monkeypatch.setattr(indexing, "add_links", lambda db: pytest.fail("links added twice"))
    store._add_links(db)  # what the second runs, after the first has finished
    assert rows(db, "links") == 6


# --- searching by links ------------------------------------------------------


def ids(hits):
    return [h["id"] for h in hits]


def test_search_the_documents_linking_a_note(linked):
    index = Index(linked[0])
    assert ids(index.search("Hans", links_to=["Acme AG"])) == [HANS]  # not Anna Keller, who links Hans
    assert ids(index.search("Hans", links_to=["[[Acme AG]]"])) == [HANS]
    assert ids(index.search("Hans", links_to=[ACME], collection="vault")) == [HANS]
    assert ids(index.browse(links_to=["Hans Müller"])) == [ANNA, ACME, OFFER]
    assert ids(index.browse(links_to=["Acme AG", "Wartungsvertrag 2027"])) == [HANS, OFFER]  # any; one not written yet


def test_search_the_documents_a_note_links(linked):
    index = Index(linked[0])
    assert ids(index.browse(linked_from=["Hans Müller"])) == [ACME]  # and Peter Frei, who has no note
    assert ids(index.search("customer", linked_from="vault/01_People/Hans Müller.md")) == [ACME]
    assert ids(index.browse(links_to=["Acme AG"], linked_from=["Anna Keller"])) == [HANS]  # both must hold
    assert index.browse(links_to=["Hans Müller"], linked_from=["Anna Keller"]) == []


def test_every_search_takes_links(linked, fake_models):
    from sift import embed

    index = Index(linked[0])
    embed.embed_pending(linked[0])
    assert ids(index.vsearch("customer", k=1, links_to=["Hans Müller"])) == [ACME]
    assert ids(index.query("customer since 2019", linked_from=["Hans Müller"])) == [ACME]
    assert ids(index.similar(OFFER, links_to=["Acme AG"])) == [HANS]


@pytest.mark.parametrize(
    "notes, error",
    [
        (["Nobody"], "no note 'Nobody' in the index, and no link to one"),
        (["vault/Nobody.md"], "no note 'vault/Nobody.md'"),
        ([], "links_to needs at least one note"),
        ([1], "links_to takes ids of documents or link targets"),
        ([" "], "links_to takes ids of documents or link targets"),
    ],
)
def test_links_to_unknown_notes(linked, notes, error):
    with pytest.raises((LookupError, ValueError), match=error):
        Index(linked[0]).search("Hans", links_to=notes)


def test_linked_from_an_unknown_or_ambiguous_note(tmp_path, home):
    db, _ = note_vault(tmp_path, home, {"a/Same.md": "", "b/Same.md": "", "n.md": "[[Missing]]"})
    with pytest.raises(LookupError, match="no note 'Missing' in the index"):
        Index(db).browse(linked_from=["Missing"])  # a note not written yet links nowhere
    with pytest.raises(LookupError, match=r"'v/Same\.md' matches v/a/Same\.md, v/b/Same\.md; give the whole id"):
        Index(db).browse(linked_from=["v/Same.md"])


def test_a_note_named_in_several_places_must_be_chosen(linked, tmp_path):
    db = linked[0]
    write(tmp_path / "other" / "Acme AG.md", "Another Acme.\n")
    store.add_collection(db, tmp_path / "other", "other")
    indexing.index(db)
    index = Index(db)
    with pytest.raises(LookupError, match=r"'Acme AG' matches other/Acme AG.md, vault/02_Organizations/Acme AG.md"):
        index.search("Hans", links_to=["Acme AG"])
    assert ids(index.search("Hans", links_to=["Acme AG"], collection="vault")) == [HANS]
    write(tmp_path / "other" / "Sub" / "Acme AG.md", "A third.\n")
    indexing.index(db)
    with pytest.raises(LookupError, match=r"'Acme AG' matches other/Acme AG.md, other/Sub/Acme AG.md"):
        index.search("Hans", links_to=["Acme AG"], collection="other")


def test_a_wikilink_filter_matches_the_note_however_the_link_is_written(tmp_path, home):
    db, _ = note_vault(
        tmp_path,
        home,
        {
            "Orgs/Acme AG.md": "---\naliases: [Acme]\n---\n",
            "a.md": '---\norg: "[[Acme AG]]"\n---\n',
            "b.md": '---\norg: "[[Acme AG|Acme]]"\n---\n',
            "c.md": "---\norg: [[acme ag]]\n---\n",
            "d.md": '---\norg: ["[[Orgs/Acme AG]]", "[[Beta]]"]\n---\n',
            "e.md": '---\norg: "[[Acme#History]]"\n---\n',  # by its alias
            "f.md": '---\norg: "[[Beta]]"\n---\n',
            "g.md": '---\nclient: "[[Acme AG]]"\n---\n',  # another field
            "h.md": "---\norg: Acme AG\n---\n",  # text, no link
            "r.jsonl": json.dumps({"id": "r1", "org": "[[Acme AG]]", "text": "A record."}) + "\n",
        },
    )
    index = Index(db)
    acme = ["v/a.md", "v/b.md", "v/c.md", "v/d.md", "v/e.md"]
    assert ids(index.browse(filters={"org": "[[Acme AG|Acme]]"})) == acme
    assert ids(index.browse(filters={"org": "[[Acme AG]]"})) == [*acme, "v/r/r1"]  # a record by its value, as before
    assert ids(index.browse(filters={"org": "[[Orgs/Acme AG]]"})) == acme
    assert ids(index.browse(filters={"org": {"not": "[[Acme AG]]"}})) == [
        "v/Orgs/Acme AG.md",
        "v/f.md",
        "v/g.md",
        "v/h.md",
    ]
    assert ids(index.browse(filters={"org": "Acme AG"})) == ["v/h.md"]  # a value that is no link compares exactly
    # a note not written yet: the links to it, however written
    assert ids(index.browse(filters={"org": ["[[beta.md]]", "Acme AG"]})) == ["v/d.md", "v/f.md", "v/h.md"]
    assert ids(index.browse(filters={"client": "[[Acme]]"})) == ["v/g.md"]


# --- mentions without links --------------------------------------------------


def test_documents_that_mention_a_note_without_linking_it(linked):
    index = Index(linked[0])
    assert ids(index.mentions(HANS)) == ["work/contract.md"]  # Anna, Acme AG and the offer link him
    assert ids(index.mentions(ACME)) == ["work/contract.md"]  # Hans names Acme AG, but links it too
    hit = index.mentions(HANS)[0]
    assert set(hit) == {"id", "title", "lines", "score", "text"}
    assert "Hans Müller" in hit["text"]
    assert index.mentions(HANS, collection="vault") == []
    assert set(index.mentions(HANS, compact=True, fields=["*"])[0]) == {"id", "title", "lines", "score", "fields"}


def test_mentions_by_every_name_of_a_note(tmp_path, home):
    db, _ = note_vault(
        tmp_path,
        home,
        {
            "Jürg Bühler.md": '---\ntitle: "Dr. J. Bühler"\naliases: [JB Consulting]\nemails: [jb@example.com]\n---\n',
            "a.md": "Met Juerg Buehler today.",  # as written without umlauts
            "b.md": "Met Jurg Buhler today.",  # the keyword index ignores diacritics
            "c.md": "Invoice from JB Consulting.",
            "d.md": "Letter to Dr. J. Bühler.",
            "e.md": "Mail from jb@example.com.",
            "f.md": "Nothing about anyone.",
            "AI.md": "",
        },
    )
    index = Index(db)
    assert sorted(ids(index.mentions("v/Jürg Bühler.md"))) == ["v/a.md", "v/b.md", "v/c.md", "v/d.md"]
    assert "v/e.md" in ids(index.mentions("v/Jürg Bühler.md", also=["emails"]))
    assert "v/e.md" in ids(index.mentions("v/Jürg Bühler.md", also="emails"))
    assert index.mentions("v/AI.md") == []  # too short a name: it would be everywhere


@pytest.mark.parametrize("also", [[1], [""], 5])
def test_mentions_takes_field_names(linked, also):
    with pytest.raises(ValueError, match="also must name front-matter fields"):
        Index(linked[0]).mentions(HANS, also=also)
