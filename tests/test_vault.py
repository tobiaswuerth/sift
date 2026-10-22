"""An Obsidian vault of linked notes and a folder of other documents, worked with as an agent works, through MCP: the
acceptance test of sift's links."""

import pytest

from sift import embed, indexing, store
from sift.search import Index
from sift.server import build

from .conftest import write
from .test_server import call, data

HANS, ANNA, ACME = (
    "vault/01_People/Hans Müller.md",
    "vault/01_People/Anna Keller.md",
    "vault/02_Organizations/Acme AG.md",
)
OFFER, TOPIC = "vault/03_Interactions/2026-10-14 Offerte.md", "vault/06_Topics/Wartungsvertrag 2027.md"


@pytest.fixture
def agent(linked, fake_models):
    """What an agent reaches: the MCP tools of a server over the vault and the work folder, embedded and searched."""
    db, vault, _ = linked
    embed.embed_pending(db)
    mcp = build(Index(db))[0]

    def tool(name, **args):
        return data(call(mcp, name, args))

    tool("vsearch", query="anything")  # the vectors are in memory, as in a running server
    return tool, vault


def ids(found):
    return [item["id"] for item in found]


def test_an_agent_works_with_a_vault(agent):
    tool, vault = agent
    hans = tool("get", id=HANS, fields=["aliases"])
    assert (hans["title"], hans["fields"]["aliases"]) == ("Hans Müller", ["Hans Mueller", "H. Müller"])

    assert tool("links", id=ACME, direction="in")["in"][0]["links"][0]["field"] == "org"
    assert ids(tool("links", id=ACME, direction="in")["in"]) == [HANS]

    out = tool("links", id=OFFER)["out"]
    assert [(lk["target"], lk.get("id"), lk.get("via"), lk.get("unresolved")) for lk in out] == [
        ("Hans Mueller", HANS, "alias", None),
        ("Wartungsvertrag 2027", None, None, True),
    ]  # and neither code example

    into = tool("links", id=HANS, direction="in")["in"]
    assert ids(into) == [ANNA, ACME, OFFER]
    assert into[0]["links"][0]["field"] == "colleague_of"
    assert into[1]["links"][0]["line"] == 4  # Acme AG links Hans in its text

    assert [t["target"] for t in tool("unresolved", collection="vault")] == ["Peter Frei", "Wartungsvertrag 2027"]

    assert ids(tool("search", query="Summary", collection="vault", filters={"org": "[[Acme AG|Acme]]"})) == [HANS]
    assert ids(tool("search", query="Hans", links_to=["Acme AG"])) == [HANS]  # not Anna Keller

    assert ids(tool("mentions", id=HANS)) == ["work/contract.md"]

    write(vault / "06_Topics" / "Wartungsvertrag 2027.md", "## Summary\n\nPress line maintenance from 2027.\n")
    assert tool("reindex", paths=[TOPIC]) == [{"id": TOPIC, "status": "added", "chunks": 1, "embedded": 1}]
    assert tool("links", id=OFFER)["out"][1]["id"] == TOPIC  # without the offer being read again
    assert ids(tool("search", query="press line")) == [TOPIC]
    assert ids(tool("vsearch", query="press line maintenance", k=1)) == [TOPIC]

    (vault / "02_Organizations" / "Acme AG.md").unlink()
    assert tool("reindex", paths=[ACME]) == [{"id": ACME, "status": "removed"}]
    assert tool("links", id=HANS)["out"][0] == {
        "target": "Acme AG",
        "line": 3,
        "kind": "wikilink",
        "field": "org",
        "unresolved": True,
    }


def test_an_index_of_sift_1_0_opens_with_its_links(linked, home, capsys):
    # sift 1.0 kept no links: its index gains them on first use, from the text it holds, and searches as before.
    db = linked[0]
    before = Index(db).search("Acme")
    db.executescript("DROP TABLE links; DROP TABLE names; PRAGMA user_version = 3;")
    db.close()
    db = store.connect(home)
    assert "reading links" in capsys.readouterr().err
    assert Index(db).search("Acme") == before
    assert ids(Index(db).links(HANS, "in")["in"]) == [ANNA, ACME, OFFER]
    assert {c["name"]: c["links"] for c in Index(db).status()["collections"]} == {"vault": 6, "work": 0}
    indexing.index(db)  # and nothing is left to do
    assert "0 added, 0 changed, 0 removed" in capsys.readouterr().err
