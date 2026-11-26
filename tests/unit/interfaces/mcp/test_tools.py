from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from sift.api.collections import AddCollection, Audience, CollectionStatus, Counted, Status, Unavailable
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, Problem, SiftError
from sift.api.jobs import ServedJob
from sift.api.wire import Json
from sift.app.collections import add
from sift.app.embedding import embed
from sift.interfaces.acts.acting import Acting
from sift.interfaces.mcp.offered import TOOLS
from sift.interfaces.mcp.reading import MOST_TEXT
from sift.interfaces.mcp.searches import MOST_QUERIES
from sift.interfaces.mcp.tools import Started, Tool
from sift.interfaces.server.handled import HANDLED, for_agents
from sift.settings import Settings
from tests.files import write
from tests.homes import collection, indexed, quiet
from tests.models import Hashed

type Object = dict[str, Json]


@dataclass(slots=True)
class Direct:
    acting: Acting
    asked: list[str] = field(default_factory=list[str])
    sent: list[Json] = field(default_factory=list[Json])
    finished: bool = True

    def read(self, name: str, data: Json) -> object:
        self.asked.append(name)
        self.sent.append(data)
        return for_agents(HANDLED)[name].run(self.acting, data)

    def job(self, name: str, data: Json) -> Started:
        self.asked.append(name)
        return Started(4, for_agents(HANDLED)[name].run(self.acting, data) if self.finished else None)

    def jobs(self) -> tuple[ServedJob, ...]:
        return (ServedJob(4, "reindex", running=True, seconds=2.5, watched=False, collection="open", done=1),)


@pytest.fixture
def direct(tmp_path: Path) -> Direct:
    home = tmp_path / "home"
    plan = "---\ncreated: 2026-01-02\n---\n# Plan\n\nNote: the plan of Kestrel names [[Secret]].\n"
    collection(tmp_path, home, "open", {"Plan.md": plan, "Other.md": "Note: an osprey over the lake.\n"})
    write(tmp_path / "hidden" / "Secret.md", "Note: Kestrel is the secret project, see [[Plan]].\n")
    add(home, AddCollection(str(tmp_path / "hidden"), agents=False))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    return Direct(Acting(Settings(home=home, user=None, cwd=tmp_path), quiet, float, Hashed, Audience.AGENTS))


def tool(name: str) -> Tool:
    return next(each for each in TOOLS if each.name == name)


def called(direct: Direct, name: str, arguments: Mapping[str, object]) -> Object:
    return cast("Object", tool(name).call(direct, arguments))


def ids(answer: Object) -> list[str]:
    hits = cast("list[Object]", answer["hits"])
    return [cast("str", hit["id"]) for hit in hits]


def test_five_tools_are_offered_and_only_reindex_changes_anything() -> None:
    assert [(each.name, each.read_only) for each in TOOLS] == [
        ("search", True),
        ("get", True),
        ("related", True),
        ("collections", True),
        ("reindex", False),
    ]


def test_every_parameter_of_every_tool_is_described() -> None:
    for each in TOOLS:
        properties = cast("dict[str, Object]", each.schema["properties"])
        assert [name for name, schema in properties.items() if not schema.get("description")] == [], each.name


def test_a_search_is_hybrid_unless_another_mode_is_asked_for(direct: Direct) -> None:
    for mode, act in (({}, "query"), ({"mode": "keyword"}, "search"), ({"mode": "semantic"}, "vsearch")):
        assert ids(called(direct, "search", {"query": "Kestrel", "k": 1, **mode})) == ["open/Plan.md"]
        assert direct.asked[-1] == act


def test_a_hit_says_where_it_is_and_what_it_holds(direct: Direct) -> None:
    (hit,) = cast("list[Object]", called(direct, "search", {"query": "Kestrel", "mode": "keyword"})["hits"])
    assert set(hit) == {"id", "title", "first_line", "last_line", "score", "text"}
    assert (hit["id"], hit["title"], hit["first_line"], hit["last_line"]) == ("open/Plan.md", "Plan", 4, 6)
    compact = called(direct, "search", {"query": "Kestrel", "mode": "keyword", "compact": True, "fields": ["*"]})
    (shown,) = cast("list[Object]", compact["hits"])
    assert ("text" in shown, shown["fields"]) == (False, {"created": "2026-01-02"})


def test_several_phrasings_are_searched_in_one_call(direct: Direct) -> None:
    answer = called(direct, "search", {"query": ["Kestrel", "osprey"], "mode": "keyword"})
    results = cast("list[Object]", answer["results"])
    assert [(each["query"], ids(each)) for each in results] == [
        ("Kestrel", ["open/Plan.md"]),
        ("osprey", ["open/Other.md"]),
    ]


@pytest.mark.parametrize(
    "arguments,wrong",
    [
        ({"query": "x", "mode": "fuzzy"}, "mode"),
        ({"query": []}, "query"),
        ({"query": ["x"] * (MOST_QUERIES + 1)}, "query"),
        ({"query": 3}, "query"),
        ({}, "query"),
        ({"query": "x", "mode": "keyword", "candidates": 5}, "candidates"),
        ({"query": "x", "k": 0}, "k"),
    ],
)
def test_a_wrong_search_is_refused_naming_its_field(direct: Direct, arguments: Object, wrong: str) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        called(direct, "search", arguments)
    assert caught.value.field == wrong


def test_a_search_says_when_more_results_wait(direct: Direct) -> None:
    answer = called(direct, "search", {"query": "Note", "mode": "keyword", "k": 1})
    assert (len(ids(answer)), answer["more"]) == (1, True)
    assert "more" not in called(direct, "search", {"query": "Note", "mode": "keyword", "k": 5})


def test_get_reads_documents_and_lines_and_says_what_it_could_not(direct: Direct) -> None:
    answer = called(
        direct,
        "get",
        {"documents": [{"id": "open/Plan.md", "first_line": 6, "last_line": 6}, {"id": "hidden/Secret.md"}]},
    )
    assert answer["documents"] == [
        {"id": "open/Plan.md", "title": "Plan", "text": "Note: the plan of Kestrel names [[Secret]].\n"}
    ]
    (refusal,) = cast("list[str]", answer["not_read"])
    assert refusal.startswith("hidden/Secret.md: no note or document 'hidden/Secret.md'; fix: ")


def test_get_cuts_a_text_at_a_line_where_one_call_is_full_and_says_how_to_get_the_rest(
    direct: Direct, tmp_path: Path
) -> None:
    for name in ("a", "b"):
        write(tmp_path / "open" / f"{name}.md", f"# {name}\n\n" + "word " * (MOST_TEXT // 8) + "\nthe end\n")
    indexed(direct.acting.settings.home)
    asked = ["open/a.md", {"id": "open/b.md"}, {"id": "open/b.md", "first_line": 3}, "open/Plan.md"]
    whole, cut, later, short = cast("list[Object]", called(direct, "get", {"documents": asked})["documents"])
    assert (set(whole), cast("str", whole["text"]).endswith("the end\n")) == ({"id", "title", "text"}, True)
    assert (set(short), cast("str", short["text"]).startswith("---\ncreated")) == ({"id", "title", "text"}, True)
    full = f"this answer's text is at its {MOST_TEXT:,} characters"
    assert cut == {
        "id": "open/b.md",
        "title": "b",
        "text": "# b\n\n",
        "first_line": 1,
        "last_line": 2,
        "lines": 4,
        "left_out": f"lines 3 to 4 are left out, as {full}: get them with first_line 3",
    }
    assert later == {
        "id": "open/b.md",
        "title": "b",
        "left_out": f"left out, as this answer's text is at its {MOST_TEXT:,} characters: get it in a call of its own",
    }
    rest = called(direct, "get", {"documents": [{"id": "open/b.md", "first_line": 3}]})
    (read,) = cast("list[Object]", rest["documents"])
    assert (set(read), cast("str", read["text"]).endswith("the end\n")) == ({"id", "title", "text"}, True)


def test_get_reads_a_document_longer_than_one_call_holds_in_calls_that_follow_each_other(
    direct: Direct, tmp_path: Path
) -> None:
    write(tmp_path / "open" / "long.md", "".join(f"line {n:05} of a long note\n" for n in range(1, 6_001)))
    indexed(direct.acting.settings.home)
    read, first = list[str](), 1
    while first:
        (one,) = cast(
            "list[Object]",
            called(direct, "get", {"documents": [{"id": "open/long.md", "first_line": first}]})["documents"],
        )
        read.append(cast("str", one["text"]))
        assert len(read[-1]) <= MOST_TEXT
        first = cast("int", one["last_line"]) + 1 if "left_out" in one else 0
    assert (len(read), "".join(read).count("\n"), read[0][:28]) == (2, 6_000, "line 00001 of a long note\nli")


def test_get_gives_the_start_of_a_line_longer_than_one_call_holds(direct: Direct, tmp_path: Path) -> None:
    write(tmp_path / "open" / "wide.txt", "first\n" + "word " * (MOST_TEXT // 4) + "\nlast\n")
    indexed(direct.acting.settings.home)
    (one,) = cast(
        "list[Object]", called(direct, "get", {"documents": [{"id": "open/wide.txt", "first_line": 2}]})["documents"]
    )
    assert (len(cast("str", one["text"])), one["first_line"], one["last_line"], one["lines"]) == (MOST_TEXT, 2, 2, 3)
    assert one["left_out"] == (
        f"line 2 goes on for 25,000 characters more than the {MOST_TEXT:,} an answer holds; they are left out, "
        "and the lines after it: get those with first_line 3"
    )


def test_get_refuses_a_document_that_is_neither_an_id_nor_an_object_saying_what_to_give(direct: Direct) -> None:
    with pytest.raises(InvalidRequestError) as refused:
        called(direct, "get", {"documents": [7]})
    assert (refused.value.field, refused.value.remedy) == (
        "documents[0]",
        'give each document as its id, or as {"id": "...", "first_line": 17, "last_line": 26} for some of its lines',
    )


def test_related_finds_documents_like_one_its_links_and_its_mentions(direct: Direct) -> None:
    assert ids(called(direct, "related", {"id": "open/Plan.md", "k": 1})) == ["open/Other.md"]
    assert direct.asked[-1] == "similar"
    links = called(direct, "related", {"id": "open/Plan.md", "kind": "links"})
    assert [(each["target"], each["via"]) for each in cast("list[Object]", links["out"])] == [("Secret", "unresolved")]
    assert "into" not in links
    backlinks = called(direct, "related", {"id": "open/Plan.md", "kind": "backlinks"})
    assert (backlinks["into"], "out" in backlinks) == ([], False)
    called(direct, "related", {"id": "open/Plan.md", "kind": "mentions"})
    assert direct.asked[-1] == "mentions"


def test_related_refuses_what_its_kind_does_not_take(direct: Direct) -> None:
    for arguments, wrong in (
        ({"id": "open/Plan.md", "kind": "links", "compact": True}, "compact"),
        ({"id": "open/Plan.md", "kind": "similar", "depth": 2}, "depth"),
        ({"id": "open/Plan.md", "kind": "friends"}, "kind"),
        ({"id": "open/Plan.md", "kind": "links", "direction": "in"}, "direction"),
    ):
        with pytest.raises(SiftError) as caught:
            called(direct, "related", arguments)
        assert caught.value.field == wrong


def test_collections_lists_what_an_agent_may_search_and_the_jobs_running(direct: Direct) -> None:
    answer = called(direct, "collections", {})
    (shown,) = cast("list[Object]", answer["collections"])
    assert {key: shown[key] for key in ("name", "documents", "passages", "embedded", "fields", "links")} == {
        "name": "open",
        "documents": 2,
        "passages": 2,
        "embedded": 2,
        "fields": ["created"],
        "links": 1,
    }
    assert answer["jobs"] == [
        {"job": 4, "command": "reindex", "running": True, "seconds": 2.5, "collection": "open", "done": 1}
    ]


def test_reindex_says_how_each_file_went_or_that_its_job_still_runs(direct: Direct, tmp_path: Path) -> None:
    write(tmp_path / "open" / "Plan.md", "Note: the plan of Osprey.\n")
    answer = called(direct, "reindex", {"files": ["open/Plan.md"]})
    assert answer == {"job": 4, "files": [{"id": "open/Plan.md", "state": "changed"}], "embedded": 1}
    direct.finished = False
    assert called(direct, "reindex", {"files": ["open/Other.md"]}) == {
        "job": 4,
        "running": "it runs on: collections lists it until it ends",
    }


def test_get_shows_the_front_matter_asked_for(direct: Direct) -> None:
    answer = called(direct, "get", {"documents": [{"id": "open/Plan.md"}], "fields": ["created"]})
    (document,) = cast("list[Object]", answer["documents"])
    assert document["fields"] == {"created": "2026-01-02"}


@dataclass(frozen=True, slots=True)
class Faked:
    status: Status

    def read(self, name: str, data: Json) -> object:
        assert (name, data) == ("status", {})
        return self.status

    def job(self, name: str, data: Json) -> Started:
        raise AssertionError((name, data))

    def jobs(self) -> tuple[ServedJob, ...]:
        return (ServedJob(2, "index", running=False, seconds=0.04, watched=True, collection="hidden"),)


def test_collections_says_which_are_inactive_unreadable_or_out_of_reach() -> None:
    counted = Counted(5, 9, 0, 2, ())
    gone = Unavailable(Problem.STORE_DAMAGED, "its index is damaged", "rebuild it")
    status = Status(
        "/home",
        (
            CollectionStatus("paused", "/p", "/s/p", counted, active=False),
            CollectionStatus("broken", "/b", "/s/b", gone),
        ),
    )
    answer = cast("Object", tool("collections").call(Faked(status), {}))
    assert answer == {
        "collections": [
            {
                "name": "paused",
                "folder": "/p",
                "active": False,
                "documents": 5,
                "passages": 9,
                "embedded": 0,
                "fields": [],
                "links": 0,
                "unresolved": 0,
                "unreadable": 2,
            },
            {"name": "broken", "folder": "/b", "problem": "its index is damaged; fix: rebuild it"},
        ],
        "jobs": [{"job": 2, "command": "index", "running": False, "seconds": 0.0}],
    }


def test_collections_names_at_most_200_fields_of_a_collection_and_counts_the_others() -> None:
    names = tuple(f"field {n}" for n in range(201))
    listed = Status(
        "/home",
        (
            CollectionStatus("few", "/f", "/s/f", Counted(1, 1, 0, 0, (), fields=names[:200])),
            CollectionStatus("many", "/m", "/s/m", Counted(1, 1, 0, 0, (), fields=names)),
        ),
    )
    answer = cast("Object", tool("collections").call(Faked(listed), {}))
    few, many = cast("list[Object]", answer["collections"])
    assert (few["fields"], "more_fields" in few) == (list(names[:200]), False)
    assert (many["fields"], many["more_fields"]) == (list(names[:200]), 1)


def test_a_refusal_names_what_the_tool_takes_for_its_kind_and_mode(direct: Direct) -> None:
    with pytest.raises(InvalidRequestError) as compact:
        called(direct, "related", {"id": "open/Plan.md", "kind": "links", "compact": True})
    with pytest.raises(InvalidRequestError) as fuzzy:
        called(direct, "search", {"query": "x", "mode": "fuzzy"})
    assert (compact.value.message, compact.value.remedy) == (
        "related of kind links takes no 'compact'",
        "leave out 'compact': it takes id, kind, depth, field, k",
    )
    assert fuzzy.value.message == "mode is 'fuzzy', not hybrid, keyword or semantic"


def test_several_queries_share_the_time_budget_one_call_has(direct: Direct) -> None:
    called(direct, "search", {"query": ["Kestrel", "osprey"], "mode": "keyword", "budget": 2.0})
    assert [cast("Object", each)["budget"] for each in direct.sent] == [1.0, 1.0]
    called(direct, "search", {"query": "Kestrel", "mode": "keyword", "budget": 2.0})
    assert cast("Object", direct.sent[-1])["budget"] == 2.0


@pytest.mark.parametrize(
    "arguments, said",
    [
        ({"query": ["a", "b"], "budget": -2}, "^-2 is outside"),
        ({"query": ["a"] * 20, "budget": 0.01}, r"0\.01 s shared by 20 queries"),
        ({"query": ["a", "b"], "k": 5_001}, r"10,000 hits"),
    ],
)
def test_a_budget_or_k_too_wrong_for_several_queries_is_refused_as_asked(
    direct: Direct, arguments: Object, said: str
) -> None:
    with pytest.raises(InvalidRequestError, match=said):
        called(direct, "search", arguments)
    assert direct.sent == []


def test_several_queries_may_ask_for_as_many_hits_in_all_as_one_search(direct: Direct) -> None:
    answer = called(direct, "search", {"query": ["Kestrel", "osprey"], "mode": "keyword", "k": 5_000})
    assert len(cast("list[Object]", answer["results"])) == 2
