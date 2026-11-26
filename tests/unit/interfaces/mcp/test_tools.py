from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from sift.api.collections import CollectionStatus, Counted, Status, Unavailable, UnreadableFile
from sift.api.errors import InvalidRequestError, Problem, SiftError
from sift.api.jobs import ServedJob
from sift.api.links import Target, Unresolved
from sift.api.wire import Json
from sift.interfaces.mcp.offered import TOOLS
from sift.interfaces.mcp.tools import Started
from tests.agent_tools import Direct, Object, called, ids, tool
from tests.files import write


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


def test_related_finds_documents_like_one_its_links_and_its_mentions(direct: Direct) -> None:
    assert ids(called(direct, "related", {"id": "open/Plan.md", "k": 1})) == ["open/Other.md"]
    assert direct.asked[-1] == "similar"
    links = called(direct, "related", {"id": "open/Plan.md", "kind": "links"})
    assert [(each["target"], each["via"]) for each in cast("list[Object]", links["out"])] == [("Secret", "unresolved")]
    assert "into" not in links
    assert set(cast("list[Object]", links["out"])[0]) == {"target", "via", "line"}
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


def test_reindex_says_how_each_file_went_or_that_its_job_still_runs(direct: Direct, tmp_path: Path) -> None:
    write(tmp_path / "open" / "Plan.md", "Note: the plan of Osprey.\n")
    answer = called(direct, "reindex", {"files": ["open/Plan.md"]})
    assert answer == {"job": 4, "files": [{"id": "open/Plan.md", "state": "changed"}], "embedded": 1}
    direct.finished = False
    assert called(direct, "reindex", {"files": ["open/Other.md"]}) == {
        "job": 4,
        "running": True,
        "note": "it runs on: collections lists it until it ends",
    }


def test_collections_lists_the_link_targets_no_note_answers_to_when_asked(direct: Direct) -> None:
    answer = called(direct, "collections", {"unresolved": True, "names": ["open"]})
    assert answer["unresolved_targets"] == [{"target": "Secret", "documents": 1, "ids": ["open/Plan.md"]}]
    assert direct.sent == [{"names": ["open"]}, {"collections": ["open"]}]
    assert "unresolved_targets" not in called(direct, "collections", {})
    with pytest.raises(InvalidRequestError) as refused:
        called(direct, "collections", {"unresolved": "yes"})
    assert refused.value.field == "unresolved"


@dataclass(frozen=True, slots=True)
class Linking:
    def read(self, name: str, data: Json) -> object:
        assert isinstance(data, dict)
        return Status("/home", ()) if name == "status" else Unresolved((Target("Gone", 2, ("v/a.md",)),), more=True)

    def job(self, name: str, data: Json) -> Started:
        raise AssertionError((name, data))

    def jobs(self) -> tuple[ServedJob, ...]:
        return ()


def test_collections_says_when_more_link_targets_wait_than_it_lists() -> None:
    answer = cast("Object", tool("collections").call(Linking(), {"unresolved": True}))
    assert (answer["unresolved_targets"], answer["more_unresolved"]) == (
        [{"target": "Gone", "documents": 2, "ids": ["v/a.md"]}],
        True,
    )


def test_collections_lists_the_files_that_could_not_be_read_when_asked() -> None:
    unread = tuple(UnreadableFile(f"scan{n}.pdf", "no text in it") for n in range(101))
    status = Status("/home", (CollectionStatus("notes", "/n", "/s/n", Counted(5, 9, 0, 101, unread)),))
    answer = cast("Object", tool("collections").call(Faked(status, {"unreadable": True}), {"unreadable": True}))
    (shown,) = cast("list[Object]", answer["collections"])
    listed = cast("list[Object]", shown["unreadable_files"])
    assert (shown["unreadable"], len(listed), listed[0]) == (
        101,
        100,
        {"path": "scan0.pdf", "problem": "no text in it"},
    )


@dataclass(frozen=True, slots=True)
class Faked:
    status: Status
    asked: Mapping[str, Json] = MappingProxyType({})

    def read(self, name: str, data: Json) -> object:
        assert (name, data) == ("status", dict(self.asked))
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
