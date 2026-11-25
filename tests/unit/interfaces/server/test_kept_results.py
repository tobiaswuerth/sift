import threading
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from sift.api.errors import Problem
from sift.api.events import Noted, Phase, PhaseDone, PhaseStarted
from sift.api.indexing import IndexCollections, IndexResult
from sift.api.notices import Notice
from sift.api.search import Hit, SearchKeywords, SearchResult
from sift.interfaces.acts.acting import Act, Acting
from sift.interfaces.server.handled import HANDLED, Handled, handled
from sift.interfaces.server.results import KEPT_HITS, RESULTS
from sift.settings import Settings
from tests.files import write
from tests.homes import indexed
from tests.serving import Line, asked, counted, last, running
from tests.waits import joined, waited

HIT = Hit("n/a.md", "n", "a.md", "A", 1, 1, 1.0, None)
UNSEARCHED = Notice(Problem.STORE_MISSING, "the store is missing", None, "m")


def hits_of(lines: list[Line]) -> list[str]:
    return [cast("str", hit["id"]) for hit in cast("list[Line]", last(lines, "answer")["hits"])]


def answering(answers: list[SearchResult], ran: list[str]) -> Mapping[str, Handled]:
    def run(acting: Acting, request: SearchKeywords) -> SearchResult:
        ran.append(request.query)
        if request.query == "noisy":
            acting.report(Noted("a note along the way"))
        if request.query == "loading":
            acting.report(PhaseStarted(Phase.LOAD, None))
            acting.report(PhaseDone(1))
        return answers[0]

    return {**HANDLED, "search": handled(Act("search", SearchKeywords, SearchResult, run), kept=True)[1]}


def test_a_search_asked_again_is_answered_from_memory_until_the_home_is_written(
    settings: Settings, notes: Path
) -> None:
    ran = list[str]()
    with running(settings, counted(ran)) as server:
        asked(server, "add", {"folder": str(notes)})
        asked(server, "index", {})
        first = asked(server, "search", {"query": "robot OR halo"})
        again = asked(server, "search", {"k": 10, "query": "robot OR halo"})
        assert (again, ran.count("search")) == (first, 1)
        write(notes / "robots.md", "# Robots\n\nA robot arm.\n")
        asked(server, "index", {})
        after = asked(server, "search", {"query": "robot OR halo"})
    assert (ran.count("search"), len(hits_of(after[1]))) == (2, 3)


@pytest.mark.parametrize(
    "answer",
    [
        SearchResult((), 1, (), partial=True, scored_by="keywords"),
        SearchResult((), 1, (UNSEARCHED,), scored_by="keywords"),
        SearchResult((HIT,) * (KEPT_HITS + 1), 1, (), scored_by="keywords"),
    ],
)
def test_an_answer_cut_short_passing_a_store_over_or_too_large_is_not_kept(
    settings: Settings, answer: SearchResult
) -> None:
    ran = list[str]()
    with running(settings, answering([answer], ran)) as server:
        for _ in range(2):
            asked(server, "search", {"query": "x"})
    assert ran == ["x", "x"]


def test_an_answer_whose_search_noted_something_along_the_way_is_not_kept_one_that_only_progressed_is(
    settings: Settings,
) -> None:
    ran = list[str]()
    with running(settings, answering([SearchResult((HIT,), 1, (), scored_by="keywords")], ran)) as server:
        said = [asked(server, "search", {"query": "noisy"})[1] for _ in range(2)]
        progressed = [asked(server, "search", {"query": "loading"})[1] for _ in range(2)]
    assert ran == ["noisy", "noisy", "loading"]
    assert [lines[0]["event"] for lines in said] == ["Noted", "Noted"]
    assert [len(lines) for lines in progressed] == [3, 1]


def test_only_the_latest_answers_are_kept(settings: Settings) -> None:
    ran = list[str]()
    with running(settings, answering([SearchResult((HIT,), 1, (), scored_by="keywords")], ran)) as server:
        for n in [*range(RESULTS + 1), RESULTS, 0]:
            asked(server, "search", {"query": f"q{n}"})
    assert ran == [*(f"q{n}" for n in range(RESULTS + 1)), "q0"]


def test_an_answer_asked_again_is_kept_as_the_latest(settings: Settings) -> None:
    ran = list[str]()
    with running(settings, answering([SearchResult((HIT,), 1, (), scored_by="keywords")], ran)) as server:
        for n in [*range(RESULTS), 0, RESULTS, 0, 1]:
            asked(server, "search", {"query": f"q{n}"})
    assert ran == [*(f"q{n}" for n in range(RESULTS)), f"q{RESULTS}", "q1"]


def test_answers_are_kept_again_once_a_write_is_done(settings: Settings, notes: Path) -> None:
    ran = list[str]()
    with running(settings, counted(ran)) as server:
        asked(server, "add", {"folder": str(notes)})
        asked(server, "index", {})
        for _ in range(3):
            asked(server, "search", {"query": "halo"})
    assert ran.count("search") == 1


def test_an_answer_is_not_kept_past_a_write_made_beside_the_home_process(settings: Settings, notes: Path) -> None:
    ran = list[str]()
    with running(settings, counted(ran)) as server:
        asked(server, "add", {"folder": str(notes)})
        asked(server, "index", {})
        before = [hits_of(asked(server, "search", {"query": "zebra"})[1]) for _ in range(2)]
        write(notes / "zebra.md", "# Zebra\n\nA zebra grazes.\n")
        indexed(settings.home)
        after = [hits_of(asked(server, "search", {"query": "zebra"})[1]) for _ in range(2)]
    assert (before, after, ran.count("search")) == ([[], []], [["notes/zebra.md"]] * 2, 2)


def test_an_answer_whose_search_saw_a_write_beside_the_home_process_is_not_kept(
    settings: Settings, notes: Path
) -> None:
    ran = list[str]()

    def written_meanwhile(acting: Acting, request: SearchKeywords) -> SearchResult:
        ran.append(request.query)
        write(notes / f"{len(ran)}.md", f"# Note {len(ran)}\n\nWritten while a search ran.\n")
        indexed(acting.settings.home)
        return SearchResult((HIT,), 1, (), scored_by="keywords")

    searched = {
        **HANDLED,
        "search": handled(Act("search", SearchKeywords, SearchResult, written_meanwhile), kept=True)[1],
    }
    with running(settings, searched) as server:
        asked(server, "add", {"folder": str(notes)})
        asked(server, "index", {})
        for _ in range(2):
            asked(server, "search", {"query": "x"})
    assert ran == ["x", "x"]


def test_an_answer_found_while_the_home_is_written_is_not_kept(settings: Settings) -> None:
    started, release = threading.Event(), threading.Event()

    def writing(_acting: Acting, _request: IndexCollections) -> SearchResult:
        started.set()
        waited(release, 30)
        return SearchResult((), 0, (), scored_by="keywords")

    ran = list[str]()
    index = handled(Act("index", IndexCollections, IndexResult, writing, writes=True))[1]
    blocked = {**answering([SearchResult((HIT,), 1, (), scored_by="keywords")], ran), "index": index}
    with running(settings, blocked) as server:
        write_thread = threading.Thread(target=asked, args=(server, "index", {}))
        write_thread.start()
        waited(started, 30)
        for _ in range(2):
            asked(server, "search", {"query": "x"})
        release.set()
        joined(write_thread, 30)
    assert ran == ["x", "x"]
