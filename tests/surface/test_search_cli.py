import io
import json
from pathlib import Path
from typing import cast

from sift.interfaces.cli.main import Host, run
from sift.settings import Settings
from tests.surface.commands import indexed_notes, ran


def searched_notes(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    for n in range(6):
        (notes / f"filler{n}.md").write_text(f"# Filler {n}\n\nNothing here, note {n}.\n", encoding="utf-8")
    (notes / "rl.md").write_text(
        "# Reward shaping\n\nReward shaping speeds up learning.\n" + "x " * 200, encoding="utf-8"
    )
    assert ran(settings, "index")[0] == 0


def ticking(settings: Settings, *argv: str) -> tuple[int, str, str]:
    out, err, ticks = io.StringIO(), io.StringIO(), iter([10.0, 10.25])
    code = run(argv, settings, out, err, Host(clock=lambda: next(ticks)))
    return code, out.getvalue(), err.getvalue()


def test_search_ranks_passages_with_where_they_are_and_says_what_it_searched(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    code, out, err = ticking(settings, "search", "shaping OR brush", "-k", "5")
    lines = out.split("\n")
    assert code == 0
    assert lines[:3] == [f"1  {lines[0][3:7]}  Brush", "         notes/robots/brush.md:1-1", "         # Brush"]
    assert lines[3] == f"2  {lines[3][3:7]}  Reward shaping"
    assert lines[4] == "         notes/rl.md:1-4"
    assert lines[5].startswith("         # Reward shaping Reward shaping speeds up learning. x x")
    assert lines[5].endswith(" ...")
    assert len(lines[5]) <= 9 + 200
    assert err == "2 results  \N{MIDDLE DOT}  keyword  \N{MIDDLE DOT}  1 collection  \N{MIDDLE DOT}  0.2 s\n"


def test_search_answers_in_json_when_asked(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    code, out, err = ran(settings, "search", "brush", "--json", "--compact")
    assert (code, err) == (0, "")
    assert json.loads(out) == {
        "hits": [
            {
                "id": "notes/robots/brush.md",
                "title": "Brush",
                "lines": "1-1",
                "score": json.loads(out)["hits"][0]["score"],
                "text": None,
                "also": [],
                "fields": None,
            }
        ],
        "searched": 1,
        "unsearched": [],
        "partial": False,
        "ruled_out": 0,
    }


def test_a_search_without_matches_says_so(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    code, out, err = ticking(settings, "search", "zzz", "-c", "notes")
    assert (code, out) == (0, "")
    assert err.startswith("no matches  \N{MIDDLE DOT}  keyword  \N{MIDDLE DOT}  1 collection")


def test_a_query_outside_the_grammar_is_refused_with_how_to_write_one(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    code, out, err = ran(settings, "search", "a AND")
    assert (code, out) == (1, "")
    assert err.startswith("error  the query is not understood: it ends where words are missing\n  fix  write words")


def test_search_fails_if_a_collection_could_not_be_searched_however_it_answers(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    (settings.home / "collections" / "notes.sqlite").unlink()
    code, out, err = ran(settings, "search", "brush")
    assert (code, out) == (1, "")
    assert err.startswith("error  notes: the store is missing: ")
    assert ran(settings, "search", "brush", "--json")[0] == 1


def test_a_search_without_any_collection_says_how_to_add_one(settings: Settings) -> None:
    code, out, err = ran(settings, "search", "x")
    assert (code, out) == (0, "")
    assert err == "empty  no collections yet\n next  sift add FOLDER\n"


def test_k_is_a_whole_number(settings: Settings) -> None:
    code, out, err = ran(settings, "search", "x", "-k", "many")
    assert (code, out) == (2, "")
    assert "'many' is no whole number" in err


def test_a_budget_of_no_time_is_refused(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    code, out, err = ran(settings, "search", "brush", "--budget", "0")
    assert (code, out) == (1, "")
    assert "--budget 0.0 leaves no time to search" in err


def test_search_keeps_to_front_matter_and_paths_shows_it_and_orders_by_it(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    for name, year in (("a", 2023), ("b", 2024), ("c", 2022)):
        (notes / "papers").mkdir(exist_ok=True)
        text = f"---\nyear: {year}\n---\n# {name}\n\nbrush\n"
        (notes / "papers" / f"{name}.md").write_text(text, encoding="utf-8")
    assert ran(settings, "index")[0] == 0
    argv = ("search", "brush", "-f", "year>=2023", "--path", "papers/*", "--fields", "year", "--sort=-year", "--json")
    code, out, _ = ran(settings, *argv, "--compact")
    found = cast("list[dict[str, object]]", json.loads(out)["hits"])
    assert code == 0
    assert [(hit["id"], hit["fields"]) for hit in found] == [
        ("notes/papers/b.md", {"year": 2024}),
        ("notes/papers/a.md", {"year": 2023}),
    ]
    code, out, err = ran(settings, "search", "brush", "-f", "year")
    assert (code, out) == (1, "")
    assert err.startswith("error  the filter 'year' is no KEY=VALUE")


def test_unresolved_fails_if_a_collection_could_not_be_read_however_it_answers(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    (settings.home / "collections" / "notes.sqlite").unlink()
    code, out, err = ran(settings, "unresolved")
    assert (code, out) == (1, "")
    assert err.startswith("error  notes: the store is missing: ")
    assert ran(settings, "unresolved", "--json")[0] == 1


def test_links_of_a_note_named_while_a_collection_could_not_be_read_fail(
    settings: Settings, notes: Path, tmp_path: Path
) -> None:
    searched_notes(settings, notes)
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.md").write_text("# X\n", encoding="utf-8")
    assert ran(settings, "add", str(other))[0] == 0
    assert ran(settings, "index")[0] == 0
    (settings.home / "collections" / "other.sqlite").unlink()
    code, out, err = ran(settings, "links", "Brush")
    assert (code, out) == (1, "Brush (notes/robots/brush.md) links to: nothing\n")
    assert err.startswith("error  other: the store is missing: ")
    assert ran(settings, "links", "Brush", "--json")[0] == 1
    assert ran(settings, "links", "notes/robots/brush.md")[0] == 0
