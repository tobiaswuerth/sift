import io
import json
from pathlib import Path
from typing import cast

import pytest

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
    assert lines[:3] == ["1  3.01  Brush", "         notes/robots/brush.md:1-1", "         # Brush"]
    assert lines[3] == "2  1.17  Reward shaping"
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
                "score": pytest.approx(3.00625, rel=1e-5),
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
    assert (code, out) == (2, "")
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
    assert (code, out) == (2, "")
    assert "error  --budget: 0.0 seconds leave no time to search" in err


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
    assert (code, out) == (2, "")
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


def _stopped() -> float:
    raise KeyboardInterrupt


def test_an_interrupt_says_what_was_kept_where_sift_writes_and_ends_with_130(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    for argv, said in ((("index",), "stopped: what was written so far is kept\n"), (("search", "brush"), "")):
        out, err = io.StringIO(), io.StringIO()
        assert run(argv, settings, out, err, Host(clock=_stopped)) == 130
        assert (out.getvalue(), err.getvalue()) == ("", said)


def test_a_stream_that_cannot_print_the_glyphs_gets_ascii_ones(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    raw = io.BytesIO()
    err = io.TextIOWrapper(raw, encoding="cp1252")
    assert run(("search", "brush"), settings, io.StringIO(), err) == 0
    err.flush()
    said = raw.getvalue().decode("utf-8")
    assert "  |  keyword  |  " in said
    assert "\N{MIDDLE DOT}" not in said


def test_an_error_asked_for_in_json_is_json_on_stdout(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    code, out, err = ran(settings, "search", "brush", "-c", "nope", "--json")
    said = cast("dict[str, dict[str, str]]", json.loads(out))["error"]
    assert (code, said["kind"], said["message"]) == (1, "not_found", err.splitlines()[0].removeprefix("error  "))


def test_a_mistaken_argument_exits_2_with_a_fix_and_no_usage_block(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    for argv in (("search", "brush", "-k", "abc"), ("search", "brush", "-k", "0")):
        code, out, err = ran(settings, *argv)
        lines = err.splitlines()
        assert (code, out, len(lines)) == (2, "", 2)
        assert (lines[0].startswith("error  "), lines[1].startswith("  fix  ")) == (True, True)
    code, _, err = ran(settings, "search", "brush", "--sort", "-year")
    assert code == 2
    assert "a value that starts with a dash is written --sort=-value; see sift search --help" in err


def test_an_option_given_twice_is_refused_and_collections_may_be_given_again(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    code, _, err = ran(settings, "search", "brush", "-k", "2", "-k", "3")
    assert (code, err.splitlines()[0]) == (2, "error  argument -k/--k: given twice: give it once")
    other = notes.parent / "other"
    other.mkdir()
    (other / "brush.md").write_text("# Brush again\n", encoding="utf-8")
    assert ran(settings, "add", str(other))[0] == 0
    assert ran(settings, "index")[0] == 0
    code, out, _ = ran(settings, "search", "brush", "-c", "notes", "-c", "other", "--json")
    found = cast("dict[str, list[dict[str, str]]]", json.loads(out))["hits"]
    assert (code, sorted(hit["id"] for hit in found)) == (0, ["notes/robots/brush.md", "other/brush.md"])


def test_with_no_collection_every_command_that_reads_one_says_how_to_add_one(settings: Settings) -> None:
    empty = "empty  no collections yet\n next  sift add FOLDER\n"
    for argv in (("get", "a/b.md"), ("links", "A"), ("unresolved",), ("mentions", "A"), ("similar", "a/b.md")):
        assert ran(settings, *argv) == (0, "", empty)


def test_status_of_a_store_that_is_gone_fails_and_json_may_come_before_the_command(
    settings: Settings, notes: Path
) -> None:
    indexed_notes(settings, notes)
    code, out, _ = ran(settings, "--json", "status")
    assert (code, cast("dict[str, str]", json.loads(out))["home"]) == (0, str(settings.home))
    (settings.home / "collections" / "notes.sqlite").unlink()
    assert ran(settings, "status")[0] == 1


def test_a_refused_value_is_said_with_the_flag_that_gave_it(settings: Settings, notes: Path) -> None:
    searched_notes(settings, notes)
    code, _, err = ran(settings, "search", "brush", "-k", "0")
    assert (code, err.splitlines()[0]) == (2, "error  -k: 0 is outside 1 to 10,000")
    code, out, _ = ran(settings, "search", "brush", "-k", "0", "--json")
    said = cast("dict[str, dict[str, str]]", json.loads(out))["error"]
    assert (said["message"], said["field"]) == ("0 is outside 1 to 10,000", "k")
