import contextlib
import json
import sqlite3
from collections.abc import Generator

import numpy as np

from sift.api.fields import Fields
from sift.api.filters import Filter, FilterValue, Operator
from sift.domain.arrays import ints
from sift.domain.query.filters import UNLINKED, Linked, Linking, condition
from sift.storage.database import Reading
from sift.storage.passing import Passing

ANY, NONE = Operator.ANY, Operator.NONE
PAPERS: dict[str, Fields] = {
    "brush.md": {"categories": ["cs.GR", "cs.RO"], "year": 2024},
    "rl.md": {"year": 2023},
    "record.md": {"year": "2024", "share": "0.5", "open": "true"},
    "typed.md": {"year": 2024, "share": 0.5, "open": True, "code": "007"},
}


@contextlib.contextmanager
def stored(documents: dict[str, Fields]) -> Generator[sqlite3.Connection]:
    with contextlib.closing(sqlite3.connect(":memory:")) as database:
        database.execute("CREATE TABLE sources (id INTEGER PRIMARY KEY, path TEXT)")
        database.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, source INTEGER, fields TEXT)")
        database.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, document INTEGER)")
        for at, (name, fields) in enumerate(documents.items()):
            database.execute("INSERT INTO sources VALUES (?, ?)", (at, name))
            database.execute("INSERT INTO documents VALUES (?, ?, ?)", (at, at, json.dumps(fields)))
            database.execute("INSERT INTO chunks VALUES (?, ?)", (at, at))
        yield database


def passing(
    documents: dict[str, Fields], filters: list[Filter], path: str | None = None, linked: Linked = UNLINKED
) -> list[str]:
    found = condition(path, filters, reads_quotes=True, linked=linked)
    names = list(documents)
    with stored(documents) as database:
        if found is None:
            return sorted(names)
        listed = sorted(names[chunk] for chunk in ints(Passing(Reading(database), found).listed()))
        every = np.arange(len(names), dtype=np.int64)[::-1]
        let = Passing(Reading(database), found).checked(every)
        assert sorted(names[chunk] for chunk in ints(every[let])) == listed
        return listed


def one(field: str, operator: Operator, *values: FilterValue) -> Filter:
    return Filter(field, operator, values)


def test_a_value_matches_a_field_holding_it_or_a_list_holding_it_and_a_path_glob_its_documents() -> None:
    assert passing(PAPERS, [one("categories", ANY, "cs.GR")]) == ["brush.md"]
    assert passing(PAPERS, [one("categories", ANY, "cs.LG")]) == []
    assert passing(PAPERS, [], "b*") == ["brush.md"]


def test_a_value_matches_as_text_as_the_number_and_as_the_yes_or_no_it_reads_as() -> None:
    for year in (2024, "2024"):
        assert passing(PAPERS, [one("year", ANY, year)]) == ["brush.md", "record.md", "typed.md"]
    for share in (0.5, "0.5"):
        assert passing(PAPERS, [one("share", ANY, share)]) == ["record.md", "typed.md"]
    for yes in (True, "true"):
        assert passing(PAPERS, [one("open", ANY, yes)]) == ["record.md", "typed.md"]
    assert passing(PAPERS, [one("open", ANY, "false")]) == []
    assert passing(PAPERS, [one("code", ANY, "007")]) == ["typed.md"]
    assert passing(PAPERS, [one("year", ANY, "9" * 30)]) == []


def test_several_values_mean_any_of_them_and_none_of_them() -> None:
    assert passing(PAPERS, [one("categories", ANY, "cs.LG", "cs.GR")]) == ["brush.md"]
    assert passing(PAPERS, [one("categories", NONE, "cs.GR")]) == ["record.md", "rl.md", "typed.md"]
    assert passing(PAPERS, [one("categories", NONE, "cs.LG", "cs.CV")]) == sorted(PAPERS)
    assert passing(PAPERS, [one("year", ANY, "1999", "2024"), one("share", ANY, 0.5)]) == ["record.md", "typed.md"]


def test_ranges_compare_numbers_as_numbers_and_text_at_the_precision_given() -> None:
    dated: dict[str, Fields] = {
        f"{day}.md": {"created": day} for day in ("2023-12-31", "2024-06-15", "2024-12-31", "2025-01-02")
    }
    within = [one("created", Operator.AT_LEAST, "2024-06"), one("created", Operator.BELOW, "2025")]
    assert passing(dated, within) == ["2024-06-15.md", "2024-12-31.md"]
    assert passing(dated, [one("created", Operator.AT_MOST, "2024")]) == [
        "2023-12-31.md",
        "2024-06-15.md",
        "2024-12-31.md",
    ]
    assert passing(PAPERS, [one("year", Operator.ABOVE, 2023), one("year", Operator.AT_MOST, 2024)]) == [
        "brush.md",
        "record.md",
        "typed.md",
    ]
    assert passing(PAPERS, [one("year", Operator.AT_LEAST, 2020), one("year", NONE, 2023)]) == [
        "brush.md",
        "record.md",
        "typed.md",
    ]


def test_a_number_range_passes_over_text_without_a_number_and_a_text_range_over_lists_and_yes_or_no() -> None:
    years: dict[str, Fields] = {
        "dated": {"year": "1990"},
        "unknown": {"year": "unknown"},
        "empty": {"year": ""},
        "nested": {"year": [[1990]]},
        "era": {"year": "-50 BC"},
        "spaced": {"year": " 1985"},
    }
    assert passing(years, [one("year", Operator.BELOW, 2000)]) == ["dated", "era", "spaced"]
    made: dict[str, Fields] = {
        "dated": {"c": "2025-03"},
        "nested": {"c": [["2025"]]},
        "flag": {"c": True},
        "number": {"c": 2025},
    }
    assert passing(made, [one("c", Operator.ABOVE, "2024")]) == ["dated", "number"]


def test_a_range_on_a_list_needs_one_element_within_it_all() -> None:
    multi: dict[str, Fields] = {"multi": {"years": [2019, 2030]}}
    assert passing(multi, [one("years", Operator.ABOVE, 2020), one("years", Operator.BELOW, 2025)]) == []


def test_any_field_name_is_addressed_and_odd_numbers_break_nothing() -> None:
    odd: dict[str, Fields] = {
        "cited": {"journal-ref": "Nature 12", "date created": "2024-05-01", 'say "hi"': "yes"},
        "odd": {"score": "NaN"},
    }
    assert passing(odd, [one("journal-ref", ANY, "Nature 12")]) == ["cited"]
    assert passing(odd, [one("date created", Operator.AT_LEAST, "2024-05")]) == ["cited"]
    assert passing(odd, [one('say "hi"', ANY, "yes")]) == ["cited"]
    assert passing(odd, [one("score", ANY, "NaN")]) == ["odd"]


def test_documents_linking_a_note_pass_its_wikilink_value_and_fail_its_negation() -> None:
    acme, not_acme = Linking("year", ANY, "Acme"), Linking("year", NONE, "Acme")
    assert passing(PAPERS, [one("year", ANY, 2023, "[[Acme]]")], linked=Linked({acme: [3]})) == ["rl.md", "typed.md"]
    assert passing(PAPERS, [one("year", ANY, "[[Acme]]")], linked=Linked({acme: []})) == []
    assert passing(PAPERS, [one("year", NONE, "[[Acme]]")], linked=Linked({not_acme: [0]})) == [
        "record.md",
        "rl.md",
        "typed.md",
    ]
    ranged = [one("year", ANY, "[[Acme]]"), one("year", Operator.ABOVE, 2000)]
    assert passing(PAPERS, ranged, linked=Linked({acme: [1]})) == []


def test_a_search_kept_to_some_documents_passes_only_those() -> None:
    assert passing(PAPERS, [], linked=Linked(within=[2, 0])) == ["brush.md", "record.md"]
    assert passing(PAPERS, [one("year", ANY, 2024)], linked=Linked(within=[2, 1])) == ["record.md"]
    assert passing(PAPERS, [], "b*", linked=Linked(within=[2])) == []
    assert passing(PAPERS, [], linked=Linked(within=[])) == []


def test_documents_left_out_never_pass() -> None:
    assert passing(PAPERS, [], linked=Linked(without=[0, 3])) == ["record.md", "rl.md"]
    assert passing(PAPERS, [one("year", ANY, 2024)], linked=Linked(within=[0, 2], without=[0])) == ["record.md"]


def steps(documents: dict[str, Fields], filters: list[Filter], linked: Linked = UNLINKED) -> int:
    found = condition(None, filters, reads_quotes=True, linked=linked)
    assert found is not None
    ticks: list[int] = []

    def tick() -> int:
        ticks.append(1)
        return 0

    with stored(documents) as database:
        database.set_progress_handler(tick, 1000)
        Passing(Reading(database), found).listed()
    return len(ticks)


def test_documents_linking_a_note_cost_about_what_a_plain_value_costs() -> None:
    many: dict[str, Fields] = {f"{n}.md": {"org": f"o{n % 7}"} for n in range(3000)}
    half = list(range(0, 3000, 2))
    plain = steps(many, [one("org", ANY, "o1")])
    assert steps(many, [one("org", ANY, "[[Acme]]")], Linked({Linking("org", ANY, "Acme"): half})) < 3 * plain
    assert steps(many, [one("org", NONE, "[[Acme]]")], Linked({Linking("org", NONE, "Acme"): half})) < 3 * plain
