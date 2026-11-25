from pathlib import Path

import pytest

from tools import limits
from tools.limits import Limit, compared, limits_in, table

SOURCE = """\
MOST_ITEMS = 1_000
WAIT_SECONDS: float = 2.5
KEPT_BYTES = 64 << 20
FEWEST_WORDS, LEAST_SHARE = 2, 0.1
OTHER = 3
MOST_DECLARED: int
MOST_A, MOST_B = PAIR


def f() -> None:
    MOST_INNER = 1


class C:
    MOST_FIELD = 2
"""


def test_bounds_are_the_module_constants_named_as_bounds_with_their_values_as_written() -> None:
    assert limits_in("sift.a", SOURCE) == [
        Limit("sift.a", "MOST_ITEMS", "1_000"),
        Limit("sift.a", "WAIT_SECONDS", "2.5"),
        Limit("sift.a", "KEPT_BYTES", "64 << 20"),
        Limit("sift.a", "FEWEST_WORDS", "2"),
        Limit("sift.a", "LEAST_SHARE", "0.1"),
    ]


def test_the_table_lists_each_bound_by_its_module_in_the_order_written() -> None:
    told = table([Limit("sift.b", "MOST_X", "1"), Limit("sift.a", "MOST_Y", "2 | 3")])
    assert told.splitlines()[-3:] == [
        "|---|---|---|",
        "| `sift.b` | `MOST_X` | `1` |",
        "| `sift.a` | `MOST_Y` | `2 \\| 3` |",
    ]


def test_a_table_unlike_the_code_is_found_and_written_anew_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    page = tmp_path / "limits.md"
    page.write_text("old\n", encoding="utf-8")
    assert compared(page, "new\n", write=False) == 1
    assert "python -m tools limits --write" in capsys.readouterr().out
    assert (compared(page, "new\n", write=True), page.read_text(encoding="utf-8")) == (0, "new\n")


def test_the_committed_table_lists_every_bound_sift_sets() -> None:
    assert limits.main([]) == 0
