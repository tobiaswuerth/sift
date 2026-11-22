import pytest

from tools import check_idioms
from tools.check_idioms import idioms


def problems(path: str, source: str) -> list[str]:
    return [finding.problem for finding in idioms(path, source)]


def test_plain_code_passes_everywhere() -> None:
    source = "import re\n\n\ndef f(text: str) -> list[str]:\n    return text.split('\\n')\n"
    for path in (
        "src/sift/domain/a.py",
        "src/sift/storage/a.py",
        "src/sift/interfaces/a.py",
        "tests/a.py",
        "tools/a.py",
    ):
        assert problems(path, source) == []


@pytest.mark.parametrize(
    "character",
    [
        "\N{NEXT LINE}",
        "\N{NO-BREAK SPACE}",
        "\N{ZERO WIDTH SPACE}",
        "\N{LINE SEPARATOR}",
        "\N{PARAGRAPH SEPARATOR}",
        "\N{ZERO WIDTH NO-BREAK SPACE}",
        "\N{REPLACEMENT CHARACTER}",
        "\N{COMBINING ACUTE ACCENT}",
        "\N{ZERO WIDTH JOINER}",
        "\N{CHARACTER TABULATION}",
        "\N{ESCAPE}",
        "\N{EN QUAD}",
        "\N{VARIATION SELECTOR-16}",
    ],
)
def test_an_invisible_character_written_as_itself_is_found(character: str) -> None:
    (problem,) = problems("tests/a.py", f"x = 'e{character}'\n")
    assert problem == f"U+{ord(character):04X} written as itself: write it as an escape"


def test_a_visible_character_written_as_itself_passes() -> None:
    visible = "M\N{LATIN SMALL LETTER U WITH DIAERESIS}ller \N{CJK UNIFIED IDEOGRAPH-65E5}"
    assert problems("tests/a.py", f"x = '{visible}'\n") == []


def test_an_invisible_character_written_as_an_escape_passes() -> None:
    assert problems("src/sift/a.py", "x = '\\u2028\\ufffd'\n") == []


@pytest.mark.parametrize("method", ["splitlines", "isdigit", "isdecimal", "isnumeric"])
def test_a_banned_method_is_found_with_its_fix(method: str) -> None:
    for path in ("src/sift/a.py", "tools/a.py"):
        (problem,) = problems(path, f"y = x.{method}()\n")
        assert problem.startswith(f".{method}: ")
    assert problems("tests/a.py", f"y = x.{method}()\n") == []


def test_a_numbered_sql_placeholder_is_found() -> None:
    numbered = f"db.execute('SELECT ?{1}, ?{2}')\n"
    for path in ("src/sift/storage/a.py", "tests/a.py", "tools/a.py"):
        (problem,) = problems(path, numbered)
        assert problem.startswith("a numbered SQL placeholder")
    assert problems("src/sift/storage/a.py", "db.execute('SELECT ?, :name')\n") == []


@pytest.mark.parametrize("call", ["open('a')", "input()", "print('a')"])
def test_io_builtins_are_found_in_the_core_alone(call: str) -> None:
    name = call.split("(", maxsplit=1)[0]
    assert problems("src/sift/domain/a.py", f"x = {call}\n") == [
        f"{name}() in the core: return the value and let the caller act"
    ]
    assert problems("src/sift/api/a.py", f"x = {call}\n") != []
    assert problems("src/sift/storage/a.py", f"x = {call}\n") == []


@pytest.mark.parametrize("source", ["raise SystemExit(1)\n", "exit(1)\n", "quit()\n", "except SystemExit: ...\n"])
def test_exits_are_found_outside_the_interfaces(source: str) -> None:
    source = f"try: ...\n{source}" if source.startswith("except") else source
    for path in ("src/sift/app/a.py", "src/sift/domain/a.py", "src/sift/__main__.py"):
        (problem,) = problems(path, source)
        assert problem.endswith("outside sift.interfaces: raise a SiftError")
    for path in ("src/sift/interfaces/cli/main.py", "tools/a.py", "tests/a.py"):
        assert problems(path, source) == []


@pytest.mark.parametrize(
    "source",
    [
        "def test(monkeypatch) -> None: ...\n",
        "def test(tmp_path, *, monkeypatch: pytest.MonkeyPatch) -> None: ...\n",
        "def test() -> None:\n    with pytest.MonkeyPatch.context() as patch: ...\n",
    ],
)
def test_monkeypatch_is_found_in_tests(source: str) -> None:
    assert set(problems("tests/a.py", source)) == {"monkeypatch: pass a fake, a value or an environment in instead"}


@pytest.mark.parametrize(
    "source",
    [
        "db = sqlite3.connect(':memory:')\n",
        "with sqlite3.connect(path) as db: ...\n",
        "rows = sqlite3.connect(path).execute('SELECT 1').fetchall()\n",
    ],
)
def test_a_database_a_test_leaves_open_is_found(source: str) -> None:
    (problem,) = problems("tests/a.py", source)
    assert problem.startswith("sqlite3.connect left to the garbage collector")
    assert problems("src/sift/storage/a.py", source) == []


@pytest.mark.parametrize("closing", ["closing", "contextlib.closing"])
def test_a_database_a_test_closes_passes(closing: str) -> None:
    assert problems("tests/a.py", f"with {closing}(sqlite3.connect(':memory:')) as db: ...\n") == []


@pytest.mark.parametrize(
    "source", ["import time\n", "import os, timeit\n", "from time import perf_counter\n", "import time.something\n"]
)
def test_a_clock_is_found_in_tests_alone(source: str) -> None:
    (problem,) = problems("tests/a.py", source)
    assert problem.startswith("a clock in a test")
    assert problems("src/sift/interfaces/a.py", source) == []
    assert problems("tests/a.py", "import datetime\nfrom . import timetable\n") == []


def test_the_repository_keeps_the_idioms() -> None:
    assert check_idioms.main([]) == 0
