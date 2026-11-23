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


@pytest.mark.parametrize(
    "code",
    ["os.environ['A']", "os.environ.get('A')", "os.getenv('A')", "from os import environ", "from os import getenv"],
)
def test_reading_the_environment_anywhere_in_sift_but_its_settings_is_found(code: str) -> None:
    source = f"import os\n{code}\n"
    (problem,) = problems("src/sift/app/a.py", source)
    assert problem.startswith("the environment read outside sift.settings")
    for path in ("src/sift/settings.py", "tests/a.py", "tools/a.py"):
        assert problems(path, source) == []


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


@pytest.mark.parametrize(
    "handler", ["except RecursionError:", "except (ValueError, RecursionError):", "except builtins.RecursionError:"]
)
def test_catching_recursion_error_is_found_outside_tests(handler: str) -> None:
    source = f"try:\n    f()\n{handler}\n    pass\n"
    assert problems("src/sift/domain/a.py", source) == [check_idioms.RECURSION_CAUGHT]
    assert problems("tests/a.py", source) == []


def test_the_repository_keeps_the_idioms() -> None:
    assert check_idioms.main([]) == 0


def test_a_flag_of_the_command_line_outside_the_interfaces_is_found() -> None:
    flagged = ["CLI syntax outside sift.interfaces: say what to do, and let the interface name its flag"]
    assert problems("src/sift/app/a.py", 'X = "ask for fewer, e.g. -k 10"\n') == flagged
    assert problems("src/sift/app/a.py", 'X = f"--budget {b} leaves no time"\n') == flagged
    assert problems("src/sift/app/a.py", 'X = "run sift embed --reset"\nY = "a -created sort, a k-means"\n') == []
    assert problems("src/sift/interfaces/cli/a.py", 'X = "-k 10"\n') == []


@pytest.mark.parametrize(
    "source",
    [
        "from sift.interfaces.console import live\nlive.REDRAW_SECONDS = 0\n",
        "import sift.interfaces.console.live as shown\nshown.REDRAW_SECONDS += 1\n",
        "from sift.app import filing\nsetattr(filing, 'READERS', None)\n",
        "from sift.app.filing import Readers\ndel Readers.lines\n",
    ],
)
def test_a_test_that_changes_a_sift_module_or_class_is_found(source: str) -> None:
    assert check_idioms.PATCHED in problems("tests/a.py", source)


def test_a_test_may_change_what_it_made() -> None:
    assert problems("tests/a.py", "from sift.app.filing import Readers\nheld = Readers()\nheld.x = 1\n") == []


@pytest.mark.parametrize(
    "source",
    [
        "import datetime\ndatetime.datetime.now()\n",
        "from datetime import datetime\ndatetime.utcnow()\n",
        "import datetime as when\nwhen.date.today()\n",
    ],
)
def test_reading_the_clock_is_found_outside_the_interfaces(source: str) -> None:
    assert problems("src/sift/app/a.py", source) == [check_idioms.CLOCK_READ]
    assert problems("tests/a.py", source) == [check_idioms.CLOCK_READ]
    assert problems("src/sift/interfaces/a.py", source) == []


@pytest.mark.parametrize(
    "source",
    [
        "import os as o\no.environ\n",
        "import os\ngetattr(os, 'environ')\n",
        "import os as o\ngetattr(o, 'getenv')('A')\n",
    ],
)
def test_the_environment_read_through_an_alias_is_found(source: str) -> None:
    assert problems("src/sift/app/a.py", source) == [check_idioms.ENVIRONMENT_READ]


def test_an_alias_hides_no_database_left_open_and_no_exit_or_print() -> None:
    assert problems("tests/a.py", "import sqlite3 as s\ns.connect(':memory:')\n") == [check_idioms.LEFT_OPEN]
    assert problems("src/sift/app/a.py", "from sys import exit as leave\nleave(1)\n") == [
        "sys.exit outside sift.interfaces: raise a SiftError"
    ]
    assert problems("src/sift/domain/a.py", "import builtins\nbuiltins.print(1)\n") == [
        "builtins.print() in the core: return the value and let the caller act"
    ]


def test_a_write_handle_is_made_in_the_storage_modules_alone() -> None:
    source = "from sift.storage.database import Writing\n\nWriting(connection)\n"
    assert problems("src/sift/app/a.py", source) == [check_idioms.HANDLE_MADE]
    assert problems("src/sift/storage/database.py", source) == []


def test_sql_on_a_transaction_s_connection_runs_in_sift_storage_alone() -> None:
    source = "transaction.connection.execute('SELECT 1')\n"
    assert problems("src/sift/app/a.py", source) == [check_idioms.RAW_SQL]
    assert problems("src/sift/storage/a.py", source) == []
