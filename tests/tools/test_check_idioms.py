import ast
from collections.abc import Mapping

import pytest

from sift.interfaces.cli.entries import ENTRIES
from tests.tools.idiom_problems import problems
from tools import check_idioms, idioms_storage
from tools.canonical import canonical


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


@pytest.mark.parametrize(
    "handlers",
    [
        "except RuntimeError:\n    pass\n",
        "except (ValueError, Exception) as error:\n    print(error)\n",
        "except BaseException:\n    raise ValueError from None\n",
        "except:\n    pass\n",
    ],
)
def test_catching_recursion_error_as_one_of_its_bases_without_letting_it_through_is_found(handlers: str) -> None:
    assert problems("src/sift/ingest/a.py", f"try:\n    f()\n{handlers}") == [check_idioms.RECURSION_CAUGHT]


@pytest.mark.parametrize(
    "handlers",
    [
        "except RecursionError:\n    raise\nexcept RuntimeError:\n    pass\n",
        "except RecursionError:\n    raise\nexcept RuntimeError as error:\n    if not mine(error):\n        raise\n",
        "except BaseException:\n    close()\n    raise\n",
        "except (ValueError, OSError):\n    pass\n",
    ],
)
def test_catching_recursion_error_and_letting_it_through_passes(handlers: str) -> None:
    assert problems("src/sift/ingest/a.py", f"try:\n    f()\n{handlers}") == []


def test_catching_recursion_error_named_in_a_tuple_of_the_module_is_found() -> None:
    named = "INNER: tuple[type[Exception], ...] = (KeyError, RecursionError)\nCAUGHT = (ValueError, INNER)\n"
    source = f"{named}try:\n    f()\nexcept CAUGHT:\n    pass\nexcept INNER as error:\n    pass\n"
    assert problems("src/sift/domain/a.py", source) == [check_idioms.RECURSION_CAUGHT] * 2
    looped = "A = (B,)\nB = (A, ValueError)\ntry:\n    f()\nexcept A:\n    pass\n"
    assert problems("src/sift/domain/a.py", looped) == []


def test_catching_recursion_error_named_in_a_tuple_of_another_module_is_found() -> None:
    elsewhere = ast.parse("from sift.b import INNER\nCAUGHT = (ValueError, INNER)\nINNER_TOO = (RecursionError,)\n")
    modules = {"sift.a": elsewhere, "sift.b": ast.parse("INNER = (KeyError, RecursionError)\n")}

    def tuples_of(module: str) -> Mapping[str, list[ast.expr]]:
        return check_idioms.module_tuples(modules[module]) if module in modules else {}

    source = "from sift.a import CAUGHT\ntry:\n    f()\nexcept CAUGHT:\n    pass\nexcept xberg.XbergError:\n    pass\n"
    tree = canonical(ast.parse(source))
    found = list(check_idioms.recursion_caught("src/sift/c.py", tree, tuples_of))
    assert [each.problem for each in found] == [check_idioms.RECURSION_CAUGHT]
    unreadable = check_idioms.first_party_tuples("sift.ingest.mail")["UNREADABLE"]
    assert [ast.unparse(each) for each in unreadable] == [
        "email.errors.MessageError",
        "ValueError",
        "LookupError",
        "IndexError",
        "AttributeError",
        "NameError",
    ]


def test_the_repository_keeps_the_idioms() -> None:
    assert check_idioms.main([]) == 0


def test_a_flag_of_the_command_line_outside_the_interfaces_is_found() -> None:
    flagged = ["CLI syntax outside sift.interfaces: say what to do, and let the interface name its command or flag"]
    assert problems("src/sift/app/a.py", 'X = "ask for fewer, e.g. -k 10"\n') == flagged
    assert problems("src/sift/app/a.py", 'X = f"--budget {b} leaves no time"\n') == flagged
    assert problems("src/sift/app/a.py", 'X = "run sift embed --reset"\n') == flagged
    assert problems("src/sift/app/a.py", 'X = "run sift embed with reset"\n') == flagged
    assert problems("src/sift/app/a.py", 'X = "a note: its id, as sift get takes it"\n') == flagged
    allowed = 'X = "embed again with reset"\nY = "a -created sort, a k-means"\nZ = "uv tool install --reinstall a"\n'
    assert problems("src/sift/app/a.py", allowed) == []
    assert problems("src/sift/app/a.py", 'W = "from one sift only, or a newer sift"\n') == []
    assert problems("src/sift/interfaces/cli/a.py", 'X = "-k 10 or sift embed"\n') == []
    assert set(check_idioms.COMMANDS) >= set(ENTRIES)


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
    assert problems("tests/a.py", "import sqlite3 as s\ns.connect(':memory:')\n") == [idioms_storage.LEFT_OPEN]
    assert problems("src/sift/app/a.py", "from sys import exit as leave\nleave(1)\n") == [
        "sys.exit outside sift.interfaces: raise a SiftError"
    ]
    assert problems("src/sift/domain/a.py", "import builtins\nbuiltins.print(1)\n") == [
        "builtins.print() in the core: return the value and let the caller act"
    ]


def test_a_write_handle_is_made_in_the_storage_modules_alone() -> None:
    source = "from sift.storage.database import Writing\n\nWriting(connection)\n"
    assert problems("src/sift/app/a.py", source) == [idioms_storage.HANDLE_MADE]
    assert problems("src/sift/storage/database.py", source) == []


def test_sql_on_a_transaction_s_connection_runs_in_sift_storage_alone() -> None:
    source = "transaction.connection.execute('SELECT 1')\n"
    assert problems("src/sift/app/a.py", source) == [idioms_storage.RAW_SQL]
    assert problems("src/sift/storage/a.py", source) == []


def test_a_name_bound_once_is_followed_to_what_it_names() -> None:
    assert problems("src/sift/app/a.py", "held = transaction.connection\nheld.execute('x')\n") == [
        idioms_storage.RAW_SQL
    ]
    assert problems("src/sift/domain/a.py", "say = print\nsay(1)\n") == [
        "print() in the core: return the value and let the caller act"
    ]
    assert problems("src/sift/app/a.py", "a = b\nb = a\na.execute('x')\n") == []


@pytest.mark.parametrize(
    "change",
    ["vars(live)['REDRAW_SECONDS'] = 0", "live.__dict__['REDRAW_SECONDS'] = 0", "live.REDRAW_SECONDS, n = 0, 1"],
)
def test_a_sift_module_changed_through_its_namespace_or_in_a_tuple_is_found(change: str) -> None:
    source = f"from sift.interfaces.console import live\n{change}\n"
    assert check_idioms.PATCHED in problems("tests/a.py", source)


def test_a_document_s_text_cut_in_sql_is_found_in_sift_alone() -> None:
    cut = 'X = "SELECT substr(d.text, c.start + 1, c.length) FROM chunks c"\n'
    assert problems("src/sift/storage/a.py", cut) == [idioms_storage.TEXT_CUT_IN_SQL]
    assert problems("src/sift/storage/a.py", 'X = "SELECT substr(path, 2), text FROM sources"\n') == []
    assert problems("tests/a.py", cut) == []


def test_a_store_is_opened_by_hand_in_sift_storage_alone() -> None:
    source = "from sift.storage.store import open_store\n\nopen_store(path, 1)\n"
    assert problems("src/sift/app/a.py", source) == [idioms_storage.STORE_OPENED_BY_HAND]
    assert problems("src/sift/storage/a.py", source) == []
    assert problems("tests/a.py", source) == []


@pytest.mark.parametrize("cut", ["LAYOUT[:-1]", "store.LAYOUT[-1]", "CATALOG_LAYOUT[: -2]", "LAYOUT[-3:]"])
def test_a_layout_cut_from_its_end_is_found(cut: str) -> None:
    assert problems("tests/a.py", f"before = {cut}\n") == [idioms_storage.LAYOUT_FROM_ITS_END]


def test_a_layout_cut_at_a_place_passes() -> None:
    assert problems("tests/a.py", "before = LAYOUT[:7]\nmade = LAYOUT[5][0]\nlast = steps[-1]\n") == []
