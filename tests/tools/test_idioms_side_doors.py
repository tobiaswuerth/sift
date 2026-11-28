import pytest

from tests.tools.idiom_problems import problems
from tools import check_idioms, idioms_storage


@pytest.mark.parametrize("called", ["time.time()", "time.perf_counter()", "time.monotonic_ns()", "perf_counter()"])
def test_a_clock_called_outside_the_interfaces_is_found(called: str) -> None:
    source = f"import time\nfrom time import perf_counter\n\n\ndef f() -> float:\n    return {called}\n"
    assert problems("src/sift/app/a.py", source) == [check_idioms.CLOCK_READ]
    assert problems("src/sift/interfaces/a.py", source) == []


def test_a_clock_handed_in_as_a_default_is_no_reading() -> None:
    source = "import time\n\n\ndef f(clock: Callable[[], float] = time.perf_counter) -> float:\n    return clock()\n"
    assert problems("src/sift/app/a.py", source) == []


@pytest.mark.parametrize("ended", ["os._exit(1)", "os.abort()"])
def test_a_process_ended_outside_the_interfaces_is_found(ended: str) -> None:
    source = f"import os\n\n{ended}\n"
    named = ended.split("(", maxsplit=1)[0]
    assert problems("src/sift/app/a.py", source) == [f"{named} outside sift.interfaces: raise a SiftError"]
    assert problems("src/sift/interfaces/a.py", source) == []


def test_the_reader_process_may_end_itself_once_its_parent_is_gone() -> None:
    assert problems("src/sift/ingest/sandbox.py", "import os\n\nos._exit(1)\n") == []
    assert problems("src/sift/ingest/sandbox.py", "import sys\n\nsys.exit(1)\n") == [
        "sys.exit outside sift.interfaces: raise a SiftError"
    ]


@pytest.mark.parametrize(
    "handler",
    [
        "except Exception:\n    if FLAG:\n        raise\n",
        "except RuntimeError as error:\n    if not mine(error):\n        raise\n    print(error)\n",
    ],
)
def test_recursion_raised_again_only_sometimes_is_caught(handler: str) -> None:
    assert problems("src/sift/app/a.py", f"try:\n    g()\n{handler}") == [check_idioms.RECURSION_CAUGHT]


@pytest.mark.parametrize("pattern", [r"10\.\d+", r"\D", r"a\\\d"])
def test_a_pattern_matching_any_script_s_digits_is_found(pattern: str) -> None:
    source = f'import re\n\nDOI = re.compile(r"{pattern}")\nre.search(pattern=r"{pattern}", string="")\n'
    assert problems("src/sift/domain/a.py", source) == [check_idioms.UNICODE_DIGITS] * 2


def test_a_backslash_before_d_and_plain_digits_are_no_digit_class() -> None:
    assert problems("src/sift/domain/a.py", 'import re\n\nA = re.compile(r"\\\\d[0-9]")\n') == []


@pytest.mark.parametrize(
    "read",
    [
        "import nt\n\nX = nt.environ\n",
        "import posix\n\nX = posix.getenv('A')\n",
        "from nt import environ\n",
        "import os\n\nX = vars(os)['environ']\n",
        "import os\n\nX = os.__dict__['getenv']('A')\n",
    ],
)
def test_the_environment_read_another_way_is_found(read: str) -> None:
    assert problems("src/sift/app/a.py", read) == [check_idioms.ENVIRONMENT_READ]


@pytest.mark.parametrize(
    "patched",
    [
        "import sys\n\nsys.modules['sift.app.a'].f = 1\n",
        "import sys\n\nsys.modules['sift.app.a'] = object()\n",
        "import sys\n\nsetattr(sys.modules['sift.app'], 'f', 1)\n",
        "from importlib import import_module\n\nimport_module('sift.app.a').f = 1\n",
        "import importlib\n\ndel importlib.import_module('sift.app.a').f\n",
    ],
)
def test_sift_patched_through_the_module_table_is_found(patched: str) -> None:
    assert problems("tests/a.py", patched) == [check_idioms.PATCHED]


def test_other_modules_in_the_module_table_are_no_patching_of_sift() -> None:
    assert problems("tests/a.py", "import sys\n\nsys.modules['json'].f = 1\nsys.modules[name] = 1\n") == []


def test_a_text_cut_by_substring_in_sql_is_found() -> None:
    cut = 'SQL = "SELECT substring(t.text, 1, 5) FROM texts AS t"\n'
    assert problems("src/sift/storage/a.py", cut) == [idioms_storage.TEXT_CUT_IN_SQL]


def test_no_property_test_pins_its_example_count() -> None:
    pinned = (
        "from hypothesis import settings\n\n\n@settings(max_examples=3, deadline=None)\ndef test_a() -> None: ...\n"
    )
    assert problems("tests/unit/test_a.py", pinned) == [check_idioms.PINNED_EXAMPLES]
    assert problems("tests/examples.py", pinned) == []
    widened = "from tests.examples import few\n\n\n@few(3)\ndef test_a() -> None: ...\n"
    assert problems("tests/unit/test_a.py", widened) == []


@pytest.mark.parametrize("dropped", ["thread.join(30)", "started.wait(30)", "threading.Event().wait(0.1)"])
def test_a_wait_in_a_test_whose_outcome_is_dropped_is_found(dropped: str) -> None:
    source = f"def test_a() -> None:\n    {dropped}\n"
    assert problems("tests/unit/test_a.py", source) == [check_idioms.WAIT_DROPPED]
    assert problems("tests/waits.py", source) == []
    assert problems("tests/unit/test_a.py", f"def test_a() -> None:\n    assert {dropped}\n") == []


@pytest.mark.parametrize("held", ["found", "found.hits", "found.hits[0]"])
def test_an_assertion_that_any_truthy_value_passes_is_found_in_tests(held: str) -> None:
    assert problems("tests/unit/test_a.py", f"assert {held}\n") == [check_idioms.TRUTHY]
    assert problems("tests/unit/test_a.py", f"assert {held} is True\nassert {held} == [1]\n") == []
    assert problems("src/sift/a.py", f"assert {held}\n") == []


def test_names_are_followed_in_each_function_alone() -> None:
    twice = "    moment = datetime.datetime\n    return moment.now()\n"
    source = f"import datetime\n\n\ndef a():\n{twice}\n\ndef b():\n{twice}"
    assert problems("src/sift/app/a.py", source) == [check_idioms.CLOCK_READ] * 2


def test_a_parameter_is_not_the_module_s_name_spelled_the_same() -> None:
    source = "import datetime\n\nmoment = datetime.datetime\n\n\ndef f(moment):\n    return moment.now()\n"
    assert problems("src/sift/app/a.py", source) == []
