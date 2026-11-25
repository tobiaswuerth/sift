import pytest

from tests.tools.idiom_problems import problems
from tools import check_idioms

TLS_MADE = (
    "import ssl\n\n\ndef made() -> ssl.SSLContext:\n"
    "    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)\n{floor}    return tls\n"
)
TLS_FLOORED_ELSEWHERE = (
    "\n\ndef floored(tls: ssl.SSLContext) -> None:\n    tls.minimum_version = ssl.TLSVersion.TLSv1_3\n"
)


def lines_found(source: str) -> list[int]:
    return [finding.line for finding in check_idioms.idioms("tests/a.py", source)]


@pytest.mark.parametrize(
    "floor",
    ["", "    tls.minimum_version = ssl.TLSVersion.TLSv1_1\n", "    tls.maximum_version = ssl.TLSVersion.TLSv1_3\n"],
)
def test_a_tls_context_made_without_a_safe_oldest_version_is_found_with_its_fix(floor: str) -> None:
    for path in ("src/sift/a.py", "tests/a.py", "tools/a.py"):
        (problem,) = problems(path, TLS_MADE.format(floor=floor))
        assert problem.startswith("a TLS context made without the oldest version it takes")
        assert "minimum_version = ssl.TLSVersion.TLSv1_2" in problem


@pytest.mark.parametrize("version", ["TLSv1_2", "TLSv1_3"])
def test_a_tls_context_given_its_oldest_version_where_it_is_made_passes(version: str) -> None:
    floor = f"    tls.minimum_version = ssl.TLSVersion.{version}\n"
    assert problems("tests/a.py", TLS_MADE.format(floor=floor)) == []


def test_a_tls_context_is_given_its_oldest_version_by_the_function_that_makes_it_not_by_another() -> None:
    assert lines_found(TLS_MADE.format(floor="") + TLS_FLOORED_ELSEWHERE) == [5]
    nested = (
        "import ssl\n\n\ndef outer() -> None:\n    def made() -> ssl.SSLContext:\n        return ssl.SSLContext()\n\n"
        "    made().minimum_version = ssl.TLSVersion.TLSv1_3\n"
    )
    assert lines_found(nested) == [6]


def test_a_tls_context_made_by_a_module_is_given_its_oldest_version_by_the_module() -> None:
    made = "from ssl import PROTOCOL_TLS_CLIENT, SSLContext\nimport ssl\n\nTLS = SSLContext(PROTOCOL_TLS_CLIENT)\n"
    assert lines_found(made + TLS_FLOORED_ELSEWHERE) == [4]
    assert lines_found(made + "TLS.minimum_version = ssl.TLSVersion.TLSv1_2\n") == []


@pytest.mark.parametrize("called", ["View[int, str](a, b)", "_Worker[A](made)", "kept.View[int](a)", "Box[int]()"])
def test_a_generic_class_called_with_its_type_arguments_is_found(called: str) -> None:
    for path in ("src/sift/app/a.py", "tests/a.py", "tools/a.py"):
        (problem,) = problems(path, f"x = {called}\n")
        assert problem.startswith("a generic class called with its type arguments")


@pytest.mark.parametrize(
    "called",
    [
        "list[str]()",
        "dict[str, int]()",
        "set[int]()",
        "frozenset[int]()",
        "tuple[int, ...]()",
        "collections.OrderedDict[str, int]()",
        "collections.defaultdict[str, list[int]](list)",
        "collections.deque[int]()",
        "View(a, b)",
        "views[0](a)",
        "CHECKS[name](argv)",
        "_KINDS[kind](a)",
    ],
)
def test_a_builtin_container_called_with_its_type_arguments_or_a_plain_call_is_no_problem(called: str) -> None:
    assert problems("src/sift/app/a.py", f"import collections\nx = {called}\n") == []
