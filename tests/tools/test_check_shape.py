import pytest

from tools import check_shape
from tools.check_shape import MOST_LINES, MOST_METHODS, shape


def problems(path: str, source: str) -> list[str]:
    return [finding.problem for finding in shape(path, source)]


def test_a_module_may_have_up_to_the_most_lines() -> None:
    assert problems("src/sift/a.py", "x = 1\n" * MOST_LINES) == []
    assert problems("src/sift/a.py", "x = 1\n" * (MOST_LINES + 1)) == [f"module of 401 lines, more than {MOST_LINES}"]


def test_a_class_may_have_up_to_the_most_methods() -> None:
    def source(n: int) -> str:
        return "class C:\n" + "".join(f"    def m{i}(self) -> None: ...\n" for i in range(n))

    assert problems("src/sift/a.py", source(MOST_METHODS)) == []
    assert problems("src/sift/a.py", source(MOST_METHODS + 1)) == ["class C has 11 methods, more than 10"]


@pytest.mark.parametrize(
    "value",
    ["[]", "{}", "{1}", "[x for x in y]", "{x: 1 for x in y}", "{x for x in y}", "dict()", "collections.deque()"],
)
def test_a_mutable_module_value_is_found(value: str) -> None:
    assert problems("src/sift/a.py", f"X = {value}\n") == ["mutable value at module level"]
    assert problems("tools/a.py", f"X: object = {value}\n") == ["mutable value at module level"]


@pytest.mark.parametrize("value", ["()", "frozenset({1})", "MappingProxyType({})", "re.compile('a')", "1", "f()"])
def test_an_immutable_module_value_passes(value: str) -> None:
    assert problems("src/sift/a.py", f"X = {value}\nY: int\n") == []


def test_values_inside_functions_and_in_tests_may_be_mutable() -> None:
    assert problems("src/sift/a.py", "def f() -> list[int]:\n    found = []\n    return found\n") == []
    assert problems("tests/a.py", "X = []\n") == []


@pytest.mark.parametrize(
    "source",
    [
        "@dataclass(frozen=True, slots=True)\nclass A:\n    x: int\n",
        "@dataclasses.dataclass(slots=True, frozen=True)\nclass A:\n    x: int\n",
        "class A(NamedTuple):\n    x: int\n",
        "class A(enum.StrEnum):\n    X = 'x'\n",
        "class A(Protocol[T]):\n    def f(self) -> None: ...\n",
        "class SiftError(Exception): ...\n",
        "class NotFound(errors.SiftError): ...\n",
    ],
)
def test_value_types_pass_in_the_core(source: str) -> None:
    assert problems("src/sift/domain/a.py", source) == []
    assert problems("src/sift/api/a.py", source) == []


@pytest.mark.parametrize(
    "source",
    [
        "class A:\n    x: int\n",
        "@dataclass\nclass A:\n    x: int\n",
        "@dataclass(frozen=True)\nclass A:\n    x: int\n",
        "@dataclass(frozen=True, slots=False)\nclass A:\n    x: int\n",
        "@total_ordering\nclass A(Base):\n    x: int\n",
        "@made()(frozen=True, slots=True)\nclass A(bases()[0]):\n    x: int\n",
        "def f() -> None:\n    class A:\n        x: int\n",
    ],
)
def test_other_classes_are_found_in_the_core_alone(source: str) -> None:
    assert problems("src/sift/domain/a.py", source) == ["class A is no frozen slotted dataclass or value type"]
    assert problems("src/sift/storage/a.py", source) == []


def test_the_repository_has_its_shape() -> None:
    assert check_shape.main([]) == 0


def test_a_dataclass_holding_arrays_compares_none() -> None:
    held = "@dataclass(frozen=True, slots=True)\nclass Held:\n    ids: Ids\n"
    assert problems("src/sift/a.py", held) == ["class Held holds arrays and compares them: eq=False"]
    assert problems("src/sift/a.py", held.replace("frozen=True", "eq=False, frozen=True")) == []
    assert problems("src/sift/a.py", "@dataclass(frozen=True, slots=True)\nclass Held:\n    n: int\n") == []
