import pytest

from tools import check_twins
from tools.check_twins import twins

KEPT = "def kept(items):\n    found = [each for each in items if each]\n    return sorted(found, key=len)\n"


def problems(sources: dict[str, str]) -> list[tuple[str, int, str]]:
    return [(finding.path, finding.line, finding.problem) for finding in twins(sources)]


def test_two_functions_that_differ_in_their_names_and_types_alone_are_found_where_the_second_is() -> None:
    sources = {
        "src/sift/a.py": KEPT,
        "src/sift/b.py": (
            "X = 1\n\n\ndef held(words: list[str]) -> list[str]:\n"
            "    some = [word for word in words if word]\n    return sorted(some, key=len)\n"
        ),
    }
    assert problems(sources) == [("src/sift/b.py", 4, "held does what kept (src/sift/a.py:1) does")]


def test_functions_that_call_or_read_other_names_or_are_decorated_otherwise_are_no_twins() -> None:
    sources = {
        "a.py": KEPT,
        "b.py": KEPT.replace("sorted", "reversed"),
        "c.py": KEPT.replace("key=len", "key=str"),
        "d.py": f"@cache\n{KEPT}",
        "e.py": KEPT.replace("(items)", "(items, more)"),
    }
    assert problems(sources) == []


def test_small_functions_alike_are_no_twins() -> None:
    assert problems({"a.py": "def a(x):\n    return x.name\n", "b.py": "def b(y):\n    return y.name\n"}) == []


def test_methods_and_functions_inside_functions_are_twins_of_any_other() -> None:
    inside = "def outer():\n" + "".join(f"    {line}\n" for line in KEPT.split("\n")[:-1]) + "    return kept\n"
    method = "class Held:\n" + "".join(f"    {line}\n" for line in KEPT.split("\n")[:-1])
    assert [problem for _, _, problem in problems({"a.py": KEPT, "b.py": inside, "c.py": method})] == [
        "kept does what kept (a.py:1) does",
        "kept does what kept (a.py:1) does",
    ]


LISTED = "def kept(items):\n    return sorted([each.name for each in items if each], key=len)\n"


@pytest.mark.parametrize(
    "first, second",
    [
        (KEPT.replace("key=len", "key=len, reverse=True"), KEPT.replace("key=len", "reverse=True, key=len")),
        (KEPT, KEPT.replace("def kept", "async def kept")),
        (KEPT, "import builtins as b\n\n\n" + KEPT.replace("sorted(", "b.sorted(")),
        (
            "from os import path\n\n\n" + KEPT.replace("[each for", "[path.join(each) for"),
            "import os.path as p\n\n\n" + KEPT.replace("[each for", "[p.join(each) for"),
        ),
        (KEPT, KEPT.replace("(items)", "(items=())")),
        (LISTED, "kept = lambda items: sorted([each.name for each in items if each], key=len)\n"),
    ],
    ids=["keywords reordered", "async", "a builtin through an alias", "an import alias", "a default", "a lambda"],
)
def test_twins_written_another_way_are_found(first: str, second: str) -> None:
    assert len(problems({"a.py": first, "b.py": second})) == 1


def test_functions_reading_each_their_own_module_s_value_are_no_twins() -> None:
    home = "def home(path):\n    return made(path / 'home', NOTES, embedded=Model(1, 2))\n"
    own, shared = f"NOTES = {{}}\n\n\n{home}", f"from tests.notes import NOTES\n\n\n{home}"
    assert problems({"a.py": own, "b.py": own}) == []
    assert len(problems({"a.py": shared, "b.py": shared})) == 1


def test_the_repository_holds_no_twins() -> None:
    assert check_twins.main([]) == 0
