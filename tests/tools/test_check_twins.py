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


def test_the_repository_holds_no_twins() -> None:
    assert check_twins.main([]) == 0
