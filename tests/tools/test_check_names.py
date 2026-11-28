import pytest

from tools.check_names import main, repeated


def problems(sources: dict[str, str]) -> list[tuple[str, int, str]]:
    return [(finding.path, finding.line, finding.problem) for finding in repeated(sources.items())]


def test_a_class_or_type_named_twice_is_found_where_it_is_named_again() -> None:
    sources = {
        "src/sift/a.py": "class Row:\n    pass\n\n\ntype Ids = list[int]\n",
        "src/sift/b.py": "x = 1\nclass Row:\n    pass\n",
        "src/sift/c.py": "def f() -> None:\n    type Ids = tuple[int, ...]\n",
    }
    assert problems(sources) == [
        ("src/sift/b.py", 2, "Row is defined in src/sift/a.py:1 too"),
        ("src/sift/c.py", 2, "Ids is defined in src/sift/a.py:5 too"),
    ]


def test_names_defined_once_pass() -> None:
    assert problems({"src/sift/a.py": "class Row:\n    pass\n", "src/sift/b.py": "type Rows = list[int]\n"}) == []


def test_the_sources_name_each_class_and_type_once(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out == ""
