from tools import check_dead
from tools.check_dead import dead, entry_points, module_of

NOTHING_KEPT: frozenset[str] = frozenset()


def unused(sources: dict[str, str], kept: frozenset[str] = NOTHING_KEPT) -> list[str]:
    return [finding.problem.split(" ", 1)[0] for finding in dead(sources, kept)]


def test_a_module_is_named_by_its_path() -> None:
    assert (module_of("src/sift/app/search.py"), module_of("src/sift/app/__init__.py")) == (
        "sift.app.search",
        "sift.app",
    )


def test_a_name_nothing_imports_or_reads_is_found() -> None:
    sources = {
        "src/sift/a.py": "def used() -> None: ...\n\n\ndef unused() -> None: ...\n\n\nHELD = 1\nREAD = HELD\n",
        "src/sift/b.py": "from sift.a import used\n\nused()\n",
    }
    assert unused(sources) == ["unused", "READ"]


def test_a_name_read_through_a_module_alias_is_used() -> None:
    sources = {"src/sift/a.py": "X = 1\n", "src/sift/b.py": "import sift.a as held\n\nprint(held.X)\n"}
    assert unused(sources) == []


def test_a_name_of_another_module_with_the_same_name_keeps_nothing_alive() -> None:
    sources = {
        "src/sift/a.py": "def search() -> None: ...\n",
        "src/sift/b.py": "def search() -> None: ...\n\n\nsearch()\n",
    }
    assert unused(sources) == ["search"]


def test_protocol_members_entry_points_the_surface_and_private_names_are_kept() -> None:
    sources = {
        "src/sift/p.py": "from typing import Protocol\n\n\nclass P(Protocol):\n    RUN: int\n\n\nx: P\nprint(x)\n",
        "src/sift/c.py": "RUN = 1\n\n\ndef main() -> None: ...\n\n\ndef _own() -> None: ...\n",
        "src/sift/api/s.py": "def offered() -> None: ...\n",
    }
    assert unused(sources, frozenset({"sift.c.main"})) == []


def test_entry_points_are_read_from_the_project_s_scripts() -> None:
    pyproject = '[project.scripts]\nsift = "sift.interfaces.cli.main:main"\n'
    assert entry_points(pyproject) == {"sift.interfaces.cli.main.main"}
    assert entry_points("[project]\n") == frozenset()


def test_the_repository_holds_no_unused_name() -> None:
    assert check_dead.main([]) == 0
