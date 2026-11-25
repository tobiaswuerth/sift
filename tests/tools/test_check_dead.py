from tools import check_dead
from tools.check_dead import dead, entry_points, module_of

NOTHING_KEPT: frozenset[str] = frozenset()


def unused(sources: dict[str, str], kept: frozenset[str] = NOTHING_KEPT) -> list[str]:
    return [finding.problem.split(" ", 1)[0] for finding in dead(sources, kept)]


def places(sources: dict[str, str], kept: frozenset[str] = NOTHING_KEPT) -> list[str]:
    return [finding.path for finding in dead(sources, kept)]


def test_a_module_is_named_by_its_path() -> None:
    assert (module_of("src/sift/app/search.py"), module_of("src/sift/app/__init__.py")) == (
        "sift.app.search",
        "sift.app",
    )


def test_a_name_nothing_imports_or_reads_is_found() -> None:
    sources = {
        "src/sift/__main__.py": "import sift.b\n",
        "src/sift/a.py": (
            "def used() -> int:\n    return HELD\n\n\ndef unused() -> None: ...\n\n\nHELD = 1\nREAD = HELD\n"
        ),
        "src/sift/b.py": "from sift.a import used\n\nused()\n",
    }
    assert unused(sources) == ["unused", "READ"]


def test_a_name_read_through_a_module_alias_is_used() -> None:
    sources = {
        "src/sift/__main__.py": "import sift.b\n",
        "src/sift/a.py": "X = 1\n",
        "src/sift/b.py": "import sift.a as held\n\nprint(held.X)\n",
    }
    assert unused(sources) == []


def test_a_name_of_another_module_with_the_same_name_keeps_nothing_alive() -> None:
    sources = {
        "src/sift/__main__.py": "import sift.a\nimport sift.b\n",
        "src/sift/a.py": "def search() -> None: ...\n",
        "src/sift/b.py": "def search() -> None: ...\n\n\nsearch()\n",
    }
    assert places(sources) == ["src/sift/a.py"]


def test_entry_points_the_script_and_the_surface_are_roots() -> None:
    sources = {
        "src/sift/c.py": (
            "def main() -> None:\n    _own()\n\n\ndef _own() -> None: ...\n\n\n"
            "def helper() -> None: ...\n\n\ndef scripted() -> None: ...\n"
        ),
        "src/sift/api/s.py": "import sift.c\n\n\ndef offered() -> None:\n    sift.c.helper()\n",
        "src/sift/__main__.py": "from sift.c import scripted\n\nscripted()\n",
    }
    assert unused(sources, frozenset({"sift.c.main"})) == []


def test_a_protocol_member_keeps_alive_only_the_module_loaded_through_it() -> None:
    loader = (
        'from typing import TYPE_CHECKING, Protocol\n\nfrom sift.late import late\n\nLOADED = "sift.q"\n\n\n'
        "class Loaded(Protocol):\n    def run(self) -> None: ...\n\n\n"
        "def main() -> None:\n    late(LOADED, Loaded).run()\n\n\n"
        "if TYPE_CHECKING:\n    from sift import q as _q\n\n    _Q: Loaded = _q\n"
    )
    sources = {
        "src/sift/p.py": loader,
        "src/sift/q.py": "def run() -> None: ...\n",
        "src/sift/r.py": "def run() -> None: ...\n",
    }
    assert places(sources, frozenset({"sift.p.main"})) == ["src/sift/r.py"]


def test_a_protocol_holds_the_members_of_the_protocols_it_extends() -> None:
    loader = (
        "from typing import TYPE_CHECKING, Protocol\n\n\n"
        "class Closing(Protocol):\n    def close(self) -> None: ...\n\n\n"
        "class Loaded(Closing, Protocol):\n    NAME: str\n\n\n"
        "if TYPE_CHECKING:\n    import sift.q\n\n    _Q: Loaded = sift.q\n"
    )
    sources = {
        "src/sift/__main__.py": "import sift.p\n",
        "src/sift/p.py": loader,
        "src/sift/q.py": 'NAME = "q"\n\n\ndef close() -> None: ...\n\n\ndef opened() -> None: ...\n',
    }
    assert unused(sources) == ["opened"]


def test_a_class_named_like_a_protocol_is_none() -> None:
    sources = {
        "src/sift/__main__.py": "import sift.p\n",
        "src/sift/p.py": (
            "import sift.q\nfrom h11 import Protocol as H11Protocol\n\n\n"
            "class Heads(H11Protocol):\n    def run(self) -> None: ...\n\n\n"
            "_Q: Heads = sift.q\nprint(_Q)\n"
        ),
        "src/sift/q.py": "def run() -> None: ...\n",
    }
    assert unused(sources) == ["run"]


def test_functions_that_only_call_each_other_are_dead() -> None:
    sources = {
        "src/sift/__main__.py": "import sift.a\n",
        "src/sift/a.py": (
            "def ping() -> None:\n    pong()\n\n\ndef pong() -> None:\n    ping()\n\n\n"
            "def again() -> None:\n    again()\n"
        ),
    }
    assert unused(sources) == ["ping", "pong", "again"]


def test_an_unused_private_module_value_is_dead() -> None:
    sources = {
        "src/sift/__main__.py": "import sift.a\n",
        "src/sift/a.py": '__all__ = ["WHY"]\n_WHY = "because"\n_READ = 1\nprint(_READ)\n',
    }
    assert unused(sources) == ["_WHY"]


def test_a_module_s_own_statements_run_only_when_something_imports_it() -> None:
    module = "def started() -> None: ...\n\n\nstarted()\n"
    assert unused({"src/sift/a.py": module}) == ["started"]
    assert unused({"src/sift/a.py": module, "src/sift/__main__.py": "import sift.a\n"}) == []


def test_a_name_imported_for_type_checking_alone_is_used() -> None:
    sources = {
        "src/sift/__main__.py": "import sift.b\n",
        "src/sift/a.py": "class Shape: ...\n",
        "src/sift/b.py": (
            "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from sift.a import Shape\n\n\n"
            'def drawn(shape: "Shape") -> None: ...\n\n\ndrawn(None)\n'
        ),
    }
    assert unused(sources) == []


def test_a_name_in_a_quoted_annotation_or_cast_is_used() -> None:
    module = (
        "from typing import Annotated, Literal, cast\n\n\nclass _Kind: ...\n\n\nclass _Other: ...\n\n\n"
        'def made(kind: "list[_Kind]", tag: Literal["a b"], said: Annotated[int, "a b c"]) -> "_Other | None":\n'
        '    return cast("_Other", kind)\n\n\nasync def waited(kind: "_Kind"): ...\n\n\n'
        'made([], "a b", 1)\nprint(waited)\n'
    )
    assert unused({"src/sift/__main__.py": "import sift.a\n", "src/sift/a.py": module}) == []


def test_a_name_offered_again_by_a_package_is_used_through_it() -> None:
    sources = {
        "src/sift/__main__.py": "from sift.p import shown\n\nshown()\n",
        "src/sift/p/__init__.py": "from sift.p.inner import shown\n",
        "src/sift/p/inner.py": "def shown() -> None: ...\n\n\ndef hidden() -> None: ...\n",
    }
    assert unused(sources) == ["hidden"]


def test_entry_points_are_read_from_the_project_s_scripts() -> None:
    pyproject = '[project.scripts]\nsift = "sift.interfaces.cli.main:main"\n'
    assert entry_points(pyproject) == {"sift.interfaces.cli.main.main"}
    assert entry_points("[project]\n") == frozenset()


def test_the_repository_holds_no_unused_name() -> None:
    assert check_dead.main([]) == 0
