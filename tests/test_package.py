"""The package as it is published: what it depends on, and the rules its modules keep."""

import ast
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
ENVIRONMENT = {"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"}


def test_every_package_sift_imports_is_a_dependency():
    # sift imported uvicorn, starlette, anyio, pydantic and markupsafe, which only mcp and jinja2 declared: a change
    # in their dependencies would have broken sift.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {re.match(r"[\w.-]+", d)[0].lower() for d in project["dependencies"]}
    imported = {
        name.split(".")[0]
        for path in (ROOT / "sift").glob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Import) or (isinstance(node, ast.ImportFrom) and not node.level)  # not `from . import`
        for name in ([a.name for a in node.names] if isinstance(node, ast.Import) else [node.module])
    }
    distributions = importlib.metadata.packages_distributions()
    undeclared = {
        module
        for module in imported - set(sys.stdlib_module_names)
        if not {d.lower() for d in distributions[module]} & declared
    }
    assert undeclared == set()


def test_only_settings_touches_the_environment():
    # vectors.py once set NUMBA_CACHE_DIR itself: settings.py alone reads and sets the environment, so every variable
    # sift heeds is in one place.
    touching = {
        f"{path.name}:{node.lineno}"
        for path in (ROOT / "sift").glob("*.py")
        if path.name != "settings.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if _touches_environment(node)
    }
    assert touching == set(), "read and set environment variables in settings.py, and ask it"


def _touches_environment(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
        return node.attr in ENVIRONMENT  # os.environ, os.getenv(...)
    if isinstance(node, ast.ImportFrom) and node.module == "os":
        return bool({a.name for a in node.names} & ENVIRONMENT)  # from os import environ
    return False


INVISIBLE = {"\x85", "\xa0", "\u200b", "\u2028", "\u2029", "\ufeff", "\ufffd"}


def test_sources_write_invisible_characters_as_escapes():
    # A U+FFFD or U+2028 written as itself looks like nothing, or like any other mark, in an editor and a diff: the
    # tests that hold them can't be read, and one slipped into code unseen.
    sources = [*(ROOT / "sift").glob("*.py"), *(ROOT / "tests").glob("*.py")]
    found = {
        f"{path.name}:{n}"
        for path in sources
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1)
        if INVISIBLE & set(line)
    }
    assert found == set(), r"write them as escapes, such as \ufffd"


MACHINE = re.compile(  # each written so that it doesn't match itself
    r"\b(?:R[T]X|G[T]X|GeF[o]rce|Quadr[o]|Ra[d]eon|Ryz[e]n|Threadr[i]pper|X[e]on|EP[Y]C)\b|\bCore i[3579]\b"
    r"|\bi[3579]-[0-9]{4,5}"
)


def test_published_text_names_no_machine():
    # Timings named the GPU and CPU models of the machine they were measured on, and what else ran on it: facts of
    # one setup, not of sift.
    texts = [
        *ROOT.glob("*.md"),
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / ".github").rglob("*.*"),
        *(ROOT / "sift").rglob("*.py"),
        *(ROOT / "tests").glob("*.py"),
    ]
    found = {
        f"{path.relative_to(ROOT)}:{n}"
        for path in texts
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1)
        if MACHINE.search(line)
    }
    assert found == set(), "state hardware by kind (a GPU, 8 CPU threads), not by the model of the machine measured"


def test_only_newlines_end_lines():
    # str.splitlines also ends a line at U+2028, U+2029, NEL and five control characters: it shifted search hits' line
    # numbers, cut front-matter values that JSON had written with them, and split .siftignore patterns in two.
    found = {
        f"{path.name}:{node.lineno}"
        for path in (ROOT / "sift").glob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr == "splitlines"
    }
    assert found == set(), 'split on "\n" (see readers.LINE): only it ends a line, as in editors and git'


def test_workflows_run_actions_by_commit():
    # A tag or a branch of an action can be moved to other code, and the release workflow publishes to PyPI: an
    # action is named by the commit reviewed, with its version beside it, which Dependabot keeps current.
    by_name = {
        f"{path.name}:{n}"
        for path in (ROOT / ".github" / "workflows").glob("*.yml")
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1)
        if "uses:" in line and not re.search(r"uses: [\w./-]+@[0-9a-f]{40}  # v\d+\.\d+\.\d+", line)
    }
    assert by_name == set(), "name the action by its commit: `uses: owner/action@<40 hex digits>  # v1.2.3`"
