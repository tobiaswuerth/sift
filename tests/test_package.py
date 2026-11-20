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


def test_the_readme_lists_every_variable_sift_heeds():
    # settings.py counted six variables, its tests five: the README's table is where they are listed, all of them.
    source = ast.parse((ROOT / "sift" / "settings.py").read_text(encoding="utf-8"))
    heeded = {n.value for n in ast.walk(source) if isinstance(n, ast.Constant) and str(n.value).startswith("SIFT_")}
    listed = set(re.findall(r"^\| `(SIFT_\w+)` \|", (ROOT / "README.md").read_text(encoding="utf-8"), re.MULTILINE))
    assert heeded == listed, "list each SIFT_ variable settings.py reads in the README's table, and only those"


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


NOT_YET_STRICT = {  # the modules the type checker doesn't read yet. It only shrinks: modules leave it, none joins it
    "sift/__main__.py",
    "sift/accelerator.py",
    "sift/bench.py",
    "sift/catalog.py",
    "sift/cli.py",
    "sift/embed.py",
    "sift/indexing.py",
    "sift/jobs.py",
    "sift/links.py",
    "sift/local.py",
    "sift/models.py",
    "sift/ocr.py",
    "sift/order.py",
    "sift/postings.py",
    "sift/readers.py",
    "sift/remote.py",
    "sift/scan.py",
    "sift/search.py",
    "sift/server.py",
    "sift/store.py",
    "sift/topk.py",
    "sift/ui.py",
    "sift/upgrade.py",
    "sift/vectors.py",
    "sift/worker.py",
}
SILENCED = re.compile(r"#\s*(?:type:\s*ignore|pyright:)")


def _pyright():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pyright"]


def test_the_modules_checked_for_types_only_grow():
    # The type checker came to code written without it, some modules at a time: those it checks ([tool.pyright] in
    # pyproject.toml) are all but the ones still waiting, so a new module is checked from its start, and one checked
    # stays so.
    modules = {f"sift/{path.name}" for path in (ROOT / "sift").glob("*.py")}
    assert modules >= NOT_YET_STRICT, "a module that is gone waits no more: take it out of NOT_YET_STRICT"
    assert set(_pyright()["include"]) == modules - NOT_YET_STRICT, (
        "list a new module under include in [tool.pyright]; one that became strict there, and no more in "
        "NOT_YET_STRICT, which only shrinks"
    )


def test_nothing_silences_the_type_checker():
    # Green by a rule switched off, or by a line the checker is told to pass over, says nothing: its settings are
    # the strict ones and no more, for the oldest Python sift runs on, and no module talks it out of a finding.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    settings = _pyright()
    assert sorted(settings) == ["include", "pythonVersion", "typeCheckingMode"], "no rule is set apart from strict"
    assert settings["typeCheckingMode"] == "strict"
    assert project["requires-python"] == f">={settings['pythonVersion']}"
    silenced = {
        f"{path.name}:{n}"
        for path in (ROOT / "sift").glob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1)
        if SILENCED.search(line)
    }
    assert silenced == set(), "fix what the checker finds; a rule silenced in one place needs the owner's yes first"


def test_chunks_are_cut_from_their_text_in_one_place():
    # Eight statements cut each chunk out of its document's text with SQLite's substr, or read the view that does: it
    # loads the whole text and walks to the position for every chunk, 20 s for the 3,310 chunks of an 8 MB file. The
    # view keeps the cut, as what the keyword index holds, and the store reads it for short documents alone
    # (store.fill); everything else cuts from a text read once.
    lines = [
        (path.name, line)
        for path in (ROOT / "sift").glob("*.py")
        for line in path.read_text(encoding="utf-8").split("\n")
    ]
    assert [name for name, line in lines if "c.pos + 1, c.len)" in line] == ["store.py"], "ask store.texts"
    assert [name for name, line in lines if "FROM chunk_text" in line] == ["store.py"], "ask store.fill"


PRINTING = {("console.py", "write"), ("cli.py", "_print")}  # to the user, and the results


def _prints(tree):
    return {
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    }


def test_only_the_console_and_the_results_print():
    # print() takes a stream that is None for stdout. Under pythonw, which has no stderr, what three modules printed
    # to stderr themselves went among the results: one function writes for the user, and knows of that.
    elsewhere = set()
    for path in (ROOT / "sift").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        allowed = {
            call
            for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef) and (path.name, function.name) in PRINTING
            for call in _prints(function)
        }
        elsewhere |= {f"{path.name}:{call.lineno}" for call in _prints(tree) - allowed}
    assert elsewhere == set(), "say it with console.say, or write it with console.write; results go through cli._print"


def test_sql_placeholders_have_no_numbers():
    # Python 3.12.0 to 3.12.3 take a placeholder with a number for a named one, and warn that binding it from a
    # sequence becomes an error: a test that bound one failed on the one system whose Python was 3.12.3.
    numbered = re.compile(r"\?[0-9]")
    sources = [path for folder in ("sift", "tests", "scripts") for path in (ROOT / folder).glob("*.py")]
    found = {
        f"{path.name}:{node.lineno}"
        for path in sources
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and numbered.search(node.value)
    }
    assert found == set(), "bind by name, as `:path` from a dict, or give each plain `?` its own value"


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
