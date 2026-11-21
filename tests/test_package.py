"""The package as it is published: what it depends on, and the rules its modules keep."""

import ast
import importlib
import importlib.metadata
import inspect
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
    by_name = {
        f"{path.name}:{node.lineno}"
        for path in (ROOT / "sift").glob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in ("import_module", "__import__")
    }
    assert by_name == set(), "import a package with an import statement: one imported by name isn't seen above"


def test_the_stubs_name_what_their_packages_have():
    # typings/ types what sift calls of packages that ship no types: a name or a parameter there that the package
    # doesn't have (any more) would pass the type checker and fail when called.
    for stub in (ROOT / "typings").rglob("*.pyi"):
        package = importlib.import_module(stub.parent.name if stub.stem == "__init__" else stub.stem)
        for node in ast.parse(stub.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef):
                _assert_takes(getattr(package, node.name), node, f"{stub.name}: {node.name}")
            for method in node.body if isinstance(node, ast.ClassDef) else ():
                real = getattr(getattr(package, node.name), method.name)
                _assert_takes(real, method, f"{stub.name}: {node.name}.{method.name}")


def _assert_takes(real, stubbed, where):
    """The function `real` takes every parameter its stub names, but those of methods Python calls itself."""
    if not stubbed.name.startswith("__") or stubbed.name == "__init__":
        named = {a.arg for a in stubbed.args.args} - {"self"}
        assert named <= set(inspect.signature(real).parameters), where


def test_requirements_txt_names_the_dependencies_pyproject_does():
    # pip users install from requirements.txt, uv users from pyproject.toml: two lists of the same packages, and
    # nothing held them equal. One ahead of the other installs a sift that fails at its first import.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").split("\n")
    listed = {line.strip() for line in lines if line.strip() and not line.startswith(("#", "--"))}
    assert listed == set(project["dependencies"]), "name each dependency in both, with the same versions"


def test_every_way_to_install_cuda_pytorch_names_one_index():
    # The index of PyTorch's CUDA builds stood in four places, and went stale in all: cu128 carries no PyTorch since
    # 2.10, so the fix sift prints, the README's and requirements.txt installed none at or above sift's floor.
    from sift import accelerator

    index = accelerator.CUDA_INDEX
    assert index in accelerator.cuda_pytorch_command()
    assert f"--extra-index-url {index}\n" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert set(re.findall(r"https://download\.pytorch\.org/whl/\w+", readme)) == {index}


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


UNWRAPPED = re.compile(r"\||!\[|\[!\[|\[[^\]]*\]: ")  # a table's row, an image, a badge, a link's address


def test_prose_in_the_docs_is_wrapped_as_the_code_is():
    # A clause added to a paragraph left its line at 151 characters, in a file wrapped at 120 like the code.
    line_length = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["ruff"]["line-length"]
    long = set()
    for path in [*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md"), *(ROOT / ".github").glob("*.md")]:
        in_code = False
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            in_code ^= line.lstrip().startswith("```")
            if len(line) > line_length and not in_code and not UNWRAPPED.match(line):
                long.add(f"{path.relative_to(ROOT)}:{n}")
    assert long == set(), f"wrap prose at {line_length} characters; only tables, code, images and addresses run on"


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


def test_no_comment_makes_the_formatter_wrap_its_statement():
    # A comment after a statement counts toward its line's length. Too long together, the formatter wraps the
    # statement around the comment: an annotation over three lines, a value alone in parentheses, where the statement
    # fits on one line and the comment belongs above it.
    toml = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    width = toml["tool"]["ruff"]["line-length"]
    wrapped = set()
    for path in (path for folder in ("sift", "scripts", "tests") for path in (ROOT / folder).glob("*.py")):
        lines = path.read_text(encoding="utf-8").split("\n")
        for n, (opened, held, closed) in enumerate(zip(lines, lines[1:], lines[2:]), 1):
            (held, _, said), (closed, _, after) = held.partition("  # "), closed.partition("  # ")
            if (
                opened.endswith(("(", "["))
                and closed.lstrip().startswith((")", "]"))
                and not held.rstrip().endswith(",")  # a comma the formatter keeps the statement wrapped for
                and (said or after)
                and len(opened + held.strip() + closed.strip()) <= width
            ):
                wrapped.add(f"{path.relative_to(ROOT).as_posix()}:{n}")
    assert wrapped == set(), "put the comment on a line of its own, above the statement, which then fits on one"


SILENCED = re.compile(r"#\s*(?:type:\s*ignore|pyright:)")
# The findings silenced with the owner's yes, each for one rule in one place, saying why: the module, and its line.
SILENCED_WITH_YES = {
    # Numba's @intrinsic: compiled code calls it with the word alone, Numba passing its typing context first
    (
        "scan.py",
        "distance += _popcount(words[i, k] ^ query[k])  # pyright: ignore[reportCallIssue]  # numba gives typingctx",
    ),
}


def _pyright():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pyright"]


def test_every_module_is_checked_for_types():
    # The type checker came to code written without it, some modules at a time, until it read all of them: a new
    # module is checked from its start.
    assert _pyright()["include"] == ["sift"]


def test_types_are_checked_for_the_python_whose_packages_they_read():
    # Checked for 3.11 with the packages of 3.12, every NumPy call was an error: NumPy 2.5 needs 3.12, and its types
    # name what only 3.12 has. pyright checks for the Python of CI's types job, whose packages it reads; that sift
    # still runs on the oldest Python it claims, ruff's target and CI's tests there, at full coverage, hold.
    toml = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    oldest = toml["project"]["requires-python"].removeprefix(">=")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert f'python-version: "{_pyright()["pythonVersion"]}"' in _job(ci, "types")
    assert toml["tool"]["ruff"]["target-version"] == "py" + oldest.replace(".", "")
    assert f'python: ["{oldest}",' in _job(ci, "test")
    assert toml["tool"]["coverage"]["report"]["fail_under"] == 100


def _job(workflow, name):
    """The text of a workflow's job."""
    return re.split(r"\n  (?=[\w-]+:\n)", workflow.split(f"\n  {name}:\n", 1)[1])[0]


def test_ci_looks_at_two_pythons_before_it_runs_every_system_and_python():
    # A test that left a database open failed on Python 3.13 and newer alone: twenty jobs ran to say it. The oldest
    # and the newest Python run first, on one system; the other jobs of the matrix only once those, the hooks and the
    # types pass, all under the names the matrix gave its jobs, which main requires.
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    first, rest = _job(ci, "first"), _job(ci, "test")
    pythons = re.search(r"python: \[(.*)\]", rest)[1].split(", ")
    assert f"python: [{pythons[0]}, {pythons[-1]}]" in first
    assert "os: [ubuntu-latest]" in first
    assert "needs: [lint, types, first]" in rest
    for python in (pythons[0], pythons[-1]):
        assert f"- {{os: ubuntu-latest, python: {python}}}" in rest  # left out there: `first` ran it
    for job in (first, rest):
        assert "name: test (${{ matrix.os }}, ${{ matrix.python }})" in job
    assert "steps: *test" in rest  # the steps of `first`, not a copy of them
    summary = _job(ci, "tests")
    assert "needs: [first, test]" in summary  # one check for all of them
    assert 'test "$FIRST" = success && test "$REST" = success' in summary


def test_nothing_silences_the_type_checker():
    # Green by a rule switched off, or by a line the checker is told to pass over, says nothing: its settings are
    # the strict ones and no more, and no module talks it out of a finding.
    settings = _pyright()
    assert sorted(settings) == ["include", "pythonVersion", "typeCheckingMode"], "no rule is set apart from strict"
    assert settings["typeCheckingMode"] == "strict"
    silenced = {
        (path.name, line.strip())
        for path in (ROOT / "sift").glob("*.py")
        for line in path.read_text(encoding="utf-8").split("\n")
        if SILENCED.search(line)
    }
    assert silenced == SILENCED_WITH_YES, (
        "fix what the checker finds; a rule silenced in one place needs the owner's yes first, then SILENCED_WITH_YES"
    )


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


def test_digits_are_matched_as_ascii_digits():
    # str.isdigit takes "²" and "٣" for digits, which int() refuses: a version "1.2.²" read from PyPI's answer
    # failed every command. Digits are [0-9].
    sources = [path for folder in ("sift", "scripts") for path in (ROOT / folder).glob("*.py")]
    found = {
        f"{path.name}:{node.lineno}"
        for path in sources
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr in ("isdigit", "isdecimal", "isnumeric")
    }
    assert found == set(), 'match digits as re.fullmatch("[0-9]+", text)'


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


def test_the_release_publishes_only_a_version_it_knows_is_new():
    # `git ls-remote --exit-code` says 2 for no such tag, and 128 for a failure, a network gone: any failure was taken
    # for a new version, and the release went on to publish one PyPI already had, or to tag a commit twice.
    check = _job((ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"), "check")
    assert "> /dev/null || status=$?" in check
    assert '2) echo "v$version is new."' in check
    assert 'exit "$status"' in check


def test_the_package_is_built_by_tools_reviewed():
    # The release built with whatever setuptools, twine and uv were newest that day: what PyPI got was made by code
    # nobody had looked at. Both builds take the backend by hash, twine by version, and the release its uv by version.
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "- run: uv build --build-constraints .github/requirements-build.txt --require-hashes\n" in release
    twine = re.search(r"- run: uvx (twine@\d+\.\d+\.\d+) check", release)
    assert twine
    assert not re.search(r"uv build\s*$|uvx twine check", release, re.MULTILINE)
    assert re.search(r'version: "\d+\.\d+\.\d+"', _job(release, "build"))
    package = (ROOT / "scripts" / "package.py").read_text(encoding="utf-8")  # CI's package job's, and the gate's
    assert '"--build-constraints", str(ROOT / ".github" / "requirements-build.txt")]' in package
    assert '"--require-hashes"' in package
    assert f'TWINE = "{twine[1]}"' in package
    pinned = (ROOT / ".github" / "requirements-build.txt").read_text(encoding="utf-8")
    assert re.search(r"^setuptools==\S+ \\$", pinned, re.MULTILINE)
    assert "--hash=sha256:" in pinned


# The findings of zizmor silenced with the owner's yes, each for one audit on one line: the workflow, and the line.
ZIZMOR_IGNORED_WITH_YES = {
    # The release starts when CI has ended (workflow_run), with what it takes to publish: the test below holds what
    # keeps a run that isn't main's own from starting it, which zizmor can't see.
    ("release.yml", "workflow_run:  # zizmor: ignore[dangerous-triggers]"),
}


def test_the_release_starts_only_for_ci_of_this_repository_s_main():
    # A workflow started by another's end runs with this repository's rights, whoever's run that was: a pull
    # request's, from a fork, would publish. Three conditions keep the release to CI that passed, on a push, to this
    # repository; every job waits for the one that checks them; and nothing of that run is used but its commit, never
    # in a command's text, and no file it made.
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    check = _job(release, "check")
    for condition in (
        "github.event.workflow_run.conclusion == 'success'",
        "&& github.event.workflow_run.event == 'push'",
        "&& github.event.workflow_run.head_repository.full_name == github.repository",
    ):
        assert condition in check.split("runs-on:")[0], condition
    jobs = re.findall(r"\n  ([\w-]+):\n", release.split("\njobs:\n", 1)[1].join(["\n", ""]))
    assert jobs[0] == "check"
    waits = {job: set(re.findall(r"[\w-]+", re.search(r"needs: (.*)", _job(release, job))[1])) for job in jobs[1:]}
    assert all(needed & {"check", *(j for j in waits if "check" in waits[j])} for needed in waits.values()), waits
    taken = set(re.findall(r"github\.event\.workflow_run\.([\w.]+)", release))
    assert taken == {"conclusion", "event", "head_repository.full_name", "head_sha"}
    for line in release.split("\n"):
        if "workflow_run.head_sha" in line:
            assert line.strip().startswith(("ref: ", "COMMIT: ")), line  # an input or a variable, no command's text
    assert "run-id" not in release  # the files downloaded are those this run built
    assert "artifacts_url" not in release
    ignored = {
        (path.name, line.strip())
        for path in (ROOT / ".github").rglob("*.yml")
        for line in path.read_text(encoding="utf-8").split("\n")
        if "zizmor:" in line
    }
    assert ignored == ZIZMOR_IGNORED_WITH_YES, "fix what zizmor finds; an audit silenced needs the owner's yes first"
    assert "- id: zizmor" in (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")


# Each repository of hooks in .pre-commit-config.yaml, and what docs/CONTRIBUTING.md calls what its hooks check.
HOOKS_AS_CONTRIBUTING_NAMES_THEM = {
    "local": "the `private` hook",
    "https://github.com/pre-commit/pre-commit-hooks": "whitespace, line endings, byte order marks, YAML/TOML checks",
    "https://github.com/astral-sh/ruff-pre-commit": "ruff (lint and format)",
    "https://github.com/abravalheri/validate-pyproject": "`validate-pyproject`",
    "https://github.com/python-jsonschema/check-jsonschema": "workflow and Dependabot schemas",
    "https://github.com/Mateusz-Grzelinski/actionlint-py": "actionlint",
    "https://github.com/codespell-project/codespell": "codespell",
    "https://github.com/zizmorcore/zizmor-pre-commit": "zizmor",
}


def test_contributing_names_every_hook():
    # A hook was added, zizmor, and the list of what runs on every commit stayed as it was: a contributor read of
    # checks that weren't all of them.
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert set(re.findall(r"- repo: (\S+)", config)) == set(HOOKS_AS_CONTRIBUTING_NAMES_THEM), (
        "name a new hook in docs/CONTRIBUTING.md, under how changes are made, and here"
    )
    contributing = " ".join((ROOT / "docs" / "CONTRIBUTING.md").read_text(encoding="utf-8").split())
    for repository, named in HOOKS_AS_CONTRIBUTING_NAMES_THEM.items():
        assert named in contributing, repository


def _imports(path, anywhere):
    """The modules of sift that the module at `path` imports: at its top, or `anywhere`, inside its functions too."""

    def found(node, top):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ImportFrom) and child.level == 1 and (top or anywhere):
                yield from [child.module.split(".")[0]] if child.module else [name.name for name in child.names]
            yield from found(
                child, top and not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            )

    modules = {other.stem for other in (ROOT / "sift").glob("*.py")}
    return set(found(ast.parse(path.read_text(encoding="utf-8")), True)) & modules - {path.stem}


def _cycles(imports):
    """The cycles among modules, each as its modules in order of name, given what each imports."""
    cycles = set()

    def follow(module, path):
        for other in imports[module]:
            if other in path:
                cycles.add(tuple(sorted(path[path.index(other) :])))
            else:
                follow(other, [*path, other])

    for module in imports:
        follow(module, [module])
    return cycles


CYCLES = {("catalog", "upgrade"), ("server", "ui")}  # through imports inside functions. It only shrinks


def test_modules_import_each_other_one_way():
    # Nothing held the direction of sift's imports: what imports what was known by reading all of it. At their tops
    # the modules form no cycle, so each can be imported, and read, before those that use it; inside functions two
    # pairs still import each other, and no third may.
    assert _cycles({"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}) == {("a", "b", "c")}  # the check finds one
    sources = sorted((ROOT / "sift").glob("*.py"))
    assert _cycles({path.stem: _imports(path, anywhere=False) for path in sources}) == set(), (
        "import the other module inside the function that needs it, or move what both need into a third"
    )
    assert _cycles({path.stem: _imports(path, anywhere=True) for path in sources}) == CYCLES, (
        "no new cycle: move what both modules need into one both may import; one gone leaves CYCLES"
    )


def test_tests_close_the_databases_they_open():
    # A database a test opens with sqlite3.connect and leaves to the garbage collector is a ResourceWarning on Python
    # 3.13 and newer, raised in whichever test runs when it is collected: the gate, on an older Python, passed, and CI
    # failed elsewhere. Those opened through sift are closed by a fixture (conftest.close_databases).
    left_open = set()
    for path in (ROOT / "tests").glob("*.py"):
        for function in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            source = ast.unparse(function)
            for node in ast.walk(function):
                if isinstance(node, ast.Assign) and ast.unparse(node.value).startswith("sqlite3.connect("):
                    name = ast.unparse(node.targets[0])
                    if f"{name}.close()" not in source and f"return {name}" not in source:  # its caller's to close
                        left_open.add(f"{path.name}:{node.lineno}")
    assert left_open == set(), "close it, or open it in `with closing(sqlite3.connect(...)) as db:`"
