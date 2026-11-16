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
