"""The package as it is published: what it depends on."""

import ast
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent


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
