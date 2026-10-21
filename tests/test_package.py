"""The package as it is published: what it depends on."""

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
        module
        for path in (ROOT / "sift").glob("*.py")
        for module in re.findall(r"^\s*(?:from|import) (\w+)", path.read_text(encoding="utf-8"), re.M)
    }
    distributions = importlib.metadata.packages_distributions()
    undeclared = {
        module
        for module in imported - set(sys.stdlib_module_names)
        if not {d.lower() for d in distributions[module]} & declared
    }
    assert undeclared == set()
