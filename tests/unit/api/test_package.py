import tomllib
from pathlib import Path
from typing import cast

from sift.api.package import PACKAGE


def test_the_package_is_named_as_its_project_publishes_it() -> None:
    project = cast("dict[str, dict[str, object]]", tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8")))
    assert project["project"]["name"] == PACKAGE
