import re
import tomllib
from pathlib import Path
from typing import cast

ROOT = Path(__file__).parents[2]
PLATFORMS = {
    "ubuntu-latest": "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "ubuntu-24.04-arm": "sys_platform == 'linux' and platform_machine == 'aarch64'",
    "macos-latest": "sys_platform == 'darwin' and platform_machine == 'arm64'",
    "windows-latest": "sys_platform == 'win32' and platform_machine == 'AMD64'",
}


def runners(workflow: str) -> set[str]:
    lists: list[str] = re.findall(r"\bos: \[([^\]]*)\]", workflow)
    included: list[str] = re.findall(r"\{os: ([\w.-]+),", workflow)
    return {*(name.strip() for names in lists for name in names.split(",")), *included}


def test_the_runners_of_a_workflow_are_read_from_its_lists_and_inclusions() -> None:
    workflow = "matrix:\n  os: [a-1, b.2]\n  include:\n    - {os: c, python: '3.12'}\n"
    assert runners(workflow) == {"a-1", "b.2", "c"}


def test_the_lock_requires_every_platform_ci_runs_on() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    uv = cast("dict[str, list[str]]", cast("dict[str, object]", project["tool"])["uv"])
    ci = runners((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    assert ci - PLATFORMS.keys() == set(), "name the platform of each new runner in PLATFORMS"
    assert {PLATFORMS[runner] for runner in ci} - set(uv["required-environments"]) == set()
