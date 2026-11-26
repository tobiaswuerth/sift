import re
from pathlib import Path

from tests.repository.texts import matches

ROOT = Path(__file__).parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def steps(workflow: str) -> list[str]:
    return re.split(r"\n\s+- (?=uses:|run:|name:)", workflow)[1:]


def test_every_tool_a_workflow_runs_through_uv_is_pinned() -> None:
    unpinned = {
        (path.name, tool)
        for path in WORKFLOWS
        for tool in matches(r"\buvx ([^\s@]+(?:@\S+)?)", path.read_text(encoding="utf-8"))
        if re.fullmatch(r"[\w-]+@[0-9]+(\.[0-9]+)*", tool) is None
    }
    assert unpinned == set()


def test_every_uv_a_workflow_sets_up_is_pinned() -> None:
    unpinned = {
        path.name
        for path in WORKFLOWS
        for step in steps(path.read_text(encoding="utf-8"))
        if step.startswith("uses: astral-sh/setup-uv@") and re.search(r'\n\s+version: "[0-9.]+"', step) is None
    }
    assert unpinned == set()


def test_every_hook_repository_is_pinned_at_a_commit_that_names_its_tag() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    revisions = matches(r"\n    rev: (.*)", config)
    assert len(revisions) == config.count("- repo: https://")
    assert [each for each in revisions if re.fullmatch(r"[0-9a-f]{40}  # frozen: \S+", each) is None] == []


def test_no_hook_installs_a_dependency_of_no_fixed_version() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    added = [
        each.strip()
        for listed in matches(r"additional_dependencies: \[([^\]]*)\]", config)
        for each in listed.split(",")
    ]
    assert [each for each in added if "==" not in each] == []
