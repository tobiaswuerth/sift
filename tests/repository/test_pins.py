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


def test_every_tool_a_workflow_installs_with_cargo_is_locked_at_a_version() -> None:
    installed = [
        (path.name, line.strip())
        for path in WORKFLOWS
        for line in path.read_text(encoding="utf-8").split("\n")
        if "cargo install" in line
    ]
    pinned = r"- run: cargo install [\w-]+ --locked --version [0-9]+\.[0-9]+\.[0-9]+"
    assert [each for each in installed if re.fullmatch(pinned, each[1]) is None] == []
    assert [name for name, _ in installed] == ["audit.yml"]


def test_the_crates_of_the_extension_are_audited_and_updated_as_the_packages_are() -> None:
    audit = (ROOT / ".github" / "workflows" / "audit.yml").read_text(encoding="utf-8")
    assert "- run: cargo audit --file native/Cargo.lock --deny warnings\n" in audit
    updates = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "  - package-ecosystem: cargo" in updates
    assert "    directory: /native\n" in updates
    assert updates.count("default-days: 7") == updates.count("package-ecosystem:")
