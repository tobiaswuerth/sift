import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
BUILD_CONSTRAINTS = ".github/requirements-build.txt"
SILENCED_WITH_THE_OWNER_S_YES = frozenset({("release.yml", "workflow_run:  # zizmor: ignore[dangerous-triggers]")})


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def matches(pattern: str, text: str) -> list[str]:
    found: list[str] = re.findall(pattern, text)
    return found


def jobs(workflow: str) -> str:
    return "\n" + workflow.split("\njobs:\n", 1)[1]


def job(workflow: str, name: str) -> str:
    return re.split(r"\n  [\w-]+:\n", jobs(workflow).split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]


def test_a_job_is_cut_out_of_its_workflow() -> None:
    workflow = "on: push\njobs:\n  a:\n    runs-on: x\n  b-c:\n    runs-on: y\n"
    assert job(workflow, "a") == "    runs-on: x"
    assert job(workflow, "b-c") == "    runs-on: y\n"


def test_the_release_starts_only_for_ci_of_this_repository_s_main() -> None:
    release = read(WORKFLOWS / "release.yml")
    check = job(release, "check")
    for condition in (
        "github.event.workflow_run.conclusion == 'success'",
        "&& github.event.workflow_run.event == 'push'",
        "&& github.event.workflow_run.head_repository.full_name == github.repository",
    ):
        assert condition in check.split("runs-on:")[0], condition
    names = matches(r"\n  ([\w-]+):\n", jobs(release))
    assert names[0] == "check"
    waits = {name: set(matches(r"[\w-]+", matches(r"needs: (.*)", job(release, name))[0])) for name in names[1:]}
    behind_check = {"check", *(name for name, needed in waits.items() if "check" in needed)}
    assert all(needed & behind_check for needed in waits.values()), waits


def test_the_release_takes_nothing_of_the_run_that_started_it_but_its_commit() -> None:
    release = read(WORKFLOWS / "release.yml")
    taken = set(re.findall(r"github\.event\.workflow_run\.([\w.]+)", release))
    assert taken == {"conclusion", "event", "head_repository.full_name", "head_sha"}
    for line in release.split("\n"):
        if "workflow_run.head_sha" in line:
            assert line.strip().startswith(("ref: ", "COMMIT: ")), line
    assert "run-id" not in release
    assert "artifacts_url" not in release


def test_no_audit_of_the_workflows_is_silenced_without_the_owner_s_yes() -> None:
    silenced = {
        (path.name, line.strip())
        for path in (ROOT / ".github").rglob("*.yml")
        for line in read(path).split("\n")
        if "zizmor:" in line
    }
    assert silenced == SILENCED_WITH_THE_OWNER_S_YES
    assert "- id: zizmor" in read(ROOT / ".pre-commit-config.yaml")


def test_the_release_publishes_only_a_version_it_knows_is_new() -> None:
    check = job(read(WORKFLOWS / "release.yml"), "check")
    assert "> /dev/null || status=$?" in check
    assert '2) echo "v$version is new."' in check
    assert 'exit "$status"' in check


def test_the_release_reads_the_one_version_pyproject_names() -> None:
    check = job(read(WORKFLOWS / "release.yml"), "check")
    assert """version=$(sed -n 's/^version = "\\(.*\\)"$/\\1/p' pyproject.toml)""" in check
    assert len(re.findall(r'^version = "', read(ROOT / "pyproject.toml"), re.MULTILINE)) == 1


def test_the_package_is_built_by_tools_pinned() -> None:
    release, ci, gate = read(WORKFLOWS / "release.yml"), read(WORKFLOWS / "ci.yml"), read(ROOT / "noxfile.py")
    assert f"- run: uv build --build-constraints {BUILD_CONSTRAINTS} --require-hashes\n" in release
    assert f'BUILD_CONSTRAINTS = "{BUILD_CONSTRAINTS}"' in gate
    assert '"--require-hashes"' in gate
    (twine,) = matches(r"- run: uvx (twine@\d+\.\d+\.\d+) check --strict", release)
    assert f'TWINE = "{twine}"' in gate
    (uv,) = matches(r'version: "(\d+\.\d+\.\d+)"', job(release, "build"))
    assert matches(r'version: "(\d+\.\d+\.\d+)"', ci) == [uv]
    pinned = read(ROOT / BUILD_CONSTRAINTS)
    assert re.search(r"^uv-build==\S+ \\$", pinned, re.MULTILINE)
    assert "--hash=sha256:" in pinned


def test_the_release_refuses_a_version_without_its_changelog_section_before_it_builds() -> None:
    release = read(WORKFLOWS / "release.yml")
    check = job(release, "check")
    assert 'python3 -m tools release notes "$VERSION" > /dev/null' in check
    assert "needs: check" in job(release, "build")


def test_the_notes_released_are_those_the_check_refuses_a_version_without() -> None:
    release = read(WORKFLOWS / "release.yml")
    assert 'python3 -m tools release notes "$VERSION" > notes.md' in job(release, "github-release")
    assert "awk" not in release


def test_a_pre_release_is_published_as_one() -> None:
    release = read(WORKFLOWS / "release.yml")
    assert 'echo "prerelease=$(python3 -m tools release prerelease "$VERSION")"' in job(release, "check")
    made = job(release, "github-release")
    assert "PRERELEASE: ${{ needs.check.outputs.prerelease }}" in made
    assert '--notes-file notes.md "${pre[@]}"' in made
