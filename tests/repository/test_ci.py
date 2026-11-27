import ast
from pathlib import Path
from typing import cast

from tests.repository.texts import job, matches

ROOT = Path(__file__).parents[2]


def named(source: str, name: str) -> object:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and [ast.unparse(each) for each in node.targets] == [name]:
            return cast("object", ast.literal_eval(node.value))
    raise LookupError(name)


def texts(source: str, name: str) -> tuple[str, ...]:
    value = named(source, name)
    assert isinstance(value, tuple)
    return tuple(str(each) for each in cast("tuple[object, ...]", value))


def matrix_pythons(workflow: str) -> list[list[str]]:
    return [[each.strip(' "') for each in listed.split(",")] for listed in matches(r"\bpython: \[([^\]]*)\]", workflow)]


def test_a_value_is_read_from_its_assignment_and_a_matrix_from_its_lists() -> None:
    assert named("A = 1\nB = ('3.12', '3.15')\n", "B") == ("3.12", "3.15")
    assert matrix_pythons('m:\n  python: ["3.12", "3.15"]\n  python: ["3.13"]\n') == [["3.12", "3.15"], ["3.13"]]


def test_ci_tests_every_python_the_gate_names_the_oldest_and_newest_first() -> None:
    noxfile = (ROOT / "noxfile.py").read_text(encoding="utf-8")
    pythons, oldest = texts(noxfile, "PYTHONS"), str(named(noxfile, "OLDEST"))
    newest = pythons[-1]
    assert texts(noxfile, "nox.options.sessions") == (
        "lint",
        "types",
        f"tests-{oldest}",
        f"tests-{newest}",
        "lowest",
        "kernel",
        "package",
    )
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert matrix_pythons(ci) == [[oldest, newest], list(pythons)]


def test_ci_tests_the_first_python_release_with_uv_s_own_build_of_it_and_needs_it() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    first = str(named((ROOT / "noxfile.py").read_text(encoding="utf-8"), "FIRST_RELEASE"))
    oldest = job(ci, "oldest")
    for line in (f'PYTHON: "{first}"', "SESSION: oldest", "UV_PYTHON_PREFERENCE: only-managed"):
        assert line in oldest, line
    assert (first, "needs: [first, test, lowest, oldest, kernel]" in job(ci, "tests")) == ("3.12.0", True)


def test_ci_holds_the_crate_to_its_tests_with_the_coverage_tool_of_the_version_it_names() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    kernel = job(ci, "kernel")
    assert "SESSION: kernel" in kernel
    noxfile = (ROOT / "noxfile.py").read_text(encoding="utf-8")
    assert matches(r"^\d+\.\d+\.\d+$", str(named(noxfile, "LLVM_COV"))) != []
    assert '"cargo-llvm-cov", "--locked", f"--version={LLVM_COV}"' in noxfile
    assert "KERNEL: ${{ needs.kernel.result }}" in job(ci, "tests")
    assert '"llvm-tools-preview"' in (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
