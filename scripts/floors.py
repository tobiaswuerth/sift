"""The lowest versions sift names of what it depends on are versions an install reaches: `python scripts/floors.py`,
run where they were asked for (`uv pip install --resolution lowest-direct -e ".[dev]"`), as CI's lowest job and the
gate do once they are installed. A version named below what another dependency takes is one sift never ran with,
though pyproject.toml says it does."""

import importlib.metadata
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
REQUIREMENT = re.compile(r"([A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*(?:>=|==)\s*([0-9]+(?:\.[0-9]+)*)")


def named(requirements: list[str]) -> dict[str, str]:
    """Each package required, and the lowest version named for it."""
    lowest: dict[str, str] = {}
    for requirement in requirements:
        found = REQUIREMENT.fullmatch(requirement)
        if found is None:
            raise SystemExit(f"floors: {requirement!r} names no lowest version: write it as name>=version")
        lowest[found[1]] = found[2]
    return lowest


def above(lowest: dict[str, str], installed: dict[str, str]) -> list[str]:
    """The packages installed at another version than the lowest named for them, as far as that names it: 2.0 is met
    by 2.0.0, and 12 by 12.535.77, the lowest release there is."""
    found: list[str] = []
    for name, version in lowest.items():
        parts = version.split(".")
        if installed[name].split("+")[0].split(".")[: len(parts)] != parts:
            found.append(f"{name} {installed[name]}, named from {version}")
    return found


def main() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = [requirement for extra in project["optional-dependencies"].values() for requirement in extra]
    lowest = named(project["dependencies"] + extras)
    if found := above(lowest, {name: importlib.metadata.version(name) for name in lowest}):
        raise SystemExit(
            f"floors: installed above the lowest version named: {'; '.join(found)}. Name the version installed, in "
            "pyproject.toml and requirements.txt"
        )
    print(f"floors: the {len(lowest)} lowest versions named are those installed")


if __name__ == "__main__":
    main()
