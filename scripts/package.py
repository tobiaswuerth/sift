"""The package as PyPI and its users get it, checked: `python scripts/package.py`. The changelog has a section for the
version; the package builds, with its backend pinned by hash, as the release builds it; twine finds nothing wrong
with it; its wheel installs alone in a clean environment, where both commands start and the pages of `sift ui` are
there; and its dependencies resolve. CI's package job runs it, and so does the gate. What it makes is in a folder of
its own, deleted at the end. It needs uv on the PATH."""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUILD = ["uv", "build", "--quiet", "--build-constraints", str(ROOT / ".github" / "requirements-build.txt")]
BUILD += ["--require-hashes"]  # as the release builds it
TWINE = "twine@7.0.0"  # as the release checks it
MINUTES = 10  # a step may take, downloads included


def version() -> str:
    """The version sift/__init__.py names."""
    found = re.search(r'^__version__ = "(.+)"$', (ROOT / "sift" / "__init__.py").read_text(encoding="utf-8"), re.M)
    if found is None:
        raise SystemExit("package: sift/__init__.py names no __version__")
    return found[1]


def run(*command: str | Path) -> None:
    subprocess.run([str(part) for part in command], check=True, timeout=60 * MINUTES)


def main() -> None:
    named = version()
    if not re.search(rf"^## \[{re.escape(named)}\]", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M):
        raise SystemExit(f"package: CHANGELOG.md has no section for {named}, the version in sift/__init__.py")
    with tempfile.TemporaryDirectory(prefix="sift-package-") as folder:
        dist, clean = Path(folder) / "dist", Path(folder) / "clean"
        run(*BUILD, "--out-dir", dist, ROOT)
        run("uvx", TWINE, "check", "--strict", *sorted(dist.glob("sift_rag-*")))
        (wheel,) = dist.glob("*.whl")
        run("uv", "venv", "--quiet", clean)
        python = clean / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        run("uv", "pip", "install", "--quiet", "--python", python, "--no-deps", wheel)
        for command in ("sift", "sift-rag"):
            run(python.parent / command, "--version")
        web = "import pathlib, sift; web = pathlib.Path(sift.__file__).parent / 'web'"
        run(
            python,
            "-I",
            "-c",
            f"{web}; assert (web / 'page.html').is_file() and (web / 'static/htmx.min.js').is_file()",
        )
        run("uv", "pip", "install", "--quiet", "--python", python, "--dry-run", wheel)  # its dependencies resolve
    print(f"package: sift-rag {named} builds, checks, installs and starts")


if __name__ == "__main__":
    main()
