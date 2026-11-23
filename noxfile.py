import shutil
from pathlib import Path

import nox

nox.options.default_venv_backend = "uv"
nox.options.error_on_missing_interpreters = True
nox.options.reuse_venv = "yes"
nox.options.stop_on_first_error = True
nox.options.sessions = ("lint", "types", "tests-3.12", "tests-3.15", "lowest", "package")

OLDEST = "3.12"
PYTHONS = ("3.12", "3.13", "3.14", "3.15")
PLATFORMS = ("Linux", "Darwin", "Windows")
BUILD_CONSTRAINTS = ".github/requirements-build.txt"
TWINE = "twine@7.0.0"


def synced(session: nox.Session, *also: str) -> None:
    session.run_install(
        "uv",
        "sync",
        "--locked",
        *also,
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
        external=True,
    )


def said(session: nox.Session, *command: str) -> str:
    output = session.run(*command, external=True, silent=True)
    return output.strip() if isinstance(output, str) else ""


@nox.session(python=OLDEST)
def lint(session: nox.Session) -> None:
    synced(session)
    if unknown := said(session, "git", "ls-files", "--others", "--exclude-standard"):
        session.error(f"git neither tracks nor ignores these, and the hooks check tracked files alone:\n{unknown}")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure")
    session.run("lint-imports")


@nox.session(python=OLDEST)
def types(session: nox.Session) -> None:
    synced(session, "--group=gate")
    python = shutil.which("python", path=session.bin)
    if python is None:
        session.error(f"no python in {session.bin}")
    for platform in PLATFORMS:
        session.run("basedpyright", "--pythonpath", python, "--pythonplatform", platform)


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    synced(session)
    session.run("pytest", "-q", "-n", "auto", "--cov", "--durations=10")


@nox.session(python=OLDEST)
def lowest(session: nox.Session) -> None:
    session.install("--resolution=lowest-direct", "--group=dev", "-e", ".")
    session.run("pytest", "-q", "-n", "auto")


@nox.session(python=OLDEST)
def package(session: nox.Session) -> None:
    synced(session)
    dist = Path(session.create_tmp()) / "dist"
    session.run(
        "uv",
        "build",
        "--clear",
        f"--out-dir={dist}",
        f"--build-constraints={BUILD_CONSTRAINTS}",
        "--require-hashes",
        external=True,
    )
    (wheel,) = map(str, dist.glob("*.whl"))
    (sdist,) = map(str, dist.glob("*.tar.gz"))
    session.run("uv", "tool", "run", TWINE, "check", "--strict", wheel, sdist, external=True)
    versions = {
        said(
            session,
            "uv",
            "tool",
            "run",
            "--quiet",
            "--isolated",
            "--torch-backend=cpu",
            "--from",
            wheel,
            name,
            "--version",
        )
        for name in ("sift", "sift-rag")
    }
    session.run("python", "-m", "tools", "package", str(dist), "\n".join(sorted(versions)))
