import os
import re
import shutil
from pathlib import Path

import nox

nox.options.default_venv_backend = "uv"
nox.options.error_on_missing_interpreters = True
nox.options.reuse_venv = "yes"
nox.options.stop_on_first_error = True
nox.options.sessions = ("lint", "types", "tests-3.12", "tests-3.15", "lowest", "kernel", "package")

OLDEST = "3.12"
FIRST_RELEASE = "3.12.0"
PYTHONS = ("3.12", "3.13", "3.14", "3.15")
PLATFORMS = ("Linux", "Darwin", "Windows")
BUILD_CONSTRAINTS = ".github/requirements-build.txt"
CRATE = "native/Cargo.toml"
CRATE_COVERAGE = "native/target/coverage"
LLVM_COV = "0.9.2"
CRATE_REACHED_BY = ("tests/unit/domain/codec", "tests/unit/storage", "tests/unit/retrieval/keyword")
CRATE_TESTS = r"src[/\\]tests"
BINDING = Path("native/src/python")
NO_BINDING = (
    "the coverage report holds nothing of {missing}: the instrumented extension was not found, so its lines "
    "were not counted. Look where maturin put it and where cargo llvm-cov looks."
)
TWINE = "twine@7.0.0"
MOST_WORKERS = 8
WORKERS = str(min(MOST_WORKERS, os.cpu_count() or 1))


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


def said(session: nox.Session, *command: str, env: dict[str, str] | None = None) -> str:
    output = session.run(*command, external=True, silent=True, env=env)
    return output.strip() if isinstance(output, str) else ""


def instrumented(session: nox.Session) -> dict[str, str]:
    target = {"CARGO_TARGET_DIR": str(Path(CRATE_COVERAGE).resolve())}
    shown = said(session, "cargo", "llvm-cov", "show-env", f"--manifest-path={CRATE}", env=target)
    pairs = (line.split("=", 1) for line in shown.split("\n") if "=" in line)
    return target | {name: value.strip().strip("'") for name, value in pairs}


@nox.session(python=OLDEST)
def lint(session: nox.Session) -> None:
    synced(session)
    if unknown := said(session, "git", "ls-files", "--others", "--exclude-standard"):
        session.error(f"git neither tracks nor ignores these, and the hooks check tracked files alone:\n{unknown}")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure")
    session.run("cargo", "fmt", f"--manifest-path={CRATE}", "--check", external=True)
    for built in ("--features=python", "--tests"):
        session.run("cargo", "clippy", f"--manifest-path={CRATE}", "--locked", "--release", built, external=True)
    session.run("lint-imports")
    session.run("python", "-m", "tools", "imports")
    session.run("python", "-m", "tools", "stubs")


@nox.session(python=OLDEST)
def types(session: nox.Session) -> None:
    synced(session, "--group=gate")
    python = shutil.which("python", path=session.bin)
    if python is None:
        session.error(f"no python in {session.bin}")
    for platform in PLATFORMS:
        for version in (OLDEST, PYTHONS[-1]):
            session.run(
                "basedpyright", "--pythonpath", python, "--pythonplatform", platform, "--pythonversion", version
            )


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    synced(session)
    session.run("pytest", "-q", "-n", WORKERS, "--cov", "--durations=10")


@nox.session(python=OLDEST)
def explore(session: nox.Session) -> None:
    synced(session)
    session.run("pytest", "-q", "-n", WORKERS, "--hypothesis-profile=explore", "--hypothesis-show-statistics")


@nox.session(python=OLDEST)
def real(session: nox.Session) -> None:
    synced(session)
    session.run("python", "-m", "tests.real_models")


@nox.session(python=OLDEST, reuse_venv=False)
def lowest(session: nox.Session) -> None:
    session.install("--resolution=lowest-direct", "--group=dev", "-e", ".")
    session.run("python", "-m", "tools", "floors")
    session.run("pytest", "-q", "-n", WORKERS)


@nox.session(python=FIRST_RELEASE)
def oldest(session: nox.Session) -> None:
    synced(session)
    session.run("pytest", "-q", "-n", WORKERS)


@nox.session(python=OLDEST)
def kernel(session: nox.Session) -> None:
    synced(session)
    session.run("cargo", "install", "cargo-llvm-cov", "--locked", f"--version={LLVM_COV}", external=True)
    env = instrumented(session)
    session.run("cargo", "llvm-cov", "clean", f"--manifest-path={CRATE}", "--workspace", external=True, env=env)
    session.run("cargo", "test", f"--manifest-path={CRATE}", "--locked", "--release", external=True, env=env)
    dist = Path(CRATE_COVERAGE) / "wheel"
    session.run(
        "uv",
        "build",
        "--wheel",
        "--clear",
        f"--out-dir={dist}",
        f"--build-constraints={BUILD_CONSTRAINTS}",
        "--require-hashes",
        external=True,
        env=env,
    )
    (wheel,) = map(str, dist.glob("*.whl"))
    session.run("uv", "pip", "install", "--reinstall", "--no-deps", wheel, external=True)
    session.run("pytest", "-q", "-n", WORKERS, *CRATE_REACHED_BY, env=env)
    report = ("cargo", "llvm-cov", "report", f"--manifest-path={CRATE}", "--release")
    checked = ("--fail-under-lines=100", f"--ignore-filename-regex={CRATE_TESTS}", "--show-missing-lines")
    told = said(session, *report, *checked, env=env)
    session.log(told)
    bound = [Path("python.rs"), *sorted(Path("python", each.name) for each in BINDING.glob("*.rs"))]
    if missing := [str(each) for each in bound if not re.search(rf"^{re.escape(str(each))}\s", told, re.MULTILINE)]:
        session.error(NO_BINDING.format(missing=", ".join(missing)))


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
