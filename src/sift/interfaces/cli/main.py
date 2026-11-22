import argparse
from collections.abc import Sequence
from importlib.metadata import version


def parser() -> argparse.ArgumentParser:
    built = argparse.ArgumentParser(prog="sift")
    built.add_argument("--version", action="version", version=f"sift {version('sift-rag')}")
    return built


def main(argv: Sequence[str] | None = None) -> None:
    command = parser()
    command.parse_args(argv)
    command.print_help()
