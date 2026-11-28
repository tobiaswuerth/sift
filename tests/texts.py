from collections.abc import Iterable


def runs_joined(runs: Iterable[tuple[str, int]]) -> str:
    return "".join(character * count for character, count in runs)
