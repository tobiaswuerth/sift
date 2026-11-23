from pathlib import Path


def at_most(path: Path, most: int) -> bytes:
    with path.open("rb") as raw:
        return raw.read(most + 1)
