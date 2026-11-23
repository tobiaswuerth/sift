from collections.abc import Callable
from pathlib import Path, PurePath

MOST_HOPS = 40
LONG = "\\\\?\\"
LONG_SHARE = "\\\\?\\UNC\\"

type LinkTarget = Callable[[Path], str | None]


def plain(written: str) -> str:
    return "\\\\" + written.removeprefix(LONG_SHARE) if written.startswith(LONG_SHARE) else written.removeprefix(LONG)


def link_target(path: Path) -> str | None:
    return plain(str(path.readlink())) if path.is_symlink() or path.is_junction() else None


def _within(parts: tuple[str, ...], base: tuple[str, ...]) -> bool:
    return len(parts) >= len(base) and all(
        PurePath(part) == PurePath(held) for part, held in zip(parts, base, strict=False)
    )


def lands_within(relative: str, root: Path, target: LinkTarget = link_target) -> bool:
    base = root.parts
    at, rest, hops = list(base), list(reversed(PurePath(relative).parts)), 0
    while rest:
        part = rest.pop()
        if part == "..":
            if len(at) == len(base):
                return False
            at.pop()
            continue
        at.append(part)
        pointed = target(Path(*at))
        if pointed is None:
            continue
        written, hops = PurePath(pointed), hops + 1
        if hops > MOST_HOPS or (written.anchor and not _within(written.parts, base)):
            return False
        at = list(base) if written.anchor else at[:-1]
        rest.extend(reversed(written.parts[len(base) :] if written.anchor else written.parts))
    return True
