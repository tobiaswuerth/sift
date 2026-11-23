from collections.abc import Callable, Sequence
from pathlib import Path, PurePath

MOST_HOPS = 40
LONG = "\\\\?\\"
LONG_SHARE = "\\\\?\\UNC\\"

type LinkTarget = Callable[[Path], str | None]


def plain(written: str) -> str:
    return "\\\\" + written.removeprefix(LONG_SHARE) if written.startswith(LONG_SHARE) else written.removeprefix(LONG)


def link_target(path: Path) -> str | None:
    return plain(str(path.readlink())) if path.is_symlink() or path.is_junction() else None


def _within(parts: Sequence[str], base: tuple[str, ...]) -> bool:
    return len(parts) >= len(base) and all(
        PurePath(part) == PurePath(held) for part, held in zip(parts, base, strict=False)
    )


def collapsed(parts: Sequence[str]) -> tuple[str, ...] | None:
    kept: list[str] = []
    for part in parts:
        if part != "..":
            kept.append(part)
        elif len(kept) > 1:
            kept.pop()
        else:
            return None
    return tuple(kept)


def _spelled(written: PurePath, base: tuple[str, ...]) -> tuple[str, ...]:
    if not written.anchor or _within(written.parts, base) or PurePath(written.anchor) != PurePath(base[0]):
        return written.parts
    return (*Path(written.parent).resolve().parts, written.name)


def _lands_within(relative: str, base: tuple[str, ...], target: LinkTarget, *, textual: bool) -> bool:
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
        whole = _spelled(written, base) if written.anchor else (*at[:-1], *written.parts)
        landed = collapsed(whole) if textual else whole
        if hops > MOST_HOPS or landed is None or not _within(landed, base):
            return False
        at = list(base)
        rest.extend(reversed(landed[len(base) :]))
    return True


def lands_within(relative: str, root: Path, target: LinkTarget = link_target) -> bool:
    base = root.parts
    return all(_lands_within(relative, base, target, textual=textual) for textual in (False, True))
