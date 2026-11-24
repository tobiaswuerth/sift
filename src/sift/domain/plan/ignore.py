import re
from collections.abc import Callable, Iterable, Sequence, Sized
from dataclasses import dataclass
from enum import Enum, auto

from sift.domain.text.links import folded

ANY_CHARACTER = "(?s:.)"
NEGATIONS = ("!", "^")


class _Wild(Enum):
    STAR = auto()
    DEEP = auto()


type NameTest = Callable[[str], bool]
type PathTest = Callable[[Sequence[str]], bool]
type Stretch = tuple[re.Pattern[str], int]


@dataclass(frozen=True, slots=True)
class IgnoreRule:
    matches: PathTest
    taken_back: bool
    folders_only: bool


@dataclass(frozen=True, slots=True)
class Ignore:
    rules: tuple[IgnoreRule, ...] = ()
    caseless: bool = False


NOTHING_LEFT_OUT = Ignore()


def in_order[G, W: Sized](
    groups: Sequence[G],
    at: Callable[[G, W, int], bool],
    find: Callable[[G, W, int], int],
    size: Callable[[G], int],
    *,
    at_least: bool = False,
) -> Callable[[W], bool]:
    if len(groups) == 1:
        (whole,) = groups
        return lambda found: len(found) == size(whole) and at(whole, found, 0)
    first, *middle, last = groups

    def test(found: W) -> bool:
        if not at(first, found, 0):
            return False
        reached = size(first)
        for group in middle:
            if (reached := find(group, found, reached)) < 0:
                return False
            reached += size(group)
        end = len(found) - size(last)
        return end >= reached + at_least and at(last, found, end)

    return test


def _split[T](items: Iterable[T | _Wild]) -> list[list[T]]:
    groups: list[list[T]] = [[]]
    for item in items:
        if isinstance(item, _Wild):
            groups.append([])
        else:
            groups[-1].append(item)
    return groups


def _stretch_at(stretch: Stretch, name: str, at: int) -> bool:
    return stretch[0].match(name, at) is not None


def _stretch_from(stretch: Stretch, name: str, at: int) -> int:
    found = stretch[0].search(name, at)
    return found.start() if found else -1


def _stretch_size(stretch: Stretch) -> int:
    return stretch[1]


def _names_at(group: Sequence[NameTest], names: Sequence[str], at: int) -> bool:
    held = names[at : at + len(group)]
    return len(held) == len(group) and all(test(name) for test, name in zip(group, held, strict=True))


def _names_from(group: Sequence[NameTest], names: Sequence[str], at: int) -> int:
    return next((each for each in range(at, len(names) - len(group) + 1) if _names_at(group, names, each)), -1)


def _class_end(pattern: str, start: int) -> int:
    at = start + 1
    at += pattern[at : at + 1] in NEGATIONS
    at += pattern[at : at + 1] == "]"
    return pattern.find("]", at)


def _class(body: str) -> str:
    negated = body[:1] in NEGATIONS
    body, at = body[negated:], 0
    held: list[str] = []
    while at < len(body):
        if at + 2 < len(body) and body[at + 1] == "-":
            last = f"-{re.escape(body[at + 2])}" if body[at] <= body[at + 2] else ""
            held.append(re.escape(body[at]) + last)
            at += 3
        else:
            held.append(re.escape(body[at]))
            at += 1
    return f"[{'^' if negated else ''}{''.join(held)}]"


def _tokens(part: str) -> list[str | _Wild]:
    tokens: list[str | _Wild] = []
    at = 0
    while at < len(part):
        character = part[at]
        if character == "*":
            while at < len(part) and part[at] == "*":
                at += 1
            tokens.append(_Wild.STAR)
            continue
        if character == "[" and (end := _class_end(part, at)) > 0:
            tokens.append(_class(part[at + 1 : end]))
            at = end + 1
            continue
        escaped = character == "\\" and at + 1 < len(part)
        if escaped:
            at += 1
            character = part[at]
        tokens.append(ANY_CHARACTER if character == "?" and not escaped else re.escape(character))
        at += 1
    return tokens


def _glob(part: str) -> NameTest:
    stretches = [(re.compile("".join(tokens)), len(tokens)) for tokens in _split(_tokens(part))]
    return in_order(stretches, _stretch_at, _stretch_from, _stretch_size)


def _kept(line: str) -> str:
    kept = line.rstrip(" ")
    backslashes = len(kept) - len(kept.rstrip("\\"))
    return kept + " " if backslashes % 2 and len(kept) < len(line) else kept


def _rule(line: str) -> IgnoreRule | None:
    kept = _kept(line)
    if not kept or kept.startswith("#"):
        return None
    taken_back = kept.startswith("!")
    pattern = kept[1:] if taken_back else kept
    folders_only = pattern.endswith("/")
    pattern = pattern.rstrip("/")
    if not pattern:
        return None
    if "/" not in pattern:
        name = _glob(pattern)
        return IgnoreRule(lambda names: name(names[-1]), taken_back, folders_only)
    parts = [_Wild.DEEP if part == "**" else _glob(part) for part in pattern.lstrip("/").split("/")]
    path = in_order(_split(parts), _names_at, _names_from, len, at_least=parts[-1] is _Wild.DEEP)
    return IgnoreRule(path, taken_back, folders_only)


def _as_compared(text: str, *, caseless: bool) -> str:
    return folded(text) if caseless else text


def ignore_of(lines: Iterable[str], *, caseless: bool) -> Ignore:
    rules = (_rule(_as_compared(line, caseless=caseless)) for line in lines)
    return Ignore(tuple(rule for rule in rules if rule is not None), caseless)


def left_out(ignore: Ignore, path: str, *, folder: bool = False) -> bool:
    names = _as_compared(path, caseless=ignore.caseless).split("/")
    for rule in reversed(ignore.rules):
        if (folder or not rule.folders_only) and rule.matches(names):
            return not rule.taken_back
    return False
