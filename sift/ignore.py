"""`.siftignore`: the files and folders of a collection that sift leaves out, besides hidden folders, `node_modules`
and `__pycache__`. One file in the collection's folder, in gitignore's syntax: `*`, `?`, `[abc]` and `**`; a pattern
with a slash at its start or in its middle is taken from the collection's folder, one without matches at any level; a
slash at the end matches folders only; `!` takes back in what an earlier pattern left out, but never a file in a
folder left out; `#` starts a comment; `\\` escapes. Case counts as on the file system: not on Windows and macOS,
where neither does the Unicode form of a name.

A name's pattern is matched as the pieces between its stars, and a path's as the names between its `**` parts: the
first at the start, the last at the end, those between found from left to right. That takes at most the path's length
times the pattern's. As one regex, `*a*a*a*a*a*a*a*a*a*b` took 73 s against a name of 50 letters, each star trying
every split of the rest.
"""

import os
import re
import sys

from . import readers

NAME = ".siftignore"
CASELESS = sys.platform in ("win32", "darwin")  # as git takes it there, with core.ignorecase
STAR = object()  # in a name's tokens, a run of stars: any characters (an escaped star is "*" itself)
DEEP = object()  # in a path's parts, `**`: any number of folders


class Ignore:
    """The patterns of a .siftignore, to ask of paths in the collection, `/`-separated, whether they are left out."""

    def __init__(self, lines, ignore_case=CASELESS):
        self.fold = readers.fold if ignore_case else str  # with case, Unicode form: macOS hands out names decomposed
        self.rules = [rule for line in lines if (rule := _rule(self.fold(line)))]

    @classmethod
    def of(cls, folder):
        """The .siftignore of a collection's folder; none leaves nothing out. One sift can't read is an error: indexing
        all would take in the very files it should keep out."""
        path = os.path.join(folder, NAME)
        if not os.path.lexists(path):
            return cls([])
        with open(path, "rb") as file:
            return cls(readers.decode(file.read()).splitlines())

    def __call__(self, path, folder=False):
        """Whether the pattern that last matches the file, or with `folder` the folder, at `path` leaves it out."""
        names = self.fold(path).split("/")
        for matches, taken_back, folders_only in reversed(self.rules):
            if (folder or not folders_only) and matches(names):
                return not taken_back
        return False

    def excludes(self, path):
        """Whether the file at `path` is left out, by itself or with a folder above it."""
        parts = path.split("/")
        return any(self("/".join(parts[:i]), folder=True) for i in range(1, len(parts))) or self(path)


def _rule(line):
    """A line of a .siftignore as (a test of a path's names, taken back?, folders only?), or None for a blank line or
    a comment."""
    kept = line.rstrip(" ")
    backslashes = len(kept) - len(kept.rstrip("\\"))
    if backslashes % 2 and len(kept) < len(line):  # an escaped space at the end stays, not one after `\\`
        kept += " "
    if not kept or kept.startswith("#"):
        return None
    taken_back = kept.startswith("!")
    pattern = kept[1:] if taken_back else kept
    folders_only = pattern.endswith("/")
    pattern = pattern.rstrip("/")
    if not pattern:
        return None
    if "/" not in pattern:  # a name, at any level
        name = _glob(pattern)
        return (lambda names: name(names[-1])), taken_back, folders_only
    parts = [DEEP if part == "**" else _glob(part) for part in pattern.lstrip("/").split("/")]
    # `logs/**` is what is inside logs: at least one more name, not logs itself
    path = _sequence(_split(parts, DEEP), _names_at, _names_from, len, at_least=parts[-1] is DEEP)
    return path, taken_back, folders_only


def _glob(part):
    """A test of names for a name's pattern: the pieces between its stars, each a regex of one character per token,
    without quantifiers, so it can't backtrack."""
    pieces = [(re.compile("".join(piece)), len(piece)) for piece in _split(_tokens(part), STAR)]
    return _sequence(pieces, _piece_at, _piece_from, lambda piece: piece[1])


def _split(items, star):
    """The groups of items between the stars."""
    groups = [[]]
    for item in items:
        if item is star:
            groups.append([])
        else:
            groups[-1].append(item)
    return groups


def _sequence(groups, at, find, size, at_least=False):
    """A test of sequences (a name's characters, a path's names) against groups that stars separate: the first group
    at the start, the last at the end, those between at the first place each fits, from left to right. `at(group, seq,
    i)` tells whether a group matches at i, `find(group, seq, i)` where it first does from i on, `size(group)` how
    many elements it takes. With `at_least`, the star at the end takes at least one element."""
    if len(groups) == 1:
        (whole,) = groups
        return lambda seq: len(seq) == size(whole) and at(whole, seq, 0)
    first, *middle, last = groups

    def test(seq):
        if not at(first, seq, 0):
            return False
        i = size(first)
        for group in middle:
            if (i := find(group, seq, i)) < 0:
                return False
            i += size(group)
        end = len(seq) - size(last)
        return end >= i + at_least and at(last, seq, end)

    return test


def _piece_at(piece, name, i):
    return piece[0].match(name, i) is not None


def _piece_from(piece, name, i):
    found = piece[0].search(name, i)
    return found.start() if found else -1


def _names_at(group, names, i):
    return i + len(group) <= len(names) and all(test(name) for test, name in zip(group, names[i:]))


def _names_from(group, names, i):
    return next((j for j in range(i, len(names) - len(group) + 1) if _names_at(group, names, j)), -1)


def _tokens(part):
    """A name's pattern as tokens: STAR, or the regex of one character: itself, any for `?`, or a class."""
    tokens, i = [], 0
    while i < len(part):
        c = part[i]
        if c == "*":
            while i < len(part) and part[i] == "*":
                i += 1
            tokens.append(STAR)
            continue
        if c == "[" and (end := _class_end(part, i)) > 0:
            tokens.append(_class(part[i + 1 : end]))
            i = end + 1
            continue
        escaped = c == "\\" and i + 1 < len(part)
        if escaped:
            i += 1
            c = part[i]
        tokens.append("(?s:.)" if c == "?" and not escaped else re.escape(c))
        i += 1
    return tokens


def _class(body):
    """`[body]` as the regex of one character: its characters and ranges, `!` or `^` first for any other. A range
    backwards, `z-a`, holds nothing, as in git; a regex would refuse it."""
    negated = body[:1] in ("!", "^")
    body, held, i = body[negated:], [], 0
    while i < len(body):
        if i + 2 < len(body) and body[i + 1] == "-":
            if body[i] <= body[i + 2]:
                held.append(f"{re.escape(body[i])}-{re.escape(body[i + 2])}")
            i += 3
        else:
            held.append(re.escape(body[i]))
            i += 1
    if not held:
        return "(?s:.)" if negated else "(?!)"
    return f"[{'^' if negated else ''}{''.join(held)}]"


def _class_end(pattern, start):
    """Where the class that opens at `start` closes, or -1 if it doesn't: a `]` first in it is part of it."""
    i = start + 1
    i += pattern[i : i + 1] in ("!", "^")
    i += pattern[i : i + 1] == "]"
    return pattern.find("]", i)
