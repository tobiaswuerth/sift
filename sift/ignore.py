"""`.siftignore`: the files and folders of a collection that sift leaves out, besides hidden folders, `node_modules`
and `__pycache__`. One file in the collection's folder, in gitignore's syntax: `*`, `?`, `[abc]` and `**`; a pattern
with a slash at its start or in its middle is taken from the collection's folder, one without matches at any level; a
slash at the end matches folders only; `!` takes back in what an earlier pattern left out, but never a file in a
folder left out; `#` starts a comment; `\\` escapes. Case counts as on the file system: not on Windows and macOS.
"""

import os
import re
import sys

from . import readers

NAME = ".siftignore"
CASELESS = sys.platform in ("win32", "darwin")  # as git takes it there, with core.ignorecase


class Ignore:
    """The patterns of a .siftignore, to ask of paths in the collection, `/`-separated, whether they are left out."""

    def __init__(self, lines, ignore_case=CASELESS):
        flags = re.IGNORECASE if ignore_case else 0
        self.rules = [rule for line in lines if (rule := _rule(line, flags))]

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
        for regex, taken_back, folders_only in reversed(self.rules):
            if (folder or not folders_only) and regex.match(path):
                return not taken_back
        return False

    def excludes(self, path):
        """Whether the file at `path` is left out, by itself or with a folder above it."""
        parts = path.split("/")
        return any(self("/".join(parts[:i]), folder=True) for i in range(1, len(parts))) or self(path)


def _rule(line, flags):
    """A line of a .siftignore as (regex, taken back?, folders only?), or None for a blank line or a comment."""
    kept = line.rstrip(" ")
    if kept.endswith("\\") and len(kept) < len(line):  # an escaped space at the end stays
        kept += " "
    if not kept or kept.startswith("#"):
        return None
    taken_back = kept.startswith("!")
    pattern = kept[1:] if taken_back else kept
    folders_only = pattern.endswith("/")
    pattern = pattern.rstrip("/")
    if not pattern:
        return None
    anywhere = "/" not in pattern
    regex = ("(?:.*/)?" if anywhere else "") + _translate(pattern.lstrip("/")) + r"\Z"
    return re.compile(regex, flags), taken_back, folders_only


def _translate(pattern):
    """A gitignore pattern as a regex: * and ? within a name, ** across folders, [...] a class, \\ escapes."""
    out, i, n = [], 0, len(pattern)
    while i < n:
        if pattern.startswith("**/", i) and (i == 0 or pattern[i - 1] == "/"):
            out.append("(?:.*/)?")  # no folder, or any
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == n:
            out.append("/.*")  # everything inside
            i += 3
        elif pattern[i] == "*":
            while i < n and pattern[i] == "*":
                i += 1
            out.append("[^/]*")
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and (end := _class_end(pattern, i)) > 0:
            body = pattern[i + 1 : end]
            negated = body[:1] in ("!", "^")
            body = "".join(c if c == "-" else re.escape(c) for c in body[negated:])
            out.append(f"[{'^/' if negated else ''}{body}]")
            i = end + 1
        elif pattern[i] == "\\" and i + 1 < n:
            out.append(re.escape(pattern[i + 1]))
            i += 2
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return "".join(out)


def _class_end(pattern, start):
    """Where the class that opens at `start` closes, or -1 if it doesn't: a `]` first in it is part of it."""
    i = start + 1
    i += pattern[i : i + 1] in ("!", "^")
    i += pattern[i : i + 1] == "]"
    return pattern.find("]", i)
