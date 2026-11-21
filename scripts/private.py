"""Keeps what is private out of the repository: `python scripts/private.py FILE ...` fails if a file holds a match of
one of a maintainer's patterns, and says where, by the pattern's line, never by what it found. A pre-commit hook runs
it on the files of a commit and on its message.

The patterns are private themselves, so they are in no repository: a file of regular expressions, one a line, case
ignored, blank lines and lines starting with # left out. It is the file PRIVATE_PATTERNS names, else
~/.private-patterns. Where that isn't there, in CI and for contributors, nothing is checked; a file named and not
there is an error. Whoever keeps the file owns it: this only reads it."""

import os
import re
import sys
from pathlib import Path


def patterns(environ, home):
    """The patterns' file, and each pattern by its line in it; no file, and none, where the default isn't there."""
    named = environ.get("PRIVATE_PATTERNS")
    path = Path(named) if named else Path(home) / ".private-patterns"
    if not path.is_file():
        if named:
            raise SystemExit(f"PRIVATE_PATTERNS names {path}, which isn't there")
        return path, {}
    found = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if line.strip() and not line.startswith("#"):
            try:
                found[n] = re.compile(line.strip(), re.IGNORECASE)
            except re.error as error:
                raise SystemExit(f"line {n} of {path} is no regular expression: {error.msg}") from None
    return path, found


def main(files, environ=os.environ, home=None):
    path, wanted = patterns(environ, home or Path.home())
    failed = 0
    for file in files if wanted else ():
        try:
            lines = Path(file).read_text(encoding="utf-8", errors="ignore").split("\n")
        except OSError:  # deleted in this commit, or a folder
            continue
        for at, line in enumerate(lines, 1):
            for n, pattern in wanted.items():
                if pattern.search(line):
                    print(f"{file}:{at}: private (pattern of line {n} of {path})")
                    failed = 1
    return failed


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
