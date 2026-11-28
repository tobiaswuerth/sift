import os
import stat
import sys
from pathlib import Path

from sift.domain.plan.ignore import NOTHING_LEFT_OUT, Ignore, ignore_of
from sift.domain.text.decoding import decoded
from sift.ingest.files import Identity, at_most
from sift.ingest.links import lands_within

IGNORE_FILE = ".siftignore"
CASE_FOLDING = frozenset({"win32", "darwin"})
MOST_IGNORE_BYTES = 1 << 20
NOT_A_FILE = "it is not a file"
LEADS_OUTSIDE = "it is a link that leads outside the folder"


class IgnoreUnreadableError(OSError):
    pass


def ignore_at(folder: Path, most: int = MOST_IGNORE_BYTES, *, platform: str = sys.platform) -> Ignore:
    path = folder / IGNORE_FILE
    if not os.path.lexists(path):
        return NOTHING_LEFT_OUT
    if not lands_within(IGNORE_FILE, folder.resolve()):
        raise IgnoreUnreadableError(LEADS_OUTSIDE)
    found = path.stat()
    if not stat.S_ISREG(found.st_mode):
        raise IgnoreUnreadableError(NOT_A_FILE)
    raw = at_most(path, most, Identity(found.st_dev, found.st_ino))
    if len(raw) > most:
        raise IgnoreUnreadableError(f"it holds more than the {most:,} bytes sift reads of it")
    lines = (line.removesuffix("\r") for line in decoded(raw).split("\n"))
    return ignore_of(lines, caseless=platform in CASE_FOLDING)
