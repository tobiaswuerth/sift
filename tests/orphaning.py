import sys
from pathlib import Path

from sift.api.collections import TitleFrom
from sift.ingest.readers import ToRead
from sift.ingest.sandbox import sandboxed
from tests.sandboxed import held

if __name__ == "__main__":
    with sandboxed(held, workers=1) as reading:
        _ = list(reading([ToRead(Path("held.pdf"), int(sys.argv[1]), 1 << 20, None, TitleFrom.HEADING)]))
