import os
import subprocess
import sys
from pathlib import Path


def sift(
    home: Path, *args: str, model: str = "", reranker: str = "", cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sift", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=180,
        env={
            **os.environ,
            "SIFT_HOME": str(home),
            "PYTHONIOENCODING": "utf-8",
            "NO_COLOR": "1",
            "SIFT_EMBED_MODEL": model,
            "SIFT_RERANK_MODEL": reranker,
        },
    )
