import io
from pathlib import Path

from sift.interfaces.cli.main import run
from sift.settings import Settings


def ran(settings: Settings, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run(argv, settings, out, err)
    return code, out.getvalue(), err.getvalue()


def indexed_notes(settings: Settings, notes: Path) -> None:
    (notes / "rl.md").write_text("# Reward shaping\n\nReward shaping speeds up learning.\n", encoding="utf-8")
    (notes / "robots").mkdir()
    (notes / "robots" / "brush.md").write_text("# Brush\n", encoding="utf-8")
    assert ran(settings, "add", str(notes))[0] == 0
