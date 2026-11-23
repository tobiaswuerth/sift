import tracemalloc
from pathlib import Path

from sift.domain.text.records import Skipped
from sift.ingest.records import lines_of

MOST = 10


def test_lines_are_read_numbered_and_stripped(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_bytes(b' {"a": 1} \r\n\n{"b": 2}')
    assert list(lines_of(path, MOST)) == [(1, '{"a": 1}'), (2, ""), (3, '{"b": 2}')]


def test_a_line_longer_than_a_line_may_be_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    long, exact, over = "x" * (3 * MOST), "y" * MOST, "z" * (MOST + 1)
    path.write_text(f"first\n{long}\n{exact}\r\n{over}\nlast\n{over}", encoding="utf-8", newline="")
    why = f"longer than the {MOST:,} characters a line may hold"
    found = list(lines_of(path, MOST))
    assert found == [
        (1, "first"),
        (2, Skipped(2, why)),
        (3, exact),
        (4, Skipped(4, why)),
        (5, "last"),
        (6, Skipped(6, why)),
    ]


def test_a_line_of_megabytes_is_never_held_whole(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text("x" * 8_000_000 + "\nlast\n", encoding="utf-8", newline="")
    tracemalloc.start()
    try:
        found = [number for number, _ in lines_of(path, MOST)]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert found == [1, 2]
    assert peak < 1 << 20
