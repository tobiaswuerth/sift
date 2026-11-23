from sift.domain.text.wording import counted


def test_counts_are_written_with_their_noun() -> None:
    assert [counted(n, "file") for n in (0, 1, 2, 12_345)] == ["0 files", "1 file", "2 files", "12,345 files"]
