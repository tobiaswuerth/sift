from pathlib import Path

from sift.interfaces.cli.flags import file_or_id


def test_a_file_named_is_a_path_where_it_is_found_or_written_from_here_and_an_id_else(tmp_path: Path) -> None:
    found = tmp_path / "notes" / "a.md"
    found.parent.mkdir()
    found.write_bytes(b"# A\n")
    assert file_or_id(str(found)) == (str(found),)
    assert file_or_id("./gone.md") == (str(Path("./gone.md").resolve()),)
    assert file_or_id("../gone.md") == (str(Path("../gone.md").resolve()),)
    assert file_or_id("notes-nowhere/a.md") == ("notes-nowhere/a.md",)
