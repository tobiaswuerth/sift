""".siftignore: gitignore patterns of the files and folders of a collection that sift leaves out."""

import pytest

from sift.ignore import Ignore


def ignores(lines, path, folder=False, ignore_case=False):
    return Ignore(lines.split("\n"), ignore_case)(path, folder)


@pytest.mark.parametrize(
    "lines, path, folder, ignored",
    [
        ("*.log", "a.log", False, True),
        ("*.log", "x/y/a.log", False, True),  # a pattern without a slash matches at any level
        ("*.log", "a.log.md", False, False),
        ("drafts/", "drafts", True, True),
        ("drafts/", "x/drafts", True, True),
        ("drafts/", "drafts", False, False),  # a slash at the end: folders only
        ("drafts", "drafts", False, True),
        ("/drafts", "drafts", True, True),  # a slash at the start: from the collection's folder only
        ("/drafts", "x/drafts", True, False),
        ("docs/*.md", "docs/a.md", False, True),  # a slash in the middle, too
        ("docs/*.md", "x/docs/a.md", False, False),
        ("docs/*.md", "docs/sub/a.md", False, False),  # * matches no slash
        ("**/temp", "temp", True, True),
        ("**/temp", "a/b/temp", True, True),
        ("a/**/b", "a/b", False, True),
        ("a/**/b", "a/x/y/b", False, True),
        ("logs/**", "logs/a/b.md", False, True),
        ("logs/**", "logs", True, False),  # what is inside, not the folder itself
        ("**", "any/thing.md", False, True),
        ("?.md", "a.md", False, True),
        ("?.md", "ab.md", False, False),
        ("[ab].md", "b.md", False, True),
        ("[!ab].md", "c.md", False, True),
        ("[!ab].md", "a.md", False, False),
        ("[a-c]x.md", "bx.md", False, True),
        ("[unclosed.md", "[unclosed.md", False, True),
        ("# a comment", "# a comment", False, False),
        ("\\#hash.md", "#hash.md", False, True),
        ("\\!bang.md", "!bang.md", False, True),
        ("trailing.md   ", "trailing.md", False, True),  # spaces at the end don't count ...
        ("space\\ ", "space ", False, True),  # ... unless escaped
        ("a.md\n\n\n", "a.md", False, True),
        ("*.md\n!keep.md", "keep.md", False, False),  # the last pattern that matches decides
        ("*.md\n!keep.md", "other.md", False, True),
        ("!keep.md\n*.md", "keep.md", False, True),
        ("", "a.md", False, False),
        ("/", "a", True, False),  # no pattern left: nothing
        ("!", "a.md", False, False),
    ],
)
def test_patterns_as_in_gitignore(lines, path, folder, ignored):
    assert ignores(lines, path, folder) == ignored


def test_case_as_the_file_system_takes_it():
    assert ignores("Drafts/", "drafts", True, ignore_case=True)
    assert not ignores("Drafts/", "drafts", True, ignore_case=False)


def test_a_file_in_a_folder_left_out_stays_out():
    # As in git, a file can't be taken back in when a folder above it is left out.
    ignore = Ignore(["private/", "!private/keep.md"])
    assert ignore.excludes("private/keep.md")
    assert ignore.excludes("x/private/a/b.md")
    assert not ignore.excludes("public/keep.md")


def test_of_a_folder(tmp_path):
    assert not Ignore.of(tmp_path)("a.md")  # no .siftignore: nothing more is left out
    (tmp_path / ".siftignore").write_bytes("\ufeff# exports\r\n*.pdf\r\n".encode())
    assert Ignore.of(tmp_path)("exports/x.pdf")
    assert not Ignore.of(tmp_path)("x.md")


def test_a_siftignore_sift_cannot_read_is_an_error(tmp_path):
    # Indexing everything would put the very files it should keep out into the index.
    (tmp_path / ".siftignore").mkdir()
    with pytest.raises(OSError, match=r"\.siftignore"):
        Ignore.of(tmp_path)


def test_many_patterns_stay_fast():
    ignore = Ignore([f"folder{i}/**/*.tmp" for i in range(200)] + ["*" * 10_000 + "x"])
    assert not ignore("a/" * 200 + "b.md")
