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
        ("**/private/**", "x/private/a.md", False, True),
        ("**/private/**", "private/a.md", False, True),
        ("**/private/**", "x/private", True, False),  # what is inside, not the folder
        ("a/**/b/**/c", "a/x/b/y/z/c", False, True),
        ("a/**/b/**/c", "a/x/c", False, False),
        ("logs/**", "logs/a/b.md", False, True),
        ("logs/**", "logs", True, False),  # what is inside, not the folder itself
        ("**", "any/thing.md", False, True),
        ("notes*", "notes", False, True),  # a star matches nothing too
        ("*notes*", "my notes 2", False, True),
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
        ("what\\?.md", "what?.md", False, True),  # an escaped wildcard is itself
        ("what\\?.md", "whatx.md", False, False),
        ("a\\\\?.md", "a\\x.md", False, True),  # an escaped backslash, then a wildcard
        ("\\*.md", "x.md", False, False),
        ("trailing.md   ", "trailing.md", False, True),  # spaces at the end don't count ...
        ("space\\ ", "space ", False, True),  # ... unless escaped
        ("a\\\\  ", "a\\", False, True),  # ... and not after an escaped backslash
        ("a\\\\\\  ", "a\\ ", False, True),
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


def test_a_range_backwards_matches_nothing():
    # As in git; as a regex it raised "bad character range", and stopped sift index.
    assert not ignores("[z-a].md", "b.md")
    assert ignores("[z-ab].md", "b.md")
    assert ignores("[!z-a].md", "b.md")


def test_names_match_as_fnmatch_matches_them():
    # Within one name, gitignore's * ? [...] mean what fnmatch's do: thousands of random patterns and names agree.
    import fnmatch
    import random

    rng = random.Random(3)
    for _ in range(5000):
        pattern = "".join(rng.choice("ab*?[]!-") for _ in range(rng.randrange(1, 9)))
        name = "".join(rng.choice("ab-]![") for _ in range(rng.randrange(9)))
        if not pattern.startswith("!"):  # which takes back in, in a .siftignore
            assert ignores(pattern, name) == fnmatch.fnmatchcase(name, pattern), (pattern, name)


def test_case_as_the_file_system_takes_it():
    assert ignores("Drafts/", "drafts", True, ignore_case=True)
    assert not ignores("Drafts/", "drafts", True, ignore_case=False)


def test_names_in_any_unicode_form_where_case_does_not_count():
    # macOS hands out names decomposed, u and a combining diaeresis for ü, while a .siftignore is typed composed.
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "Entwürfe")
    assert ignores("entwürfe/", decomposed, True, ignore_case=True)
    assert ignores(decomposed + "/", "ENTWÜRFE", True, ignore_case=True)
    assert not ignores("Entwürfe/", decomposed, True, ignore_case=False)  # elsewhere names are as they are, as in git


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
    (tmp_path / ".siftignore").write_bytes(
        "drafts/\r\n".encode("utf-16")
    )  # `"drafts/" > .siftignore` in PowerShell 5.1
    assert Ignore.of(tmp_path)("drafts", folder=True)


def test_a_siftignore_sift_cannot_read_is_an_error(tmp_path):
    # Indexing everything would put the very files it should keep out into the index.
    (tmp_path / ".siftignore").mkdir()
    with pytest.raises(OSError, match=r"\.siftignore"):
        Ignore.of(tmp_path)


def test_a_pattern_of_many_stars_stays_fast():
    # As a regex, nine stars took 73 s against a name of 50 a's: each star tried every split of the rest.
    import time

    ignore, name = Ignore(["*a" * 9 + "*b", "x/" + "*a" * 9 + "*b/**/*c"]), "a" * 200
    start = time.perf_counter()
    assert not ignore(name)
    assert not ignore("x/" + name + "/" + name)
    assert time.perf_counter() - start < 1


def test_many_patterns_stay_fast():
    ignore = Ignore([f"folder{i}/**/*.tmp" for i in range(200)] + ["*" * 10_000 + "x"])
    assert not ignore("a/" * 200 + "b.md")
