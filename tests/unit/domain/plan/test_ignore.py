import fnmatch
import sys
import unicodedata
from collections.abc import Callable
from types import BuiltinFunctionType, FrameType

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from sift.domain.plan import ignore as ignore_module
from sift.domain.plan.ignore import NOTHING_LEFT_OUT, excludes, ignore_of, in_order, left_out

PATTERN_CHARACTERS = "ab*?[]!-"
NAME_CHARACTERS = "ab-]!["
SEARCHES = frozenset({"find", "rfind", "index", "rindex", "search", "match"})


def ignores(lines: str, path: str, *, folder: bool = False, caseless: bool = False) -> bool:
    return left_out(ignore_of(lines.split("\n"), caseless=caseless), path, folder=folder)


@pytest.mark.parametrize(
    "lines, path, folder, ignored",
    [
        ("*.log", "a.log", False, True),
        ("*.log", "x/y/a.log", False, True),
        ("*.log", "a.log.md", False, False),
        ("drafts/", "drafts", True, True),
        ("drafts/", "x/drafts", True, True),
        ("drafts/", "drafts", False, False),
        ("drafts", "drafts", False, True),
        ("/drafts", "drafts", True, True),
        ("/drafts", "x/drafts", True, False),
        ("docs/*.md", "docs/a.md", False, True),
        ("docs/*.md", "x/docs/a.md", False, False),
        ("docs/*.md", "docs/sub/a.md", False, False),
        ("**/temp", "temp", True, True),
        ("**/temp", "a/b/temp", True, True),
        ("a/**/b", "a/b", False, True),
        ("a/**/b", "a/x/y/b", False, True),
        ("**/private/**", "x/private/a.md", False, True),
        ("**/private/**", "private/a.md", False, True),
        ("**/private/**", "x/private", True, False),
        ("a/**/b/**/c", "a/x/b/y/z/c", False, True),
        ("a/**/b/**/c", "a/x/c", False, False),
        ("logs/**", "logs/a/b.md", False, True),
        ("logs/**", "logs", True, False),
        ("**", "any/thing.md", False, True),
        ("notes*", "notes", False, True),
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
        ("what\\?.md", "what?.md", False, True),
        ("what\\?.md", "whatx.md", False, False),
        ("a\\\\?.md", "a\\x.md", False, True),
        ("\\*.md", "x.md", False, False),
        ("trailing.md   ", "trailing.md", False, True),
        ("space\\ ", "space ", False, True),
        ("a\\\\  ", "a\\", False, True),
        ("a\\\\\\  ", "a\\ ", False, True),
        ("a.md\n\n\n", "a.md", False, True),
        ("*.md\n!keep.md", "keep.md", False, False),
        ("*.md\n!keep.md", "other.md", False, True),
        ("!keep.md\n*.md", "keep.md", False, True),
        ("", "a.md", False, False),
        ("/", "a", True, False),
        ("!", "a.md", False, False),
    ],
)
def test_patterns_mean_what_they_mean_in_gitignore(lines: str, path: str, *, folder: bool, ignored: bool) -> None:
    assert ignores(lines, path, folder=folder) == ignored


def test_nothing_is_left_out_without_patterns() -> None:
    assert not left_out(NOTHING_LEFT_OUT, "a.md")
    assert ignore_of((), caseless=False) == NOTHING_LEFT_OUT


def test_a_range_backwards_holds_its_first_character_alone() -> None:
    assert [ignores(lines, name) for lines, name in (("[z-a].md", "b.md"), ("[z-a].md", "a.md"))] == [False, False]
    assert [ignores("[z-a].md", "z.md"), ignores("[z-ab].md", "b.md"), ignores("[!z-a].md", "b.md")] == [True] * 3
    assert not ignores("[!z-a].md", "z.md")


def backwards(pattern: str) -> bool:
    return any(
        first > last for first, dash, last in zip(pattern, pattern[1:], pattern[2:], strict=False) if dash == "-"
    )


@given(st.text(PATTERN_CHARACTERS, min_size=1, max_size=8), st.text(NAME_CHARACTERS, max_size=8))
def test_a_name_s_pattern_matches_as_fnmatch_matches_it(pattern: str, name: str) -> None:
    assume(not pattern.startswith("!") and not backwards(pattern))
    assert ignores(pattern, name) == fnmatch.fnmatchcase(name, pattern)


def test_case_counts_unless_the_file_system_folds_it() -> None:
    assert ignores("Drafts/", "drafts", folder=True, caseless=True)
    assert not ignores("Drafts/", "drafts", folder=True, caseless=False)


def test_names_in_any_unicode_form_match_where_case_does_not_count() -> None:
    decomposed = unicodedata.normalize("NFD", "Entwürfe")
    assert ignores("entwürfe/", decomposed, folder=True, caseless=True)
    assert ignores(decomposed + "/", "ENTWÜRFE", folder=True, caseless=True)
    assert not ignores("Entwürfe/", decomposed, folder=True, caseless=False)


def test_groups_between_wildcards_are_each_looked_for_once_from_where_the_one_before_ended() -> None:
    looked: list[int] = []

    def at(group: str, whole: str, start: int) -> bool:
        looked.append(start)
        return whole.startswith(group, start)

    def find(group: str, whole: str, start: int) -> int:
        looked.append(start)
        return whole.find(group, start)

    groups = ["", *["a"] * 9, "b"]
    test = in_order(groups, at, find, len)
    assert not test("a" * 200)
    assert len(looked) <= len(groups)
    assert looked == sorted(looked)


def test_a_pattern_of_many_stars_matches_long_names_at_once() -> None:
    ignore = ignore_of(["*a" * 9 + "*b", "x/" + "*a" * 9 + "*b/**/*c"], caseless=False)
    assert not left_out(ignore, "a" * 200)
    assert not left_out(ignore, "x/" + "a" * 200 + "/" + "a" * 200)
    assert left_out(ignore, "x/" + "a" * 50 + "b/y/zc")


def test_many_patterns_and_deep_paths_stay_fast() -> None:
    ignore = ignore_of([f"folder{n}/**/*.tmp" for n in range(200)] + ["*" * 10_000 + "x"], caseless=False)
    assert not left_out(ignore, "a/" * 200 + "b.md")


def test_a_file_in_a_folder_left_out_is_excluded_though_taken_back() -> None:
    ignore = ignore_of(["private/", "!private/keep.md"], caseless=False)
    assert [excludes(ignore, path) for path in ("private/keep.md", "x/private/a/b.md", "public/keep.md")] == [
        True,
        True,
        False,
    ]
    assert excludes(ignore_of(["*.md"], caseless=False), "a/b.md")


def _searches(run: Callable[[], object]) -> int:
    held: list[str] = []

    def profiled(frame: FrameType, event: str, arg: object) -> None:
        mine = frame.f_code.co_filename == ignore_module.__file__
        if mine and event == "c_call" and isinstance(arg, BuiltinFunctionType) and arg.__name__ in SEARCHES:
            held.append(arg.__name__)

    sys.setprofile(profiled)
    try:
        _ = run()
    finally:
        sys.setprofile(None)
    return len(held)


@pytest.mark.parametrize(
    "pattern, name", [("[" * 20_000, "[" * 20_000), ("[a" * 10_000 + "]", "a"), ("x[" * 10_000, "x[" * 10_000)]
)
def test_brackets_that_never_close_are_looked_for_once_however_many(pattern: str, name: str) -> None:
    assert _searches(lambda: ignore_of([pattern], caseless=False)) == 0
    assert left_out(ignore_of([pattern], caseless=False), name)
