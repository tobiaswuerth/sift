import itertools
from typing import SupportsIndex, final, override

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.text.chunks import MOST_CHARACTERS, chunks, cuts


def pieces(text: str, first_line: int = 1) -> list[tuple[int, int, str]]:
    return [(c.first_line, c.last_line, text[c.start : c.start + c.length]) for c in chunks(text, first_line)]


def test_a_short_text_is_one_chunk() -> None:
    assert pieces("# T\n\nshort body\n", 5) == [(5, 7, "# T\n\nshort body")]


def test_lines_are_those_of_the_text_not_of_blank_lines_around_it() -> None:
    assert pieces("\n\n# T\n\nbody\n\n\n", 5) == [(7, 9, "# T\n\nbody")]


@pytest.mark.parametrize("text", ["", "\n\n  \n"])
def test_an_empty_text_has_no_chunks(text: str) -> None:
    assert pieces(text) == []


def test_chunks_carry_line_numbers_and_overlap() -> None:
    text = "".join(f"line {i} " + "x" * 90 + "\n" for i in range(100))
    found = pieces(text)
    assert len(found) > 1
    assert (found[0][0], found[-1][1]) == (1, 100)
    assert all(len(passage) <= MOST_CHARACTERS for _, _, passage in found)
    assert found[1][0] <= found[0][1]


def test_a_heading_starts_a_chunk_without_overlap() -> None:
    found = pieces("# One\n" + "a " * 900 + "\n# Two\nsecond section\n")
    assert [passage.split("\n")[0] for _, _, passage in found] == ["# One", "# Two"]
    assert found[1][0] == found[0][1] + 1


@pytest.mark.parametrize(
    "line",
    ["# a comment in code, not a heading", "#include <stdio.h>", "#tag as Obsidian writes it", "####### seven"],
)
def test_a_line_that_only_starts_with_a_hash_is_no_heading(line: str) -> None:
    filler = "".join(f"x{i} = {i}  # filler filler filler filler\n" for i in range(45))
    fenced = line.startswith("# a comment")
    text = ("```python\n" if fenced else "") + filler + line + "\nthe rest\n" + ("```\n" if fenced else "")
    assert len(filler) > MOST_CHARACTERS // 2
    assert len(text) < MOST_CHARACTERS
    assert [passage for _, _, passage in pieces(text)] == [text.strip()]


def test_the_end_of_a_fence_ends_its_code() -> None:
    code = "```\n" + "".join(f"x{i} = {i}  # filler filler filler filler\n" for i in range(45)) + "```\n"
    assert [passage.split("\n")[0] for _, _, passage in pieces(code + "# After the code\nmore\n")] == [
        "```",
        "# After the code",
    ]


def test_long_lines_never_make_a_chunk_too_long() -> None:
    text = "short line\n" * 20 + "y" * (MOST_CHARACTERS - 50) + "\n" + "z " * 3000
    assert all(len(passage) <= MOST_CHARACTERS for _, _, passage in pieces(text))


def test_unbroken_text_is_cut_at_the_most_characters() -> None:
    assert [len(passage) for _, _, passage in pieces("x" * (MOST_CHARACTERS * 2 + 10))] == [
        MOST_CHARACTERS,
        MOST_CHARACTERS,
        10,
    ]


def test_a_long_line_is_searched_once_for_where_to_cut_it() -> None:
    searched: list[int] = []

    @final
    class Line(str):
        __slots__ = ()

        @override
        def rfind(self, sub: str, start: SupportsIndex | None = None, end: SupportsIndex | None = None, /) -> int:
            searched.append(len(range(*slice(start, end).indices(len(self)))))
            return super().rfind(sub, start, end)

    length = 100 * MOST_CHARACTERS + 10
    found = list(cuts(Line("y" * length)))
    assert len(found) == -(-length // MOST_CHARACTERS)
    assert (found[0][0], found[-1][1]) == (0, length)
    assert all(before[1] == after[0] for before, after in itertools.pairwise(found))
    assert 0 < sum(searched) <= length


def test_a_line_is_cut_after_its_last_space_within_the_most_characters() -> None:
    line = "a" * (MOST_CHARACTERS - 10) + " " + "b" * 20
    assert list(cuts(line)) == [(0, MOST_CHARACTERS - 9), (MOST_CHARACTERS - 9, len(line))]
    assert list(cuts("short")) == [(0, 5)]


def test_a_line_of_many_megabytes_is_cut_whole() -> None:
    length = 16_000_000
    found = chunks("y" * length)
    assert len(found) == -(-length // MOST_CHARACTERS)
    assert (found[0].start, found[-1].start + found[-1].length) == (0, length)


@pytest.mark.parametrize(
    "separator", ["\x0b", "\x0c", "\x1c", "\N{NEXT LINE}", "\N{LINE SEPARATOR}", "\N{PARAGRAPH SEPARATOR}"]
)
def test_only_a_newline_ends_a_line(separator: str) -> None:
    assert pieces(f"one\n{separator}two\nthree\n") == [(1, 3, f"one\n{separator}two\nthree")]


def test_positions_are_of_the_whole_text_from_the_offset_given() -> None:
    (chunk,) = chunks("body\n", 4, 20)
    assert (chunk.first_line, chunk.start, chunk.length) == (4, 20, 4)


WRITTEN = st.sampled_from("ab #`~>-\t\r\u00e9\u2028\u0085")
lines = st.one_of(st.text(WRITTEN, max_size=200), st.text(WRITTEN, min_size=3000, max_size=3200)).map(
    lambda line: line + "\n"
)


@given(st.lists(lines, max_size=80).map("".join), st.integers(1, 50))
def test_chunks_are_stripped_spans_of_their_lines_in_order_within_the_limit(text: str, first_line: int) -> None:
    found = chunks(text, first_line)
    for chunk in found:
        passage = text[chunk.start : chunk.start + chunk.length]
        assert passage == passage.strip() != ""
        assert len(passage) <= MOST_CHARACTERS
        assert chunk.first_line == first_line + text.count("\n", 0, chunk.start)
        assert chunk.last_line == chunk.first_line + passage.count("\n")
    assert [chunk.start for chunk in found] == sorted(chunk.start for chunk in found)
    covered = {line for chunk in found for line in range(chunk.first_line, chunk.last_line + 1)}
    written = {number for number, line in enumerate(text.split("\n"), first_line) if line.strip()}
    assert written <= covered
