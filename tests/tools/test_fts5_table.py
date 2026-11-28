from pathlib import Path

import pytest

from tools import fts5_table
from tools.fts5_table import Table, compared, fts5_tokens, lines, module_source, probed, spans


def test_fts5_tokenizes_each_text_on_its_own() -> None:
    assert fts5_tokens(["Hello, Wörld", "", "a-b"]) == [["hello", "world"], [], ["a", "b"]]


def test_a_probe_tells_separators_from_folds() -> None:
    table = probed([ord(c) for c in "a,A\N{LATIN SMALL LETTER E WITH ACUTE} \N{COMBINING GRAVE ACCENT}"])
    assert table == Table(separators=(ord(","), ord(" ")), folds=((ord("A"), "a"), (0xE9, "e"), (0x300, "")))


@pytest.mark.parametrize("tokens", [["q", "x", "q"], ["qabq"], ["x"], []])
def test_a_tokenizer_the_table_cannot_describe_is_refused(tokens: list[str]) -> None:
    with pytest.raises(ValueError, match=r"U\+0041 between two letters tokenizes as .* the table cannot say"):
        probed([0x41], lambda texts: [tokens for _ in texts])


def test_points_become_spans_of_neighbours() -> None:
    assert spans([1, 2, 3, 7, 9, 10]) == ["0001-0003", "0007", "0009-000a"]


def test_words_are_wrapped_into_lines_of_a_tuple() -> None:
    assert lines("X", ["a" * 60, "b" * 60]) == ["X = (", f'    "{"a" * 60}",', f'    "{"b" * 60}",', ")"]


def test_the_module_lists_separators_then_folds() -> None:
    source = module_source(Table(separators=(1, 2), folds=((0x41, "a"), (0x300, ""))))
    assert source == 'SEPARATORS = (\n    "0001-0002",\n)\nFOLDS = (\n    "0041=0061 0300=",\n)\n'


def test_a_table_is_written_and_then_compares_equal(tmp_path: Path) -> None:
    module = tmp_path / "table.py"
    assert compared(module, "X = 1\n", write=True) == 0
    assert compared(module, "X = 1\n", write=False) == 0


def test_a_table_that_differs_fails_and_says_how_to_write_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = tmp_path / "table.py"
    module.write_text("X = 0\n", encoding="utf-8")
    assert compared(module, "X = 1\n", write=False) == 1
    assert "python -m tools fts5 --write" in capsys.readouterr().out


def test_the_committed_table_is_what_fts5_tokenizes() -> None:
    assert fts5_table.main([]) == 0
