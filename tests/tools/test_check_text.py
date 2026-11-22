import pytest

from tools import check_text
from tools.check_text import line_length, long_prose, machines

MODELS = ("RT" + "X 4090", "GeFor" + "ce", "Ryz" + "en 9", "Xe" + "on", "EP" + "YC", "Core " + "i7", "i9" + "-13900K")


@pytest.mark.parametrize("model", MODELS)
def test_a_machine_s_model_is_found_with_its_line(model: str) -> None:
    (found,) = machines("docs/a.md", f"Measured.\nOn a {model} it took 3 s.\n")
    assert (found.line, found.problem) == (2, "names a machine's model")


def test_hardware_named_by_kind_passes() -> None:
    assert list(machines("docs/a.md", "On a GPU with 24 GB and 8 CPU threads.\n")) == []


def test_prose_longer_than_the_width_is_found() -> None:
    text = f"{'a' * 10}\n{'b' * 11}\n"
    (found,) = long_prose("a.md", text, 10)
    assert (found.line, found.problem) == (2, "prose of 11 characters on a line, more than 10")


@pytest.mark.parametrize(
    "line",
    [
        "| a table | with a long row |",
        "![an image](https://example.com/image.png)",
        "[![a badge](https://example.com/b.svg)](https://example.com)",
        "[1.0]: https://example.com/compare/v0.9...v1.0",
    ],
)
def test_tables_images_and_addresses_run_on(line: str) -> None:
    assert list(long_prose("a.md", f"{line}\n", 10)) == []


def test_code_blocks_run_on_and_prose_after_them_does_not() -> None:
    text = "```bash\nuv run --only-group gate nox\n```\nprose that is long\n"
    assert [found.line for found in long_prose("a.md", text, 10)] == [4]


def test_the_width_is_the_formatter_s() -> None:
    assert line_length("[tool.ruff]\nline-length = 88\n") == 88
    with pytest.raises(TypeError, match="line-length"):
        line_length("[tool.ruff]\n")


def test_the_repository_s_text_is_publishable() -> None:
    assert check_text.main([]) == 0
