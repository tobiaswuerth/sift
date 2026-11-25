import json
from pathlib import Path

import pytest

from sift.api.errors import InvalidRequestError
from sift.interfaces.cli.main import main
from sift.settings import VARIABLES, Settings
from tests.surface.commands import indexed_notes, ran


def test_an_option_is_given_whole_never_abbreviated(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    code, out, err = ran(settings, "search", "reward", "--js")
    assert (code, out) == (2, "")
    assert "unrecognized arguments: --js" in err
    assert "see sift search --help" in err


@pytest.mark.parametrize(
    "argv",
    [
        ("search", "reward", "--collection", "notes", "--filter", "year!=1"),
        ("search", "reward", "-c", "notes", "-k", "1"),
    ],
)
def test_an_option_given_again_for_each_value_is_named_for_one(
    settings: Settings, notes: Path, argv: tuple[str, ...]
) -> None:
    indexed_notes(settings, notes)
    code, out, _ = ran(settings, *argv, "--json")
    assert (code, json.loads(out)["searched"]) == (0, 1)


@pytest.mark.parametrize("argv", [("search", "reward", "--collections", "notes"), ("search", "reward", "--k", "1")])
def test_the_plural_and_doubled_spellings_of_options_are_gone(settings: Settings, argv: tuple[str, ...]) -> None:
    assert ran(settings, *argv)[0] == 2


def test_a_query_that_starts_with_a_dash_is_said_to_go_after_two_dashes(settings: Settings) -> None:
    code, _, err = ran(settings, "search", "-quokka")
    assert code == 2
    assert "  fix  a value that starts with a dash goes after --, e.g. sift search -- -value" in err


@pytest.mark.parametrize(
    "argv, option",
    [
        (("index", "--file"), "--file"),
        (("search", "x", "-k"), "-k"),
        (("search", "x", "-c", "notes", "-c"), "-c"),
        (("search", "x", "-k", "--json"), "-k"),
        (("--color",), "--color"),
    ],
)
def test_an_option_given_last_without_a_value_is_told_to_take_one(
    settings: Settings, argv: tuple[str, ...], option: str
) -> None:
    code, _, err = ran(settings, *argv)
    assert code == 2
    prog = "sift" if argv[0].startswith("-") else f"sift {argv[0]}"
    assert f"  fix  give {option} a value; see {prog} --help" in err


@pytest.mark.parametrize(
    "argv, kind",
    [
        (("search", "--json", "--frobnicate", "x"), "invalid_request"),
        (("get", "notes/name\udce9.md", "--json"), "invalid_request"),
    ],
)
def test_an_error_asked_for_in_json_is_json_whatever_refused_it(
    settings: Settings, argv: tuple[str, ...], kind: str
) -> None:
    code, out, _ = ran(settings, *argv)
    assert (code, json.loads(out)["error"]["kind"]) == (2, kind)


def test_a_json_after_two_dashes_is_a_value_and_asks_for_no_json(settings: Settings) -> None:
    code, out, _ = ran(settings, "search", "--frobnicate", "--", "--json")
    assert (code, out) == (2, "")


def test_settings_refused_are_json_when_json_is_asked(capsys: pytest.CaptureFixture[str]) -> None:
    def refused() -> Settings:
        raise InvalidRequestError("SIFT_OCR is 'maybe'", "set it to on, off or auto", "SIFT_OCR")

    with pytest.raises(SystemExit) as ended:
        main(["status", "--json"], refused)
    said = capsys.readouterr()
    assert (ended.value.code, json.loads(said.out)["error"]["field"]) == (2, "SIFT_OCR")


def test_help_names_every_setting_and_what_it_sets(settings: Settings) -> None:
    code, out, _ = ran(settings, "--help")
    lines = out.split("settings, read from the environment:\n", 1)[1].splitlines()
    assert (code, [line.split()[0] for line in lines]) == (0, list(VARIABLES))
    assert all(line.endswith(VARIABLES[line.split()[0]]) for line in lines)
