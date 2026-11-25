import dataclasses
import os
from pathlib import Path

import pytest

from sift.interfaces.cli.main import parser
from sift.settings import Settings
from tests.surface.commands import ran

TAKING_A_FILE = [("get", "{}"), ("links", "{}"), ("mentions", "{}"), ("similar", "{}"), ("index", "--file", "{}")]


def _taken(argv: tuple[str, ...], given: dict[str, object]) -> object:
    if "--file" in argv:
        return given["files"]
    return given["ids"] if argv[0] == "get" else [given["id"]]


@pytest.mark.parametrize("argv", TAKING_A_FILE, ids=[each[0] for each in TAKING_A_FILE])
@pytest.mark.parametrize(
    "written, place",
    [("~/notes/a.md", ("user", "notes", "a.md")), ("./a.md", ("work", "a.md")), ("../x/a.md", ("x", "a.md"))],
)
def test_a_path_is_read_alike_by_every_command_taking_a_file_or_an_id(
    tmp_path: Path, argv: tuple[str, ...], written: str, place: tuple[str, ...]
) -> None:
    built = parser(tmp_path / "user", tmp_path / "work")
    given = vars(built.parse_args([each.format(written) for each in argv]))
    assert _taken(argv, given) == [os.path.normpath(tmp_path.joinpath(*place))]


@pytest.mark.parametrize("argv", TAKING_A_FILE, ids=[each[0] for each in TAKING_A_FILE])
def test_an_id_that_is_no_path_from_here_stays_an_id(tmp_path: Path, argv: tuple[str, ...]) -> None:
    built = parser(tmp_path / "user", tmp_path / "work")
    assert _taken(argv, vars(built.parse_args([each.format("notes/a.md") for each in argv]))) == ["notes/a.md"]


def test_a_file_is_named_from_the_user_s_folder_or_the_current_one_by_index_and_get(settings: Settings) -> None:
    assert settings.user is not None
    notes = settings.user / "notes"
    notes.mkdir()
    (notes / "a.md").write_bytes(b"# A\n\nPigment.\n")
    assert ran(settings, "add", "~/notes")[0] == 0
    here = dataclasses.replace(settings, cwd=notes)
    for argv in (("index", "--file", "~/notes/a.md"), ("get", "./a.md"), ("get", "~/notes/a.md")):
        code, _, err = ran(here, *argv)
        assert (argv, code, err.startswith("error")) == (argv, 0, False)


def test_a_file_from_the_user_s_folder_needs_one(settings: Settings) -> None:
    code, _, err = ran(dataclasses.replace(settings, user=None), "get", "~/notes/a.md")
    assert (code, "no home folder is set: set SIFT_HOME to an absolute folder" in err) == (2, True)
