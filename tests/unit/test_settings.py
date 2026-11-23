from pathlib import Path

import pytest

from sift.settings import Settings, from_environment, parsed, whole_path


@pytest.fixture
def user(tmp_path: Path) -> Path:
    return tmp_path / "user"


@pytest.fixture
def cwd(tmp_path: Path) -> Path:
    return tmp_path / "work"


def test_the_home_is_a_folder_in_the_user_s_unless_set(user: Path, cwd: Path) -> None:
    assert parsed({}, user, cwd) == Settings(home=user / ".sift", user=user, cwd=cwd)
    assert parsed({"SIFT_HOME": str(cwd / "elsewhere")}, user, cwd).home == cwd / "elsewhere"


def test_colour_is_left_out_where_no_color_is_set_or_the_terminal_is_dumb(user: Path, cwd: Path) -> None:
    assert (parsed({}, user, cwd).no_color, parsed({}, user, cwd).dumb) == (False, False)
    assert parsed({"NO_COLOR": "1"}, user, cwd).no_color
    assert not parsed({"NO_COLOR": ""}, user, cwd).no_color
    assert parsed({"TERM": "dumb"}, user, cwd).dumb
    assert not parsed({"TERM": "xterm-256color"}, user, cwd).dumb


def test_the_embedding_model_is_embeddinggemma_unless_set(user: Path, cwd: Path) -> None:
    assert parsed({}, user, cwd).embed_model == "google/embeddinggemma-2"
    assert parsed({"SIFT_EMBED_MODEL": "owner/model@v2"}, user, cwd).embed_model == "owner/model@v2"


def test_an_empty_variable_counts_as_unset(user: Path, cwd: Path) -> None:
    assert parsed({"SIFT_HOME": "", "SIFT_EMBED_MODEL": ""}, user, cwd) == parsed({}, user, cwd)


@pytest.mark.parametrize(
    "written, place",
    [("~", ("user",)), ("~/data", ("user", "data")), ("data", ("work", "data")), ("../data", ("work", "..", "data"))],
)
def test_a_path_is_made_whole_from_the_user_s_folder_or_the_current_one(
    tmp_path: Path, user: Path, cwd: Path, written: str, place: tuple[str, ...]
) -> None:
    assert whole_path(written, user, cwd) == tmp_path.joinpath(*place)
    assert parsed({"SIFT_HOME": written}, user, cwd).home == tmp_path.joinpath(*place)


def test_a_whole_path_stays_as_it_is(tmp_path: Path, user: Path, cwd: Path) -> None:
    assert whole_path(str(tmp_path / "~" / "x"), user, cwd) == tmp_path / "~" / "x"


def test_settings_come_from_this_process() -> None:
    found = from_environment()
    assert (found.user, found.cwd) == (Path.home(), Path.cwd())
    assert found.home.is_absolute()


def test_no_test_reaches_the_home_of_whoever_runs_them(no_home_but_a_temporary_one: Path) -> None:
    assert from_environment().home == no_home_but_a_temporary_one
    assert not no_home_but_a_temporary_one.exists()
