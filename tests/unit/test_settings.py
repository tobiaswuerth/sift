from pathlib import Path

import pytest

from sift.api.errors import InvalidRequestError
from sift.settings import Settings, from_environment, parsed, user_folder, whole_path


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


def test_the_models_are_embeddinggemma_and_gte_s_reranker_unless_set(user: Path, cwd: Path) -> None:
    assert parsed({}, user, cwd).embed_model == "google/embeddinggemma-2"
    assert parsed({"SIFT_EMBED_MODEL": "owner/model@v2"}, user, cwd).embed_model == "owner/model@v2"
    assert parsed({}, user, cwd).rerank_model == "Alibaba-NLP/gte-reranker-modernbert-base"
    assert parsed({"SIFT_RERANK_MODEL": "owner/ranker"}, user, cwd).rerank_model == "owner/ranker"


def test_an_empty_variable_counts_as_unset(user: Path, cwd: Path) -> None:
    assert parsed({"SIFT_HOME": "", "SIFT_EMBED_MODEL": "", "SIFT_RERANK_MODEL": ""}, user, cwd) == parsed(
        {}, user, cwd
    )


@pytest.mark.parametrize(
    "written, place",
    [("~", ("user",)), ("~/data", ("user", "data")), ("data", ("work", "data")), ("../data", ("work", "..", "data"))],
)
def test_a_path_is_made_whole_from_the_user_s_folder_or_the_current_one(
    tmp_path: Path, user: Path, cwd: Path, written: str, place: tuple[str, ...]
) -> None:
    assert whole_path(written, user, cwd) == tmp_path.joinpath(*place)


@pytest.mark.parametrize("written", ["~", "~/data"])
def test_the_home_may_be_named_from_the_user_s_folder(user: Path, cwd: Path, written: str) -> None:
    assert parsed({"SIFT_HOME": written}, user, cwd).home == user.joinpath(*Path(written).parts[1:])


@pytest.mark.parametrize("written", ["data", "../data"])
def test_a_relative_home_is_refused(user: Path, cwd: Path, written: str) -> None:
    with pytest.raises(InvalidRequestError, match="SIFT_HOME is no absolute folder") as refused:
        parsed({"SIFT_HOME": written}, user, cwd)
    assert refused.value.remedy == "set SIFT_HOME to an absolute folder"


def test_without_a_user_folder_sift_home_alone_will_do(cwd: Path) -> None:
    assert parsed({"SIFT_HOME": str(cwd / "home")}, None, cwd).home == cwd / "home"
    for asked in ({}, {"SIFT_HOME": "~/home"}):
        with pytest.raises(InvalidRequestError, match="no home folder is set"):
            parsed(asked, None, cwd)


def test_a_whole_path_stays_as_it_is(tmp_path: Path, user: Path, cwd: Path) -> None:
    assert whole_path(str(tmp_path / "~" / "x"), user, cwd) == tmp_path / "~" / "x"


def test_settings_come_from_this_process() -> None:
    found = from_environment()
    assert (found.user, found.cwd) == (Path.home(), Path.cwd())
    assert found.home.is_absolute()


def test_no_test_reaches_the_home_of_whoever_runs_them(no_home_but_a_temporary_one: Path) -> None:
    assert from_environment().home == no_home_but_a_temporary_one
    assert not no_home_but_a_temporary_one.exists()


def _nowhere() -> Path:
    raise RuntimeError("Could not determine home directory.")


def test_a_user_folder_that_cannot_be_found_is_none() -> None:
    assert user_folder(_nowhere) is None
    assert user_folder() == Path.home()
