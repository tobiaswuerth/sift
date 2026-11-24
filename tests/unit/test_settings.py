from pathlib import Path

import pytest

from sift.api.errors import InvalidRequestError
from sift.domain.text.scans import OcrSetting
from sift.settings import (
    OCR_REMEDY,
    UPDATE_CHECK_REMEDY,
    Settings,
    from_environment,
    parsed,
    user_folder,
    whole_path,
)


@pytest.fixture
def user(tmp_path: Path) -> Path:
    return tmp_path / "user"


@pytest.fixture
def cwd(tmp_path: Path) -> Path:
    return tmp_path / "work"


def test_the_home_is_a_folder_in_the_user_s_unless_set(user: Path, cwd: Path) -> None:
    own = user / ".local" / "state" / "sift"
    assert parsed({}, user, cwd, "linux") == Settings(home=user / ".sift", user=user, cwd=cwd, state=own)
    assert parsed({"SIFT_HOME": str(cwd / "elsewhere")}, user, cwd).home == cwd / "elsewhere"


def test_the_user_s_own_folder_is_local_app_data_on_windows_and_the_xdg_state_folder_elsewhere(
    user: Path, cwd: Path
) -> None:
    assert parsed({}, user, cwd, "win32").state == user / "AppData" / "Local" / "sift"
    assert parsed({"LOCALAPPDATA": str(cwd / "local")}, user, cwd, "win32").state == cwd / "local" / "sift"
    assert parsed({"LOCALAPPDATA": str(cwd / "local")}, user, cwd, "darwin").state == user / ".local/state/sift"
    assert parsed({"XDG_STATE_HOME": str(cwd / "kept")}, user, cwd, "darwin").state == cwd / "kept" / "sift"
    assert parsed({"XDG_STATE_HOME": "kept"}, user, cwd, "linux").state == user / ".local/state/sift"
    home = str(cwd / "home")
    assert parsed({"SIFT_HOME": home}, None, cwd, "linux").state is None
    assert parsed({"SIFT_HOME": home, "XDG_STATE_HOME": str(cwd)}, None, cwd, "linux").state == cwd / "sift"


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


def test_scans_are_read_on_a_gpu_unless_set_and_a_setting_sift_knows_not_says_what_it_takes(
    user: Path, cwd: Path
) -> None:
    assert [parsed(asked, user, cwd).ocr for asked in ({}, {"SIFT_OCR": ""}, {"SIFT_OCR": " ON "})] == [
        OcrSetting.AUTO,
        OcrSetting.AUTO,
        OcrSetting.ON,
    ]
    with pytest.raises(InvalidRequestError, match="SIFT_OCR is 'maybe'") as refused:
        parsed({"SIFT_OCR": "maybe"}, user, cwd)
    assert refused.value.remedy == OCR_REMEDY


def test_sift_asks_pypi_for_a_newer_sift_only_when_switched_on_and_never_offline(user: Path, cwd: Path) -> None:
    asked: tuple[dict[str, str], ...] = (
        {},
        {"SIFT_UPDATE_CHECK": " On "},
        {"SIFT_UPDATE_CHECK": "off"},
        {"SIFT_UPDATE_CHECK": "on", "HF_HUB_OFFLINE": "1"},
        {"SIFT_UPDATE_CHECK": "on", "HF_HUB_OFFLINE": "yes"},
        {"SIFT_UPDATE_CHECK": "on", "HF_HUB_OFFLINE": "0"},
    )
    assert [parsed(each, user, cwd).update_check for each in asked] == [False, True, False, False, False, True]
    with pytest.raises(InvalidRequestError, match="SIFT_UPDATE_CHECK is 'daily'") as refused:
        parsed({"SIFT_UPDATE_CHECK": "daily"}, user, cwd)
    assert refused.value.remedy == UPDATE_CHECK_REMEDY


def test_the_gpus_cuda_may_use_are_kept_as_set_even_empty_which_hides_them_all(user: Path, cwd: Path) -> None:
    asked: tuple[dict[str, str], ...] = ({}, {"CUDA_VISIBLE_DEVICES": "-1"}, {"CUDA_VISIBLE_DEVICES": ""})
    assert [parsed(each, user, cwd).gpus for each in asked] == [None, "-1", ""]


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


def test_a_home_that_is_a_file_is_refused(tmp_path: Path, user: Path, cwd: Path) -> None:
    taken = tmp_path / "taken"
    taken.write_text("not a folder", encoding="utf-8")
    user.mkdir()
    (user / ".sift").write_text("not a folder", encoding="utf-8")
    for asked, path in (({"SIFT_HOME": str(taken)}, taken), ({}, user / ".sift")):
        with pytest.raises(InvalidRequestError) as refused:
            parsed(asked, user, cwd)
        assert (refused.value.message, refused.value.remedy) == (
            f"the home {path} is no folder",
            "set SIFT_HOME to an absolute folder",
        )


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


def _too_deep() -> Path:
    raise RecursionError("maximum recursion depth exceeded")


def test_a_user_folder_that_cannot_be_found_is_none() -> None:
    assert user_folder(_nowhere) is None
    with pytest.raises(RecursionError):
        user_folder(_too_deep)
    assert user_folder() == Path.home()
