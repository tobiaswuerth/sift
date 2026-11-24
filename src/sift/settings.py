import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from sift.api.errors import InvalidRequestError
from sift.domain.text.scans import OcrSetting
from sift.inference.hub import EMBED_MODEL, RERANK_MODEL

HOME = "SIFT_HOME"
EMBED_MODEL_VARIABLE = "SIFT_EMBED_MODEL"
RERANK_MODEL_VARIABLE = "SIFT_RERANK_MODEL"
OCR_VARIABLE = "SIFT_OCR"
GPUS_VARIABLE = "CUDA_VISIBLE_DEVICES"
OCR_REMEDY = (
    f"set {OCR_VARIABLE} to auto (on a GPU, the default), on (also on the CPU, which takes minutes a page) or off"
)
UPDATE_CHECK_VARIABLE = "SIFT_UPDATE_CHECK"
UPDATE_CHECK_REMEDY = f"set {UPDATE_CHECK_VARIABLE} to on, to be told of a newer sift once a day, or to off"
SWITCHES = MappingProxyType({"on": True, "off": False})
OFFLINE_VARIABLE = "HF_HUB_OFFLINE"
OFFLINE = frozenset({"1", "on", "yes", "true"})
NO_COLOR = "NO_COLOR"
TERM = "TERM"
DUMB = "dumb"
DEFAULT_HOME = ".sift"
USER = "~"
NO_HOME = "no home folder is set"
STATE_VARIABLES = MappingProxyType({"win32": "LOCALAPPDATA"})
STATE_VARIABLE = "XDG_STATE_HOME"
STATE_FOLDERS = MappingProxyType({"win32": ("AppData", "Local")})
STATE_FOLDER = (".local", "state")
OWN_FOLDER = "sift"
HOME_REMEDY = f"set {HOME} to an absolute folder"


@dataclass(frozen=True, slots=True)
class Settings:
    home: Path
    user: Path | None
    cwd: Path
    no_color: bool = False
    dumb: bool = False
    embed_model: str = EMBED_MODEL
    rerank_model: str = RERANK_MODEL
    ocr: OcrSetting = OcrSetting.AUTO
    gpus: str | None = None
    update_check: bool = False
    state: Path | None = None


def whole_path(written: str, user: Path | None, cwd: Path) -> Path:
    path = Path(written)
    if path.parts[:1] == (USER,):
        if user is None:
            raise InvalidRequestError(NO_HOME, HOME_REMEDY)
        path = user.joinpath(*path.parts[1:])
    return cwd / path


def _named_home(written: str | None, user: Path | None, cwd: Path) -> Path:
    if not written:
        if user is None:
            raise InvalidRequestError(NO_HOME, HOME_REMEDY)
        return user / DEFAULT_HOME
    if not Path(written).is_absolute() and Path(written).parts[:1] != (USER,):
        raise InvalidRequestError(f"{HOME} is no absolute folder: {written}", HOME_REMEDY)
    return whole_path(written, user, cwd)


def _home(written: str | None, user: Path | None, cwd: Path) -> Path:
    home = _named_home(written, user, cwd)
    if home.exists() and not home.is_dir():
        raise InvalidRequestError(f"the home {home} is no folder", HOME_REMEDY)
    return home


def user_folder(home: Callable[[], Path] = Path.home) -> Path | None:
    try:
        return home()
    except RecursionError:
        raise
    except RuntimeError:
        return None


def _ocr(written: str | None) -> OcrSetting:
    chosen = (written or OcrSetting.AUTO.value).strip().lower()
    found = next((each for each in OcrSetting if each.value == chosen), None)
    if found is None:
        raise InvalidRequestError(f"{OCR_VARIABLE} is {written!r}", OCR_REMEDY)
    return found


def _update_check(environment: Mapping[str, str]) -> bool:
    written = environment.get(UPDATE_CHECK_VARIABLE) or "off"
    switched = SWITCHES.get(written.strip().lower())
    if switched is None:
        raise InvalidRequestError(f"{UPDATE_CHECK_VARIABLE} is {written!r}", UPDATE_CHECK_REMEDY)
    return switched and environment.get(OFFLINE_VARIABLE, "").strip().lower() not in OFFLINE


def _state(environment: Mapping[str, str], user: Path | None, platform: str) -> Path | None:
    written = Path(environment.get(STATE_VARIABLES.get(platform, STATE_VARIABLE), ""))
    if written.is_absolute():
        return written / OWN_FOLDER
    return None if user is None else user.joinpath(*STATE_FOLDERS.get(platform, STATE_FOLDER), OWN_FOLDER)


def parsed(environment: Mapping[str, str], user: Path | None, cwd: Path, platform: str = sys.platform) -> Settings:
    return Settings(
        home=_home(environment.get(HOME), user, cwd),
        user=user,
        cwd=cwd,
        no_color=bool(environment.get(NO_COLOR)),
        dumb=environment.get(TERM) == DUMB,
        embed_model=environment.get(EMBED_MODEL_VARIABLE) or EMBED_MODEL,
        rerank_model=environment.get(RERANK_MODEL_VARIABLE) or RERANK_MODEL,
        ocr=_ocr(environment.get(OCR_VARIABLE)),
        gpus=environment.get(GPUS_VARIABLE),
        update_check=_update_check(environment),
        state=_state(environment, user, platform),
    )


def from_environment() -> Settings:
    return parsed(os.environ, user_folder(), Path.cwd())
