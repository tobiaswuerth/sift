import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from sift.api.errors import InvalidRequestError
from sift.api.serving import EMBED_REMOTE_VARIABLE, WorkerAddress, worker_address
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
UPDATE_CHECK_REMEDY = (
    f"set {UPDATE_CHECK_VARIABLE} to on (or 1, yes, true), to be told of a newer sift once a day, "
    "or to off (or 0, no, false)"
)
ON, OFF = frozenset({"1", "on", "yes", "true"}), frozenset({"0", "off", "no", "false"})
SWITCHES = MappingProxyType({**dict.fromkeys(ON, True), **dict.fromkeys(OFF, False)})
OFFLINE_VARIABLE = "HF_HUB_OFFLINE"
OFFLINE = ON
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
OWN_HOME_REMEDY = f"set {HOME} to a folder of sift's own, e.g. ~/{DEFAULT_HOME}"
OWN_PREFIX = "SIFT_"
VARIABLES = MappingProxyType(
    {
        HOME: f"the folder of sift's collections, stores and logs (default ~/{DEFAULT_HOME})",
        EMBED_MODEL_VARIABLE: f"the model that embeds passages (default {EMBED_MODEL})",
        RERANK_MODEL_VARIABLE: f"the model that reranks the passages of a query (default {RERANK_MODEL})",
        EMBED_REMOTE_VARIABLE: "the address a worker prints, to embed and rerank there (default: on this machine)",
        OCR_VARIABLE: "auto, on or off: read the text of scanned pages (default auto: on a GPU)",
        UPDATE_CHECK_VARIABLE: "on or off: be told of a newer sift once a day (default off)",
    }
)


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
    embed_remote: WorkerAddress | None = None


@dataclass(frozen=True, slots=True)
class ActSetting:
    variable: str
    value: str | None


def acting_on(settings: Settings) -> tuple[ActSetting, ...]:
    return (
        ActSetting(EMBED_MODEL_VARIABLE, settings.embed_model),
        ActSetting(RERANK_MODEL_VARIABLE, settings.rerank_model),
        ActSetting(OCR_VARIABLE, settings.ocr.value),
        ActSetting(GPUS_VARIABLE, settings.gpus),
        ActSetting(EMBED_REMOTE_VARIABLE, None if settings.embed_remote is None else settings.embed_remote.address),
    )


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
        raise InvalidRequestError(f"{HOME} is no absolute folder: {written!r}", HOME_REMEDY, HOME)
    home = whole_path(written, user, cwd)
    if home == user:
        raise InvalidRequestError(f"{HOME} is your user folder: {written!r}", OWN_HOME_REMEDY, HOME)
    return home


def _home(written: str | None, user: Path | None, cwd: Path) -> Path:
    home = _named_home(written, user, cwd)
    if home.exists() and not home.is_dir():
        raise InvalidRequestError(f"the home {home} is no folder", HOME_REMEDY, HOME)
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
        raise InvalidRequestError(f"{OCR_VARIABLE} is {written!r}", OCR_REMEDY, OCR_VARIABLE)
    return found


def _update_check(environment: Mapping[str, str]) -> bool:
    written = environment.get(UPDATE_CHECK_VARIABLE) or "off"
    switched = SWITCHES.get(written.strip().lower())
    if switched is None:
        raise InvalidRequestError(f"{UPDATE_CHECK_VARIABLE} is {written!r}", UPDATE_CHECK_REMEDY, UPDATE_CHECK_VARIABLE)
    return switched and environment.get(OFFLINE_VARIABLE, "").strip().lower() not in OFFLINE


def _state(environment: Mapping[str, str], user: Path | None, platform: str) -> Path | None:
    written = Path(environment.get(STATE_VARIABLES.get(platform, STATE_VARIABLE), ""))
    if written.is_absolute():
        return written / OWN_FOLDER
    return None if user is None else user.joinpath(*STATE_FOLDERS.get(platform, STATE_FOLDER), OWN_FOLDER)


def _known_variables(environment: Mapping[str, str]) -> None:
    unknown = sorted(name for name in environment if name.startswith(OWN_PREFIX) and name not in VARIABLES)
    if unknown:
        named, them = ", ".join(unknown), unknown[0] if len(unknown) == 1 else "them"
        said = f"sift has no setting{'s' * (len(unknown) > 1)} {named}"
        raise InvalidRequestError(said, f"unset {them}, or set one of {', '.join(VARIABLES)}", unknown[0])


def _model(environment: Mapping[str, str], variable: str, default: str) -> str:
    return (environment.get(variable) or "").strip() or default


def _remote(written: str | None) -> WorkerAddress | None:
    address = (written or "").strip()
    return worker_address(address) if address else None


def parsed(environment: Mapping[str, str], user: Path | None, cwd: Path, platform: str = sys.platform) -> Settings:
    _known_variables(environment)
    return Settings(
        home=_home(environment.get(HOME), user, cwd),
        user=user,
        cwd=cwd,
        no_color=bool(environment.get(NO_COLOR)),
        dumb=environment.get(TERM) == DUMB,
        embed_model=_model(environment, EMBED_MODEL_VARIABLE, EMBED_MODEL),
        rerank_model=_model(environment, RERANK_MODEL_VARIABLE, RERANK_MODEL),
        ocr=_ocr(environment.get(OCR_VARIABLE)),
        gpus=environment.get(GPUS_VARIABLE),
        update_check=_update_check(environment),
        state=_state(environment, user, platform),
        embed_remote=_remote(environment.get(EMBED_REMOTE_VARIABLE)),
    )


def from_environment() -> Settings:
    return parsed(os.environ, user_folder(), Path.cwd())
