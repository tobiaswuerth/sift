import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from sift.api.errors import InvalidRequestError
from sift.inference.hub import EMBED_MODEL, RERANK_MODEL

HOME = "SIFT_HOME"
EMBED_MODEL_VARIABLE = "SIFT_EMBED_MODEL"
RERANK_MODEL_VARIABLE = "SIFT_RERANK_MODEL"
NO_COLOR = "NO_COLOR"
TERM = "TERM"
DUMB = "dumb"
DEFAULT_HOME = ".sift"
USER = "~"
NO_HOME = "no home folder is set"
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


def whole_path(written: str, user: Path | None, cwd: Path) -> Path:
    path = Path(written)
    if path.parts[:1] == (USER,):
        if user is None:
            raise InvalidRequestError(NO_HOME, HOME_REMEDY)
        path = user.joinpath(*path.parts[1:])
    return cwd / path


def _home(written: str | None, user: Path | None, cwd: Path) -> Path:
    if not written:
        if user is None:
            raise InvalidRequestError(NO_HOME, HOME_REMEDY)
        return user / DEFAULT_HOME
    if not Path(written).is_absolute() and Path(written).parts[:1] != (USER,):
        raise InvalidRequestError(f"{HOME} is no absolute folder: {written}", HOME_REMEDY)
    return whole_path(written, user, cwd)


def user_folder(home: Callable[[], Path] = Path.home) -> Path | None:
    try:
        return home()
    except RuntimeError:
        return None


def parsed(environment: Mapping[str, str], user: Path | None, cwd: Path) -> Settings:
    return Settings(
        home=_home(environment.get(HOME), user, cwd),
        user=user,
        cwd=cwd,
        no_color=bool(environment.get(NO_COLOR)),
        dumb=environment.get(TERM) == DUMB,
        embed_model=environment.get(EMBED_MODEL_VARIABLE) or EMBED_MODEL,
        rerank_model=environment.get(RERANK_MODEL_VARIABLE) or RERANK_MODEL,
    )


def from_environment() -> Settings:
    return parsed(os.environ, user_folder(), Path.cwd())
