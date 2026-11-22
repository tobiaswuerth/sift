import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

HOME = "SIFT_HOME"
NO_COLOR = "NO_COLOR"
TERM = "TERM"
DUMB = "dumb"
DEFAULT_HOME = ".sift"
USER = "~"


@dataclass(frozen=True, slots=True)
class Settings:
    home: Path
    user: Path
    cwd: Path
    no_color: bool = False
    dumb: bool = False


def whole_path(written: str, user: Path, cwd: Path) -> Path:
    path = Path(written)
    if path.parts[:1] == (USER,):
        path = user.joinpath(*path.parts[1:])
    return cwd / path


def parsed(environment: Mapping[str, str], user: Path, cwd: Path) -> Settings:
    home = environment.get(HOME) or str(user / DEFAULT_HOME)
    return Settings(
        home=whole_path(home, user, cwd),
        user=user,
        cwd=cwd,
        no_color=bool(environment.get(NO_COLOR)),
        dumb=environment.get(TERM) == DUMB,
    )


def from_environment() -> Settings:
    return parsed(os.environ, Path.home(), Path.cwd())
