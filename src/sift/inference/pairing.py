import importlib.metadata
import re
from collections.abc import Callable, Iterable, Mapping
from typing import NamedTuple

BUILT_FOR = re.compile(r"triton *(?:~=|==) *([0-9]+)\.([0-9]+)")
MINOR = re.compile(r"([0-9]+)\.([0-9]+)")
UNPAIRED = (
    "PyTorch {torch} is built for Triton {wanted}, but {name} {found} is installed: with uv, "
    'uv tool install --reinstall --torch-backend auto --with "{name}~={wanted}.0" sift-rag; with pip, '
    'pip install "{name}~={wanted}.0"'
)


class Release(NamedTuple):
    name: str
    version: str


def unpaired(torch: Release, requires: Iterable[str], triton: Release | None) -> tuple[str, ...]:
    built = next((found for found in map(BUILT_FOR.match, requires) if found is not None), None)
    if built is None or triton is None:
        return ()
    held = MINOR.match(triton.version)
    if held is None or held.groups() == built.groups():
        return ()
    wanted = ".".join(built.groups())
    return (UNPAIRED.format(torch=torch.version, wanted=wanted, name=triton.name, found=triton.version),)


def read_unpaired(
    providers: Callable[[], Mapping[str, list[str]]] = importlib.metadata.packages_distributions,
    version: Callable[[str], str] = importlib.metadata.version,
    requires: Callable[[str], list[str] | None] = importlib.metadata.requires,
) -> tuple[str, ...]:
    names = providers().get("triton", [])
    try:
        triton = Release(names[0], version(names[0])) if names else None
        return unpaired(Release("torch", version("torch")), requires("torch") or [], triton)
    except importlib.metadata.PackageNotFoundError:
        return ()
