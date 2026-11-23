import errno
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

from sift.api.errors import InvalidRequestError, ModelUnavailableError, NotFoundError, SiftError, StoreUnavailableError
from sift.api.fields import FieldValue
from sift.api.search import Unsearched
from sift.interfaces.console.shell import command
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, printable_json

DONE, FAILED, USAGE = 0, 1, 2
ERROR_KINDS: MappingProxyType[type[SiftError], str] = MappingProxyType(
    {
        InvalidRequestError: "invalid_request",
        NotFoundError: "not_found",
        StoreUnavailableError: "store_unavailable",
        ModelUnavailableError: "model_unavailable",
    }
)
EXIT_CODES: MappingProxyType[type[SiftError], int] = MappingProxyType({InvalidRequestError: USAGE})
OS_REMEDIES = MappingProxyType(
    {
        errno.ENOENT: "check that the path exists",
        errno.EACCES: "check that you may read and write the path",
        errno.ENOSPC: "free space on the disk",
    }
)
OS_REMEDY = "check the path and its permissions"
NO_FLAGS: Mapping[str, str] = MappingProxyType({})


class Shown(NamedTuple):
    results: tuple[str, ...]
    messages: tuple[str, ...]
    code: int = DONE


def fix(remedy: str | None, style: Style) -> tuple[str, ...]:
    return (labelled("fix", remedy, Tone.SECONDARY, style),) if remedy else ()


def next_step(words: tuple[str, ...], platform: str, style: Style) -> str:
    return labelled("next", command(words, platform), Tone.SECONDARY, style)


def empty(platform: str, style: Style) -> Shown:
    return Shown((), (labelled("empty", "no collections yet"), next_step(("sift", "add", "FOLDER"), platform, style)))


def value_text(value: FieldValue) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def unsearched_lines(unsearched: tuple[Unsearched, ...], style: Style) -> tuple[str, ...]:
    return tuple(
        line
        for each in unsearched
        for line in (
            labelled("error", f"{each.collection}: {each.message}", Tone.ERROR, style),
            *fix(each.remedy, style),
        )
    )


def _error_json(kind: str, message: str, remedy: str, field: str | None) -> str:
    said = {"error": {"kind": kind, "message": message, "remedy": remedy, "field": field}}
    return printable_json(json.dumps(said, ensure_ascii=False, indent=2))


def failure(
    error: SiftError | OSError, style: Style, *, as_json: bool = False, flags: Mapping[str, str] = NO_FLAGS
) -> Shown:
    if isinstance(error, OSError):
        kind, message, remedy = "os", str(error), OS_REMEDIES.get(error.errno or 0, OS_REMEDY)
        code = FAILED
    else:
        kind, message, remedy = ERROR_KINDS[type(error)], error.message, error.remedy
        code = EXIT_CODES.get(type(error), FAILED)
    field = error.field if isinstance(error, InvalidRequestError) else None
    flag = flags.get(field) if field is not None else None
    said = message if flag is None else f"{flag}: {message}"
    lines = (labelled("error", said, Tone.ERROR, style), *fix(remedy, style))
    return Shown((_error_json(kind, message, remedy, field),) if as_json else (), lines, code)
