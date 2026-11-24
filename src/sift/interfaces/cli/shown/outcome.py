import errno
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

from sift.api.errors import InvalidRequestError, SiftError
from sift.api.fields import FieldValue
from sift.api.search import Left, Unsearched
from sift.api.wire import Json, encoded_error
from sift.interfaces.console.shell import command
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, printable_json

DONE, FAILED, USAGE = 0, 1, 2
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


def left_out_lines(left_out: tuple[Left, ...], style: Style) -> tuple[str, ...]:
    return tuple(
        line
        for each in left_out
        for line in (
            labelled("note", f"{each.collection}: {each.message}", Tone.WARNING, style),
            *fix(each.remedy, style),
        )
    )


def _error_json(said: Json) -> str:
    return printable_json(json.dumps({"error": said}, ensure_ascii=False, indent=2))


def failure(
    error: SiftError | OSError, style: Style, *, as_json: bool = False, flags: Mapping[str, str] = NO_FLAGS
) -> Shown:
    field = error.field if isinstance(error, InvalidRequestError) else None
    if isinstance(error, OSError):
        message, remedy, code = str(error), OS_REMEDIES.get(error.errno or 0, OS_REMEDY), FAILED
        written: Json = {"kind": "os", "code": "os", "message": message, "remedy": remedy, "field": None}
    else:
        message, remedy, code = error.message, error.remedy, EXIT_CODES.get(type(error), FAILED)
        written = encoded_error(error)
    flag = flags.get(field) if field is not None else None
    said = message if flag is None else f"{flag}: {message}"
    lines = (labelled("error", said, Tone.ERROR, style), *fix(remedy, style))
    return Shown((_error_json(written),) if as_json else (), lines, code)
