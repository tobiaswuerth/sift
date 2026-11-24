import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

from sift.api.errors import InvalidRequestError, JobStoppedError, SiftError, system_failure
from sift.api.fields import FieldValue
from sift.api.search import Left, Unsearched
from sift.api.wire import Json, encoded_error
from sift.domain.text.wording import counted
from sift.interfaces.console.progress import View
from sift.interfaces.console.shell import command
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, printable_json

DONE, FAILED, USAGE, INTERRUPTED = 0, 1, 2, 130
EXIT_CODES: MappingProxyType[type[SiftError], int] = MappingProxyType(
    {InvalidRequestError: USAGE, JobStoppedError: INTERRUPTED}
)
NO_FLAGS: Mapping[str, str] = MappingProxyType({})
STOPPED_WRITING = "stopped: what was written so far is kept"
STOPPED_KEEPING = "stopped: {files} of {name} are kept; the others are read again"


class Shown(NamedTuple):
    results: tuple[str, ...]
    messages: tuple[str, ...]
    code: int = DONE


def fix(remedy: str | None, style: Style) -> tuple[str, ...]:
    return (labelled("fix", remedy, Tone.SECONDARY, style),) if remedy else ()


def next_step(words: tuple[str, ...], platform: str, style: Style) -> str:
    return labelled("next", command(words, platform), Tone.SECONDARY, style)


def stopped_writing(view: View, platform: str, style: Style) -> tuple[str, ...]:
    if view.kept is None:
        return (STOPPED_WRITING,)
    said = STOPPED_KEEPING.format(files=counted(view.kept, "file"), name=view.name)
    return said, next_step(("sift", "index", view.name), platform, style)


def empty(platform: str, style: Style) -> Shown:
    return Shown((), (labelled("empty", "no collections yet"), next_step(("sift", "add", "FOLDER"), platform, style)))


def value_text(value: FieldValue) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


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
    return printable_json(json.dumps({"error": said}, ensure_ascii=False, indent=2, allow_nan=False))


def failure(
    error: SiftError | OSError, style: Style, *, as_json: bool = False, flags: Mapping[str, str] = NO_FLAGS
) -> Shown:
    known = system_failure(error) if isinstance(error, OSError) else error
    field = known.field if isinstance(known, InvalidRequestError) else None
    flag = flags.get(field) if field is not None else None
    said = known.message if flag is None else f"{flag}: {known.message}"
    lines = (labelled("error", said, Tone.ERROR, style), *fix(known.remedy, style))
    code = EXIT_CODES.get(type(known), FAILED)
    return Shown((_error_json(encoded_error(known)),) if as_json else (), lines, code)
