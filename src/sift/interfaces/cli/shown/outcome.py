import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

from sift.api.errors import InvalidRequestError, JobStoppedError, Problem, SiftError, system_failure
from sift.api.fields import FieldValue
from sift.api.notices import Notice
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
FIELD_HEAD = re.compile(r"[.\[]")
NOTICE_TONES: Mapping[str, Tone] = MappingProxyType({"error": Tone.ERROR, "note": Tone.WARNING})
NEXT_STEPS: Mapping[Problem, tuple[str, ...]] = MappingProxyType(
    {
        Problem.NOT_EMBEDDED: ("sift", "embed", "NAME"),
        Problem.OTHER_MODEL: ("sift", "embed", "--reset"),
        Problem.KEYWORDS_REMADE: ("sift", "index", "NAME"),
        Problem.INACTIVE_COLLECTION: ("sift", "activate", "NAME"),
        Problem.STORE_MISSING: ("sift", "rm", "NAME"),
        Problem.STORE_DAMAGED: ("sift", "rm", "NAME"),
    }
)
STOPPED_WRITING = "stopped: what was written so far is kept"
STOPPED_KEEPING = "stopped: {files} of {name} are kept; the others are read again"


class Shown(NamedTuple):
    results: tuple[str, ...]
    messages: tuple[str, ...]
    code: int = DONE


def command_next(code: Problem, style: Style, name: str = "NAME") -> tuple[str, ...]:
    words = NEXT_STEPS.get(code)
    said = None if words is None else " ".join(name if each == "NAME" else each for each in words)
    return () if said is None else (labelled("next", said, Tone.SECONDARY, style),)


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


def notice_lines(notices: tuple[Notice, ...], style: Style, label: str = "error") -> tuple[str, ...]:
    return tuple(
        line
        for each in notices
        for line in (
            labelled(label, f"{each.collection}: {each.message}", NOTICE_TONES[label], style),
            *fix(each.remedy, style),
            *command_next(each.code, style, each.collection or "NAME"),
        )
    )


def _error_json(said: Json) -> str:
    return printable_json(json.dumps({"error": said}, ensure_ascii=False, indent=2, allow_nan=False))


def failure(
    error: SiftError | OSError, style: Style, *, as_json: bool = False, flags: Mapping[str, str] = NO_FLAGS
) -> Shown:
    known = system_failure(error) if isinstance(error, OSError) else error
    flag = flags.get(FIELD_HEAD.split(known.field, 1)[0]) if known.field is not None else None
    said = known.message if flag is None else f"{flag}: {known.message}"
    lines = (labelled("error", said, Tone.ERROR, style), *fix(known.remedy, style), *command_next(known.code, style))
    code = EXIT_CODES.get(type(known), FAILED)
    return Shown((_error_json(encoded_error(known)),) if as_json else (), lines, code)
