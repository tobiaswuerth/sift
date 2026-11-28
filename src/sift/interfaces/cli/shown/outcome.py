import json
import re
import traceback
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import NamedTuple

from sift.api.errors import (
    InternalError,
    InvalidRequestError,
    JobStoppedError,
    Problem,
    SiftError,
    system_failure,
)
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


class Naming(Enum):
    IF_KNOWN = "if known"
    NEEDED = "needed"
    NEVER = "never"


class NextStep(NamedTuple):
    words: tuple[str, ...]
    naming: Naming


NEXT_STEPS: Mapping[Problem, NextStep] = MappingProxyType(
    {
        Problem.NOT_EMBEDDED: NextStep(("sift", "embed"), Naming.IF_KNOWN),
        Problem.OTHER_MODEL: NextStep(("sift", "embed", "--reset"), Naming.NEVER),
        Problem.KEYWORDS_REMADE: NextStep(("sift", "index"), Naming.IF_KNOWN),
        Problem.INACTIVE_COLLECTION: NextStep(("sift", "activate"), Naming.NEEDED),
        Problem.STORE_MISSING: NextStep(("sift", "rm"), Naming.NEEDED),
        Problem.STORE_DAMAGED: NextStep(("sift", "rm"), Naming.NEEDED),
    }
)
STOPPED_WRITING = "stopped: what was written so far is kept"
STOPPED_KEEPING = "stopped: {files} of {name} are kept; the others are read again"
UNFORESEEN_REMEDY = "report it, with the lines above it"


class Shown(NamedTuple):
    results: tuple[str, ...]
    messages: tuple[str, ...]
    code: int = DONE


def command_next(code: Problem, style: Style, name: str | None = None) -> tuple[str, ...]:
    step = NEXT_STEPS.get(code)
    if step is None or (name is None and step.naming is Naming.NEEDED):
        return ()
    said = " ".join(step.words if name is None or step.naming is Naming.NEVER else (*step.words, name))
    return (labelled("next", said, Tone.SECONDARY, style),)


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


def _placed(notice: Notice) -> str:
    where = notice.collection or notice.id
    said = notice.message
    return said if where is None or said.startswith(where) or repr(where) in said else f"{where}: {said}"


def notice_lines(notices: tuple[Notice, ...], style: Style, label: str = "error") -> tuple[str, ...]:
    return tuple(
        line
        for each in notices
        for line in (
            labelled(label, _placed(each), NOTICE_TONES[label], style),
            *fix(each.remedy, style),
            *command_next(each.code, style, each.collection),
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


def unforeseen(failed: InternalError, style: Style, *, as_json: bool) -> Shown:
    traced = "".join(traceback.format_exception(failed.__cause__ or failed)).rstrip("\n")
    shown = failure(failed, style, as_json=as_json)
    return shown._replace(messages=(traced, *shown.messages))
