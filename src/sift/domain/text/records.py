import re
from collections.abc import Mapping
from dataclasses import dataclass

from sift.api.fields import FieldValue
from sift.domain.text.chunks import chunks
from sift.domain.text.json_values import NestedTooDeeplyError, composed_fields, json_value
from sift.domain.text.markdown import heading, one_line
from sift.domain.text.notes import Document, digest, storable

RECORD_FILES = (".jsonl", ".ndjson")
RECORD_MARK = "#"
ID_FIELDS = ("id", "_id")
TITLE_FIELDS = ("title", "name", "headline", "subject")
TEXT_FIELDS = ("text", "content", "body", "abstract", "description", "summary")
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
NOT_A_KEY = "its id is no text or whole number"


@dataclass(frozen=True, slots=True)
class Chosen:
    id: str | None = None
    title: str | None = None
    text: tuple[str, ...] = ()


NO_CHOICE = Chosen()


@dataclass(frozen=True, slots=True)
class SkippedLine:
    line: int
    why: str


def as_text(value: FieldValue) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(each, str) for each in value):
        return "\n\n".join(each.strip() for each in value if isinstance(each, str) and each.strip())
    return ""


def _first(found: Mapping[str, FieldValue], names: tuple[str, ...]) -> str | None:
    return next((name for name in names if as_text(found.get(name))), None)


def _text_fields(found: Mapping[str, FieldValue], chosen: Chosen, taken: tuple[str | None, ...]) -> tuple[str, ...]:
    if chosen.text:
        return chosen.text
    known = _first(found, TEXT_FIELDS)
    if known is not None:
        return (known,)
    strings = [(len(value), name) for name, value in found.items() if name not in taken and isinstance(value, str)]
    return (max(strings)[1],) if strings else ()


def _key(written: FieldValue, line: int) -> str | None:
    match written:
        case None | "":
            return str(line)
        case bool():
            return None
        case int():
            return str(written)
        case str() if not CONTROL.search(written):
            return written
        case _:
            return None


def record(found: Mapping[str, FieldValue], line: int, chosen: Chosen) -> Document | SkippedLine:
    id_field = chosen.id or next((name for name in ID_FIELDS if name in found), None)
    named = _key(found.get(id_field) if id_field else None, line)
    if named is None:
        return SkippedLine(line, NOT_A_KEY)
    title_field = chosen.title or _first(found, TITLE_FIELDS)
    text_fields = _text_fields(found, chosen, (id_field, title_field))
    body = "\n\n".join(text for name in text_fields if (text := as_text(found.get(name))))
    if not body:
        return SkippedLine(line, "no text")
    title = one_line(as_text(found.get(title_field))) if title_field else ""
    text = storable(f"# {title}\n\n{body}" if title else body)
    fields = composed_fields({name: value for name, value in found.items() if name not in text_fields})
    return Document(storable(named), "", storable(title or heading(body) or named), fields, text, chunks(text), ())


def record_of(line: str, number: int, chosen: Chosen) -> Document | SkippedLine:
    try:
        value = json_value(line)
    except NestedTooDeeplyError:
        return SkippedLine(number, "nested too deeply")
    except ValueError:
        return SkippedLine(number, "not JSON")
    if not isinstance(value, Mapping):
        return SkippedLine(number, "not an object")
    found = record(value, number, chosen)
    if isinstance(found, SkippedLine):
        return found
    return Document(found.key, digest(line.encode()), found.title, found.fields, found.text, found.chunks, ())


def record_path(path: str, key: str) -> str:
    return f"{path}{RECORD_MARK}{key}" if key else path


def public_id(collection: str, path: str, key: str) -> str:
    return f"{collection}/{record_path(path, key)}"


def split_record_path(relative: str) -> tuple[str, str]:
    at = relative.find(RECORD_MARK)
    while at >= 0:
        if relative[:at].lower().endswith(RECORD_FILES):
            return relative[:at], relative[at + 1 :]
        at = relative.find(RECORD_MARK, at + 1)
    return relative, ""
