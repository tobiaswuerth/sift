import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from sift.api.fields import Fields, FieldValue
from sift.domain.text.json_values import MOST_DEPTH, deeper_than, depth, inside, json_value

BLOCK = re.compile(r"\A---[ \t]*\r?\n(?:(.*?)\r?\n)??---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
LIST_ITEM = re.compile(r"[ \t]*-[ \t]+(.*)")
AT_THE_MARGIN = frozenset({"-", "]", "}", "],", "},"})
BLOCK_SCALAR = re.compile(r"([|>])[+-]?[1-9]?[+-]?(?:[ \t]+#.*)?")
WIKILINK = re.compile(r"\[\[[^\[\]\n]*\]\]")
CLOSERS = frozenset({"", "]", "}"})
MOST_FLOW_NESTING = 32


@dataclass(frozen=True, slots=True)
class FrontMatter:
    fields: Fields
    inner: str
    inner_start: int
    end: int
    lost: bool = False


@dataclass(frozen=True, slots=True)
class Brackets:
    depth: int = 0
    quote: str | None = None
    at_item_start: bool = True


class Kind(Enum):
    PLAIN = auto()
    LIST = auto()
    SCALAR = auto()
    FOLDED = auto()
    FLOW = auto()


@dataclass(frozen=True, slots=True)
class Entry:
    key: str
    value: str
    kind: Kind


def read_on(text: str, state: Brackets) -> Brackets:
    level, quote, at_item_start = state.depth, state.quote, state.at_item_start
    for character in text:
        if quote:
            quote = None if character == quote else quote
        elif character in "\"'" and at_item_start:
            quote = character
        elif character in "[{":
            level += 1
        elif character in "]}":
            level -= 1
        at_item_start = character in "[{," or (at_item_start and character.isspace())
    return Brackets(level, quote, at_item_start)


def flow_items(text: str) -> list[str] | None:
    items: list[str] = []
    level, quote, start = 0, None, 1
    for index, character in enumerate(text):
        if quote:
            quote = None if character == quote else quote
        elif character in "\"'" and not text[start:index].strip():
            quote = character
        elif character == "[":
            level += 1
        elif character == "]":
            level -= 1
            if not level and index < len(text) - 1:
                return None
        elif character == "," and level == 1:
            items.append(text[start:index])
            start = index + 1
    if level or quote:
        return None
    return [*items, text[start:-1]]


def value(text: str) -> FieldValue:
    try:
        return json_value(text)
    except ValueError:
        listed = None if deeper_than(text, MOST_FLOW_NESTING) else flow_list(text)
        return text.strip("'\"") if listed is None else listed


def flow_list(text: str) -> list[FieldValue] | None:
    if not (text.startswith("[") and text.endswith("]")) or WIKILINK.fullmatch(text):
        return None
    items = flow_items(text)
    return None if items is None else [value(item.strip()) for item in items if item.strip()]


def _kind(written: str) -> Kind:
    if not written:
        return Kind.LIST
    if mark := BLOCK_SCALAR.fullmatch(written):
        return Kind.FOLDED if mark.group(1) == ">" else Kind.SCALAR
    if written[:1] in "[{" and written[-1:] in "[{," and read_on(written, Brackets()).depth > 0:
        return Kind.FLOW
    return Kind.PLAIN


def _key_line(line: str) -> Entry | None:
    key, colon, written = line.partition(":")
    if not colon or not key.strip() or key[0].isspace() or key[0] == "#":
        return None
    written = written.strip()
    return Entry(key.strip(), written, _kind(written))


def _indented_or_blank(line: str) -> bool:
    return line[:1].isspace() or not line.strip()


def _continues_flow(line: str) -> bool:
    return line[:1].isspace() or line.strip().rstrip(",") in CLOSERS


def _is_item(line: str) -> bool:
    return LIST_ITEM.match(line) is not None


def _takes_none(_line: str) -> bool:
    return False


TAKES: Mapping[Kind, Callable[[str], bool]] = MappingProxyType(
    {
        Kind.PLAIN: _takes_none,
        Kind.LIST: _is_item,
        Kind.SCALAR: _indented_or_blank,
        Kind.FOLDED: _indented_or_blank,
        Kind.FLOW: _continues_flow,
    }
)


@dataclass(frozen=True, slots=True)
class Entries:
    found: dict[str, tuple[Entry, list[str]]]
    lost: bool


def _lost_below_a_list(current: tuple[Entry, list[str]] | None, line: str) -> bool:
    return current is not None and current[0].kind is Kind.LIST and _key_line(line.lstrip()) is not None


def entries(lines: Iterable[str]) -> Entries:
    found: dict[str, tuple[Entry, list[str]]] = {}
    current: tuple[Entry, list[str]] | None = None
    brackets, lost = Brackets(), False
    for line in lines:
        if current is not None and TAKES[current[0].kind](line):
            current[1].append(line)
            if current[0].kind is Kind.FLOW and (brackets := read_on(line.strip(), brackets)).depth <= 0:
                current = None
            continue
        lost = lost or (line[:1].isspace() and _lost_below_a_list(current, line))
        if current is not None and current[0].kind is not Kind.LIST:
            current = None
        if started := _key_line(line):
            current = found[started.key] = (started, [])
            brackets = read_on(started.value, Brackets()) if started.kind is Kind.FLOW else Brackets()
    return Entries(found, lost)


def block_scalar(lines: Sequence[str], *, folded: bool) -> str:
    kept = [line.rstrip() for line in lines]
    indent = min((len(line) - len(line.lstrip()) for line in kept if line), default=0)
    text = "\n".join(line[indent:] for line in kept).strip("\n")
    if not folded:
        return text
    return "\n".join(" ".join(paragraph.split()) for paragraph in re.split(r"\n{2,}", text))


type ParsedPair = tuple[FieldValue, FieldValue]


def _plain(entry: Entry, _below: Sequence[str]) -> ParsedPair:
    return value(entry.value), entry.value


def _items(_entry: Entry, below: Sequence[str]) -> ParsedPair:
    written = [found.group(1).strip() for line in below if (found := LIST_ITEM.match(line))]
    return ([value(text) for text in written] if written else None), written


def _scalar(_entry: Entry, below: Sequence[str]) -> ParsedPair:
    text = block_scalar(below, folded=False)
    return text, text


def _folded(_entry: Entry, below: Sequence[str]) -> ParsedPair:
    text = block_scalar(below, folded=True)
    return text, text


def _flow(entry: Entry, below: Sequence[str]) -> ParsedPair:
    joined = " ".join(filter(None, [entry.value, *(line.strip() for line in below)]))
    text = joined if read_on(joined, Brackets()).depth <= 0 else entry.value
    return value(text), text


PARSE: Mapping[Kind, Callable[[Entry, Sequence[str]], ParsedPair]] = MappingProxyType(
    {Kind.PLAIN: _plain, Kind.LIST: _items, Kind.SCALAR: _scalar, Kind.FOLDED: _folded, Kind.FLOW: _flow}
)


def _shallow_enough(found: FieldValue, written: FieldValue) -> FieldValue:
    if depth(found) <= MOST_DEPTH:
        return found
    pairs = zip(inside(found) or (), inside(written) or (), strict=True)
    return [item if depth(item) < MOST_DEPTH else text for item, text in pairs]


def _fields_of(read: Entries, block: str) -> dict[str, FieldValue]:
    parsed = {key: PARSE[entry.kind](entry, below) for key, (entry, below) in read.found.items()}
    if block.count("[") + block.count("{") < MOST_DEPTH:
        return {key: found for key, (found, _) in parsed.items()}
    return {key: _shallow_enough(found, written) for key, (found, written) in parsed.items()}


def _in_mapping(line: str) -> bool:
    key, colon, _ = line.partition(":")
    if not line.strip() or line[0].isspace() or line[0] == "#" or line.rstrip() in AT_THE_MARGIN:
        return True
    return bool(LIST_ITEM.match(line) or (colon and key))


def front_matter(text: str) -> FrontMatter | None:
    found = BLOCK.match(text)
    if found is None or not all(map(_in_mapping, (found.group(1) or "").split("\n"))):
        return None
    inner = found.group(1) or ""
    start = found.start(1) if found.group(1) is not None else found.end()
    read = entries(inner.split("\n"))
    return FrontMatter(_fields_of(read, inner), inner, start, found.end(), read.lost)
