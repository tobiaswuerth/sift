"""Reading files into documents, one reader per kind of file (READERS, by extension): Markdown and text files are
one document each, with YAML front matter as metadata; JSON Lines files one per line; PDF, office files, e-books and
web pages one each, converted to Markdown by Xberg; saved emails one each, and one per attachment sift reads. Every
document's text is cut into chunks with their line numbers.

Readers run in worker processes while indexing, so they read from bytes and return plain data (`Doc`).
"""

import asyncio
import hashlib
import io
import json
import math
import os
import posixpath
import re
import unicodedata
from collections.abc import Callable, Coroutine, Iterable, Iterator
from datetime import UTC, datetime
from email.message import MIMEPart
from functools import partial
from html.parser import HTMLParser
from pathlib import Path
from typing import IO, Any, NamedTuple, TypeVar
from urllib.parse import unquote

from . import console

TEXT = (".md", ".markdown", ".txt")
JSONL = (".jsonl", ".ndjson")
DOCUMENTS = (".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".rtf", ".epub", ".html", ".htm")  # by Xberg
EMAIL = (".eml",)
CHUNK_CHARS, OVERLAP_CHARS = 3000, 300  # ~750 tokens, well inside the models' 1024-token window


FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(?:(.*?)\r?\n)??---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
LIST_ITEM = re.compile(r"[ \t]*-[ \t]+(.*)")
HEADING = re.compile(r" {0,3}#{1,6}[ \t]")  # how a Markdown (ATX) heading starts
FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")  # how a fenced code block starts and ends
LINE = re.compile(r"[^\n]*\n|[^\n]+")  # only \n ends a line, as in editors; str.splitlines knows eight more
WIKILINK_VALUE = re.compile(r"\[\[[^\[\]\n]*\]\]")
TITLE_FROM = ("heading", "name")  # where a Markdown or text file's title comes from, after its `title:` field


SURROGATE = re.compile(r"\\u[dD][89a-fA-F]")  # JSON's escape of a surrogate, "\ud83d", half an emoji: alone, a string
LONE = re.compile("[\ud800-\udfff]")  # holding it can't be stored (SQLite takes UTF-8 only)


JSON_DEPTH = 999  # how deeply a value may nest: SQLite stores JSON nested 1,000 deep at most, the document holding it
# one level of that


class NestedTooDeeply(RecursionError):
    """JSON nested deeper than SQLite stores it (JSON_DEPTH)."""


Json = None | bool | int | float | str | list[Any] | dict[str, Any]  # a JSON value, nested ones as Any


def _json(text: str) -> Any:
    """JSON as sift keeps it: NaN, Infinity and numbers too large for a float (1e999) as written, which SQLite's JSON
    rejects, and a surrogate alone as U+FFFD (a pair Python joins itself). Raises ValueError, and RecursionError for
    JSON nested deeper than Python recurses or SQLite stores."""
    value = json.loads(text, parse_constant=str, parse_float=_float)
    if text.count("[") + text.count("{") > JSON_DEPTH and _depth(value) > JSON_DEPTH:
        raise NestedTooDeeply(f"nested more than {JSON_DEPTH} deep")
    return _whole(value) if SURROGATE.search(text) else value


def _float(text: str) -> float | str:
    """A JSON number with a fraction or an exponent, as a float; as written if a float can't hold it."""
    number = float(text)
    return number if math.isfinite(number) else text


def _depth(value: Json) -> int:
    """How deeply a JSON value nests lists and objects: 0 for none."""
    deepest, todo = 0, [(value, 0)]
    while todo:
        item, depth = todo.pop()
        if isinstance(item, list | dict):
            deepest = max(deepest, depth + 1)
            todo += [(inner, depth + 1) for inner in (item.values() if isinstance(item, dict) else item)]
    return deepest


def _whole(value: Json) -> Any:
    """A JSON value with every surrogate alone in its strings as U+FFFD."""
    if isinstance(value, str):
        return LONE.sub("\ufffd", value)
    if isinstance(value, list):
        return [_whole(item) for item in value]
    if isinstance(value, dict):
        return {_whole(key): _whole(item) for key, item in value.items()}
    return value


def _value(text: str) -> Any:
    """A front-matter value: JSON if it parses (lists, numbers, quoted strings), else a YAML flow list of unquoted
    items, else plain text; as written, if nested deeper than Python recurses."""
    try:
        try:
            return _json(text)
        except ValueError:
            items = _flow_list(text)
            return text.strip("'\"") if items is None else items
    except RecursionError:
        return text


def _flow_list(text: str) -> list[Any] | None:
    """The items of a YAML flow list such as `[Hans Mueller, "H. Müller"]`, or None if `text` isn't one list. A single
    wikilink, `[[Acme AG]]`, is a link, as Obsidian users mean it, though YAML reads a list in a list."""
    if not (text.startswith("[") and text.endswith("]")) or WIKILINK_VALUE.fullmatch(text):
        return None
    items: list[str] = []
    depth, quote, start = 0, None, 1
    for i, c in enumerate(text):
        if quote:
            quote = None if c == quote else quote
        elif c in "\"'" and not text[start:i].strip():  # only at an item's start: O'Brien is no quote
            quote = c
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if not depth and i < len(text) - 1:  # closed before the end: `[[A]], [[B]]`, `[Draft] My note`
                return None
        elif c == "," and depth == 1:
            items.append(text[start:i])
            start = i + 1
    if depth or quote:
        return None
    items.append(text[start:-1])
    return [_value(item.strip()) for item in items if item.strip()]


def _front_matter(block: str) -> dict[str, Any]:
    """Flat `key: value` lines, and lists as `- item` lines below a key without a value, as Obsidian writes them."""
    meta: dict[str, Any] = {}
    listing: str | None = None
    written: dict[str, Any] = {}  # each value's text, or its `- item` lines' texts
    for line in block.split("\n"):  # only \n ends a line (LINE): JSON writes U+2028 in a string as it is
        if listing and (item := LIST_ITEM.match(line)):
            text = item.group(1).strip()
            meta[listing] = [*(meta[listing] or []), _value(text)]
            written[listing].append(text)
            continue
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key[0].isspace() and key[0] != "#":  # not nested, not a comment
            key, value = key.strip(), value.strip()
            meta[key] = _value(value) if value else None
            listing = None if value else key
            written[key] = value or []
    if block.count("[") + block.count("{") >= JSON_DEPTH:  # else nothing in it nests too deeply
        for key, value in meta.items():
            if _depth(value) > JSON_DEPTH:  # in a list, or in the front matter, one level deeper than alone: as written
                text = written[key]
                meta[key] = (
                    text if isinstance(text, str) else [v if _depth(v) < JSON_DEPTH else t for v, t in zip(value, text)]
                )
    return meta


Chunk = tuple[int, int, int, int]  # first line, last line, position, length


def _outside_code(lines: Iterable[str]) -> list[bool]:
    """For each of these lines of Markdown, in order, whether it lies outside fenced code blocks; a fence's own lines
    don't."""
    outside: list[bool] = []
    fence: str | None = None
    for line in lines:
        if m := FENCE.match(line):
            mark = m.group(1)
            if not fence:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence) and not line[m.end() :].strip():
                fence = None
            outside.append(False)
        else:
            outside.append(not fence)
    return outside


def _prose(text: str, first_line: int = 1) -> Iterator[tuple[int, str]]:
    """The lines of Markdown `text` outside fenced code blocks, with their numbers."""
    lines = text.split("\n")
    return ((n, line) for n, (line, outside) in enumerate(zip(lines, _outside_code(lines)), first_line) if outside)


def _heading(text: str) -> str | None:
    """The first Markdown heading outside fenced code blocks, or None."""
    for _, line in _prose(text):
        if HEADING.match(line):
            title = line.strip().lstrip("#").strip()
            if (bare := title.rstrip("#")) != title and (not bare or bare[-1] in " \t"):
                title = bare.strip()  # a closing sequence: "## Title ##", but not "# C#"
            if title:
                return title
    return None


def _front_matter_block(text: str) -> re.Match[str] | None:
    """The front matter a note starts with: lines between two `---` lines, each of which a mapping could hold, a `key:
    value`, a `- item`, an indented continuation, a comment or a blank. None for any other, such as a paragraph between
    two thematic breaks, which is text to search."""
    m = FRONT_MATTER.match(text)
    return m if m and all(map(_in_mapping, (m.group(1) or "").split("\n"))) else None


def _in_mapping(line: str) -> bool:
    key, colon, _ = line.partition(":")
    return not line.strip() or line[0].isspace() or line[0] == "#" or bool(LIST_ITEM.match(line) or (colon and key))


def parse(text: str, fallback_title: str, title_from: str = "heading") -> tuple[str, dict[str, Any], str, int]:
    """Split off YAML front matter. Returns title, meta, body and the body's first line number. The title is the
    `title` field, else with title_from "heading" the first heading, else `fallback_title`, the file name."""
    meta: dict[str, Any] = {}
    first_line = 1
    m = _front_matter_block(text)
    if m:
        meta = _front_matter(m.group(1) or "")
        first_line += text.count("\n", 0, m.end())
        text = text[m.end() :]
    title = meta.get("title")
    title = str(title).strip() if isinstance(title, str | int | float) and not isinstance(title, bool) else None
    return title or (title_from == "heading" and _heading(text)) or fallback_title, meta, text, first_line


def chunk(text: str, first_line: int = 1) -> Iterator[Chunk]:
    """Split text into pieces of at most CHUNK_CHARS on line boundaries, preferring headings, with a small overlap.

    Yields (first line, last line, position, length): each piece is text[position:position + length], stripped. A
    heading is a line that starts as one in Markdown (HEADING), outside fenced code: not any line starting with "#",
    as a comment in code does, an include, or a tag.
    """
    lines: list[tuple[int, int, str, bool]] = []  # (line number, position, line, whether it is a heading)
    pos, whole = 0, LINE.findall(text)
    for n, (line, prose) in enumerate(zip(whole, _outside_code(whole)), first_line):
        start, heading = 0, prose and HEADING.match(line) is not None
        while len(line) - start > CHUNK_CHARS:  # break overlong lines at whitespace, the rest of the line not copied
            space = line.rfind(" ", start, start + CHUNK_CHARS)
            end = space + 1 if space >= 0 else start + CHUNK_CHARS
            lines.append((n, pos + start, line[start:end], heading and start == 0))
            start = end
        lines.append((n, pos + start, line[start:], heading and start == 0))
        pos += len(line)

    pieces: list[list[tuple[int, int, str]]] = []
    buf: list[tuple[int, int, str]] = []
    size = 0
    for n, at, line, heading in lines:
        if buf and (size + len(line) > CHUNK_CHARS or (heading and size > CHUNK_CHARS // 2)):
            pieces.append(buf)
            buf, size = [], 0  # carry trailing lines over as overlap, unless a new section starts
            for item in reversed(pieces[-1] if not heading else []):
                if size + len(item[2]) > min(OVERLAP_CHARS, CHUNK_CHARS - len(line)):
                    break
                buf.insert(0, item)
                size += len(item[2])
        buf.append((n, at, line))
        size += len(line)
    pieces.append(buf)

    for piece in filter(None, pieces):  # empty text has an empty piece
        start, end = piece[0][1], piece[-1][1] + len(piece[-1][2])
        body = text[start:end]
        if stripped := body.strip():
            lead = len(body) - len(body.lstrip())
            first = piece[0][0] + body.count("\n", 0, lead)  # the lines of the text, not of blank lines around it
            yield first, first + stripped.count("\n"), start + lead, len(stripped)


def _encoding(start: bytes) -> str:
    """How a text file is encoded, by its first bytes: UTF-16 or UTF-32 after their byte order marks (Windows
    PowerShell 5.1 writes UTF-16), otherwise UTF-8. UTF-32's little-endian mark starts as UTF-16's does."""
    if start[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return "utf-32"
    return "utf-16" if start[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"


def decode(raw: bytes) -> str:
    """A text file's content, with invalid bytes replaced."""
    return raw.decode(_encoding(raw), "replace")


def _lines(file: IO[bytes]) -> io.TextIOWrapper:
    """The lines of a text file open for its bytes, as decode() and a split at newlines give them, but each read only
    when it is asked for."""
    start = file.read(4)
    file.seek(0)
    return io.TextIOWrapper(file, _encoding(start), "replace", newline="\n")


class Doc(NamedTuple):
    """A document read from a file: key is "" for a whole file, the record id for a JSON Lines record; chunks are
    (first line, last line, position, length) spans of text; links are the Links of a Markdown file; scan is the bytes
    of a PDF without text, for the process that writes the index to read with OCR (see `ocr`), else None; line is a JSON
    Lines record's in its file."""

    key: str
    digest: str
    title: str
    meta: dict[str, Any]
    text: str
    chunks: list[Chunk]
    links: tuple["Link", ...] = ()
    scan: bytes | None = None
    line: int | None = None


Fields = dict[str, Any]  # a collection's field choices: "id", "title", "text" (a list), "title_from"
Reader = Callable[[bytes, str, Fields], tuple[list[Doc], list[str]]]

ID_FIELDS = ("id", "_id")
TITLE_FIELDS = ("title", "name", "headline", "subject")
TEXT_FIELDS = ("text", "content", "body", "abstract", "description", "summary")


def _digest(raw: bytes) -> str:
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _as_text(value: Json) -> str:
    """A field's text: a string, or a list of strings joined as paragraphs; "" for anything else."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n\n".join(v.strip() for v in value if v.strip())
    return ""


def _record(obj: dict[str, Any], line: int, fields: Fields) -> Doc | None:
    """One JSON object as a Doc, or None if it has no text. Without chosen fields, the text is the first of
    TEXT_FIELDS, else the longest string; the title one of TITLE_FIELDS; the id one of ID_FIELDS, else the line
    number. Every field but the text is metadata, as front matter is for Markdown."""
    id_key = fields.get("id") or next((k for k in ID_FIELDS if k in obj), None)
    key = obj.get(id_key) if id_key else None
    key = str(line) if key is None or key == "" else str(key)
    title_key = fields.get("title") or next((k for k in TITLE_FIELDS if _as_text(obj.get(k))), None)
    known = next((k for k in TEXT_FIELDS if _as_text(obj.get(k))), None)
    text_keys: list[str] = fields.get("text") or ([known] if known else [])
    if not text_keys:  # no known text field: the longest string is the text
        strings = [(len(v), k) for k, v in obj.items() if k not in (id_key, title_key) and isinstance(v, str)]
        text_keys = [max(strings)[1]] if strings else []
    body = "\n\n".join(t for k in text_keys if (t := _as_text(obj.get(k))))
    if not body:
        return None
    title = _as_text(obj.get(title_key)) if title_key else ""
    text = f"# {title}\n\n{body}" if title else body
    meta = {k: v for k, v in obj.items() if k not in text_keys}
    return Doc(key, "", title or _heading(body) or key, meta, text, list(chunk(text)))


def _text(raw: bytes, name: str, fields: Fields) -> tuple[list[Doc], list[str]]:
    """A Markdown or text file: one document, with its YAML front matter as metadata."""
    text = decode(raw).replace("\r\n", "\n")  # same chunks on every OS
    return [note(text, name, fields)._replace(digest=_digest(raw))], []


def note(text: str, name: str, fields: Fields | None = None) -> Doc:
    """The document of a Markdown or text file named `name`, from its text, with the links of a Markdown file. The
    index keeps that text, so a newer sift can read the note again without its file."""
    title, meta, body, first_line = parse(text, Path(name).stem, (fields or {}).get("title_from", "heading"))
    offset = len(text) - len(body)  # the chunks are spans of the whole file, front matter included
    chunks = [(first, last, offset + pos, length) for first, last, pos, length in chunk(body, first_line)]
    if not chunks and (m := _front_matter_block(text)) and m.group(1):  # front matter alone, as contact notes are
        chunks = [(first, last, m.start(1) + pos, length) for first, last, pos, length in chunk(m.group(1), 2)]
    found = links(text, offset, first_line) if name.lower().endswith(MARKDOWN) else ()
    return Doc("", "", title, meta, text, chunks, found)


# --- links -------------------------------------------------------------------

MARKDOWN = (".md", ".markdown")  # the files whose links sift reads; links name them without the extension
ATTACHMENTS = frozenset(
    ".avif .bmp .gif .heic .ico .jpeg .jpg .png .svg .tif .tiff .webp .3gp .aac .flac .m4a .mp3 .ogg .opus .wav "
    ".avi .mkv .mov .mp4 .ogv .webm .base .canvas .csv .json .tsv .xml .yaml .yml .7z .gz .rar .tar .zip .doc .ppt "
    ".xls".split()
)  # what links point to that sift doesn't read: a link to one is neither resolved nor missing
WIKILINK = re.compile(r"(!?)\[\[([^\[\]\n]+)\]\]")  # [[target#fragment|shown]], ![[...]] embeds
MARKDOWN_LINK = re.compile(  # [shown](target#fragment "title"), CommonMark's inline link, one level of brackets in each
    r"!?\[((?:[^\[\]\n]|\[[^\[\]\n]*\])*)\]\(\s*(<[^<>\n]*>|(?:[^\s()<>]|\([^\s()<>]*\))+)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?\s*\)"
)
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]+:")  # https:, mailto:, obsidian:: not a link between notes
BACKTICKS = re.compile(r"`+")


class Link(NamedTuple):
    """A link in a Markdown file: the front-matter field it is in (None in the text), its line in the file, its kind
    ("wikilink", "embed" for ![[...]], "markdown"), its target as written (decoded, in a Markdown link) without the
    heading or block it points into (`fragment`) and the text it shows, and the target's `key`."""

    field: str | None
    line: int
    kind: str
    target: str
    key: str
    fragment: str | None
    shown: str | None


def fold(name: str) -> str:
    """A name as names compare: case-insensitively, and composed (macOS file names arrive decomposed)."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())


def key(name: str) -> str:
    """What a file name or the last part of a link's target compares as: folded, without a Markdown extension."""
    stem, ext = posixpath.splitext(name)
    return fold(stem if ext.lower() in MARKDOWN else name)


def links(text: str, offset: int, first_line: int) -> tuple[Link, ...]:
    """The links of a Markdown file's `text` in order: wikilinks in the front matter, text[:offset], and wikilinks and
    Markdown links in the body, from line `first_line`, outside code."""
    if "[[" not in text and "](" not in text:  # most files of a collection without links, at the cost of a scan
        return ()
    found: list[Link] = []
    field: str | None = None
    for n, line in enumerate(text[:offset].split("\n")[:-1], 1):  # the front matter: `key: value` lines
        name, colon, value = line.partition(":")
        if colon and name.strip() and not name[0].isspace() and name[0] not in "#-":
            field, line = name.strip(), value
        if field and "[[" in line and not line.lstrip().startswith("#"):  # a value of `field`, not a comment
            found += _in_line(line, n, field)
    for n, line in _prose(text[offset:], first_line):
        if "[[" in line or "](" in line:
            found += _in_line(line, n)
    return tuple(found)


def _in_line(line: str, n: int, field: str | None = None) -> list[Link]:
    """The links in line `n`, in order: in the body all kinds, in the front-matter field `field` wikilinks."""
    line = _without_code(line)
    found: list[tuple[int, Link | None]] = []
    for m in WIKILINK.finditer(line):
        target, _, shown = m[2].partition("|")
        target, _, fragment = target.removesuffix("\\").partition("#")  # [[a\|b]] in a table
        found.append((m.start(), _link(field, n, "embed" if m[1] else "wikilink", target, fragment, shown)))
    if field is None and "](" in line:
        line = WIKILINK.sub(lambda m: " " * len(m[0]), line)  # [[a]](b) is one link
        for m in MARKDOWN_LINK.finditer(line):
            target = m[2][1:-1] if m[2].startswith("<") else m[2]
            if not SCHEME.match(target) and not target.startswith(("#", "//")):
                target, _, fragment = target.partition("#")
                found.append((m.start(), _link(None, n, "markdown", unquote(target), unquote(fragment), m[1])))
    return [link for _, link in sorted(found, key=lambda item: item[0]) if link]


def _link(field: str | None, n: int, kind: str, target: str, fragment: str, shown: str) -> Link | None:
    """A Link, or None for one into its own note, to a folder, or to a file sift doesn't read."""
    target = target.strip()
    last = target.rsplit("/", 1)[-1]
    if last in ("", ".", "..") or posixpath.splitext(last)[1].lower() in ATTACHMENTS:
        return None
    return Link(field, n, kind, target, key(last), fragment.strip() or None, shown.strip() or None)


def _without_code(line: str) -> str:
    """The line with its code spans, `code` or ``co`de``, blanked out: a span ends at the next run of as many
    backticks, and a run without one is just backticks."""
    if "`" not in line:
        return line
    runs = [m.span() for m in BACKTICKS.finditer(line)]
    following: list[int | None] = [None] * len(runs)  # the next run of the same length, found from the end
    last_of_length: dict[int, int] = {}
    for i in range(len(runs) - 1, -1, -1):
        length = runs[i][1] - runs[i][0]
        following[i], last_of_length[length] = last_of_length.get(length), i
    pieces: list[str] = []
    i, end = 0, 0
    while i < len(runs):
        if (j := following[i]) is None:
            i += 1
            continue
        pieces += [line[end : runs[i][0]], " " * (runs[j][1] - runs[i][0])]
        i, end = j + 1, runs[j][1]
    return "".join(pieces) + line[end:]


class _Skipped:
    """The lines of a JSON Lines file that were skipped, each with its reason: how many, and the first five."""

    def __init__(self) -> None:
        self.count = 0
        self.first: list[tuple[int | None, str]] = []

    def add(self, line: int | None, why: str) -> None:
        self.count += 1
        self.first = sorted([*self.first, (line, why)])[:5]

    def said(self) -> list[str]:
        """What to say of them, as a reader's problems: one sentence, or nothing."""
        if not self.count:
            return []
        shown = "; ".join(f"line {line}: {why}" for line, why in self.first) + ("; ..." if self.count > 5 else "")
        return [f"{console.count(self.count, 'line')} skipped ({shown})"]


def _parsed(lines: Iterable[str], fields: Fields, skipped: _Skipped) -> Iterator[Doc]:
    """The documents of a JSON Lines file's lines, one per line, as they are asked for; a line without one goes to
    `skipped`, with its reason."""
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj: Json = _json(line)
        except ValueError:
            skipped.add(n, "not JSON")
            continue
        except RecursionError:
            skipped.add(n, "nested too deeply")
            continue
        if not isinstance(obj, dict):
            skipped.add(n, "not an object")
            continue
        doc = _record(obj, n, fields)
        if doc is None:
            skipped.add(n, "no text")
        else:
            yield doc._replace(digest=_digest(line.encode()), line=n)


def _records(raw: bytes, name: str, fields: Fields) -> tuple[list[Doc], list[str]]:
    """A JSON Lines file: one document per line, and the lines skipped with their reason."""
    docs: list[Doc] = []
    skipped = _Skipped()
    seen: set[str] = set()
    for doc in _parsed(decode(raw).split("\n"), fields, skipped):
        if doc.key in seen:
            skipped.add(doc.line, f"id {doc.key} used twice")
        else:
            seen.add(doc.key)
            docs.append(doc)
    return docs, skipped.said()


_T = TypeVar("_T")

STREAMED = 32 << 20  # bytes past which a JSON Lines file is read as it is stored, not whole (see Records)


def streamed(path: str | Path, size: int) -> bool:
    """Whether a file of `size` bytes is read as it is stored (Records), not whole by `load`."""
    return size > STREAMED and extension(Path(path).name) in JSONL  # the size first: asked of every file indexed


class Unread(Exception):
    """A file read as it is stored could not be read to its end: why, as "Error: what it said"."""


class Records:
    """The documents of a JSON Lines file, read line by line as they are asked for: read whole, a file of 300,000
    records, 69 MB, took ten times its size in memory, all its lines and every record's document at once. An id used
    twice is for whoever stores them to find, among the paths it stored, and to tell with `twice`: the ids of millions
    of records are not held either. Any error of reading comes as Unread."""

    def __init__(self, path: str | Path, fields: Fields | None = None) -> None:
        self.path, self.fields, self._skipped = path, fields or {}, _Skipped()

    def __iter__(self) -> Iterator[Doc]:
        try:
            with open(self.path, "rb") as file, _lines(file) as lines:
                yield from _parsed(lines, self.fields, self._skipped)
        except Exception as error:
            raise Unread(f"{type(error).__name__}: {error}") from error

    def twice(self, doc: Doc) -> None:
        self._skipped.add(doc.line, f"id {doc.key} used twice")

    @property
    def problems(self) -> list[str]:
        """The lines skipped with their reason, as `load` gives them: of those read and told of so far."""
        return self._skipped.said()


PROPERTIES = {  # the document properties sift keeps as metadata: Xberg's name, sift's name
    "title": "title",
    "authors": "authors",
    "subject": "subject",
    "keywords": "keywords",
    "language": "language",
    "created_at": "created",
    "modified_at": "modified",
}


def _xberg(raw: bytes, name: str) -> Any:
    """Xberg's reading of a file's bytes, converted to Markdown, with its properties. Raises what Xberg raises."""
    import xberg  # only indexing documents needs it
    from xberg.options import ConversionOptions, SecurityLimits

    config = xberg.ExtractionConfig(
        output_format="markdown",
        use_cache=False,  # sift keeps what it needs in the index
        disable_ocr=True,  # Xberg's needs Tesseract; sift reads scans with a model of its own (see `ocr`)
        html_options=ConversionOptions(extract_metadata=False),  # it puts the <head> as broken YAML before the text
        # Office XML compresses at up to about 1000:1, the most deflate can do, which Xberg's default of 100:1 took
        # for zip bombs. Its caps on unpacked size, 100 MB a file and 500 MB in all, still stop those.
        security_limits=SecurityLimits(max_compression_ratio=10_000),
    )
    try:
        result = _finished(lambda: xberg.extract(xberg.ExtractInput(kind="bytes", bytes=raw, filename=name), config))
    except (Exception, KeyboardInterrupt, SystemExit):
        raise
    except BaseException as panic:  # a panic in Xberg's Rust: pyo3's PanicException is no Exception
        raise RuntimeError(f"Xberg failed: {panic}") from None
    (doc,) = result.results  # one per input
    return doc


def _finished(make: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """The result of the coroutine make() returns, run to its end. No event loop runs in this thread: sift's server
    runs its tools' work on a worker thread (see `jobs`)."""
    return asyncio.run(make())


def _document(raw: bytes, name: str, fields: Fields) -> tuple[list[Doc], list[str]]:
    """A PDF, office file, e-book or web page, converted to Markdown by Xberg: one document, with its properties
    as metadata. A PDF without text is a scan, passed on for OCR."""
    try:
        doc = _xberg(raw, name)
    except Exception as error:  # a damaged file can fail in many ways; it must not stop indexing the others
        return [], [f"skipped ({error})"]
    if not doc.content.strip() and extension(name) != ".pdf":
        return [], ["skipped (no text in it)"]
    meta = {ours: value for theirs, ours in PROPERTIES.items() if (value := getattr(doc.metadata, theirs))}
    title = meta.get("title", "").strip() or _heading(doc.content) or Path(name).stem
    if not doc.content.strip():
        return [Doc("", _digest(raw), title, meta, "", [], scan=raw)], []
    return [Doc("", _digest(raw), title, meta, doc.content, list(chunk(doc.content)))], []


MAIL_DEPTH = 3  # levels of mail attached to mail that sift reads; a mail that nests mails without end must not stall it


def _email(raw: bytes, name: str, fields: Fields, depth: int = MAIL_DEPTH) -> tuple[list[Doc], list[str]]:
    """A saved email (.eml): one document, titled by its subject, with its sender, recipients, date and ids as metadata
    (the addresses bare, for filters) and above its text, which is its plain text, else its HTML as Markdown; and one
    document for each attachment sift reads, named by its file name, and a mail attached `depth` levels deep with its
    own. A damaged mail is skipped, as a document is."""
    from email import message_from_bytes, policy

    try:
        msg = message_from_bytes(raw, policy=policy.default)
        meta, shown = _mail_headers(msg)
        if names := [n for part in msg.iter_attachments() if (n := _attachment_name(part))]:
            meta["attachments"] = names
            shown.append("Attachments: " + ", ".join(names))
        title = str(msg.get("subject") or "").strip() or Path(name).stem
        body, problems = _mail_text(msg, fields)
        text = "\n\n".join(part for part in (f"# {title}", "\n".join(shown), body) if part)
        attached, trouble = _attachments(msg, fields, depth)
    except Exception as error:  # a damaged mail can fail in many ways; it must not stop indexing the others
        return [], [f"skipped ({error})"]
    return [Doc("", _digest(raw), title, meta, text, list(chunk(text))), *attached], problems + trouble


def _mail_headers(msg: MIMEPart) -> tuple[dict[str, Any], list[str]]:
    """An email's metadata, and its headers as they show above its text: From, To, Cc and Date."""
    meta: dict[str, Any] = {}
    shown: list[str] = []
    for key in ("from", "to", "cc"):
        header = msg.get(key)
        if addresses := [a.addr_spec for a in getattr(header, "addresses", ()) if a.addr_spec]:
            meta[key] = addresses[0] if key == "from" else addresses
            shown.append(f"{key.capitalize()}: {header}")
    if date := msg.get("date"):
        when = getattr(date, "datetime", None)  # None for a date that doesn't parse: then as written
        if when and when.tzinfo:  # one without a zone (-0000) as it is
            when = _in_utc(when)
        meta["date"] = when.isoformat() if when else str(date)
        shown.append(f"Date: {meta['date']}")
    for key in ("message-id", "in-reply-to"):
        if value := str(msg.get(key) or "").strip():
            meta[key.replace("-", "_")] = value
    # Raw 8-bit bytes in a header, as old clients and spam write them, come as halves of characters: as U+FFFD
    return _whole(meta), [LONE.sub("\ufffd", line) for line in shown]


def _in_utc(when: datetime) -> datetime:
    """A time in UTC, so that dates sort as times do; as it is if UTC would take it past year 9999, or before year 1."""
    try:
        return when.astimezone(UTC)
    except OverflowError:
        return when


def _mail_text(msg: MIMEPart, fields: Fields) -> tuple[str, list[str]]:
    """An email's text: its plain part, else its HTML part as Markdown, "" if it has none; and what went wrong."""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return "", []
    try:
        content = part.get_content()
    except (LookupError, UnicodeError):  # a charset Python doesn't know: as UTF-8, with what doesn't decode replaced
        content = decode(_payload(part))
    if part.get_content_subtype() != "html":
        return content.replace("\r\n", "\n").strip(), []
    try:
        text = _xberg(content.encode(), "mail.html").content.strip()
    except Exception as error:  # Xberg's converter panics on some HTML: the words are still worth having
        return _plain(content), [f"its HTML: read as plain text, as converting it failed ({error})"]
    return text, [] if text else ["its HTML: no text in it"]


class _Words(HTMLParser):
    """The words of HTML, a paragraph per block, without scripts and styles."""

    BLOCKS = frozenset(
        "address article aside blockquote br dd div dl dt figcaption footer form h1 h2 h3 h4 h5 h6 header hr li main "
        "nav ol p pre section table td th tr ul".split()
    )
    HIDDEN = frozenset(("head", "script", "style", "template"))

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self.words: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.hidden += tag in self.HIDDEN
        if tag in self.BLOCKS:
            self.end_paragraph()

    def handle_endtag(self, tag: str) -> None:
        if tag in self.HIDDEN and self.hidden:
            self.hidden -= 1
        if tag in self.BLOCKS:
            self.end_paragraph()

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.words.append(data)

    def end_paragraph(self) -> None:
        if paragraph := " ".join("".join(self.words).split()):
            self.paragraphs.append(paragraph)
        self.words = []


def _plain(html: str) -> str:
    """HTML's words, a paragraph per block: what is left when converting it to Markdown fails."""
    words = _Words()
    words.feed(html)
    words.close()
    words.end_paragraph()
    return "\n\n".join(words.paragraphs)


ATTACHED = (*TEXT, *DOCUMENTS)  # the files attached to an email sift reads, besides mails: whole files, no records


def _attachments(msg: MIMEPart, fields: Fields, depth: int) -> tuple[list[Doc], list[str]]:
    """The documents of an email's attachments sift reads, each named by its file name, made unique, with those of a
    mail attached while `depth` lasts below it; and what had to be skipped of them."""
    docs: list[Doc] = []
    problems: list[str] = []
    taken: set[str] = set()
    for part in msg.iter_attachments():
        name = _attachment_name(part)
        if not (name and (read := _reader(part, name, depth))):
            continue
        found, trouble = read(_content(part, name), name, fields)
        key = _unique(name, taken)
        docs += [doc._replace(key=_within(key, doc.key)) for doc in found]
        problems += [f"{name}: {problem}" for problem in trouble]
    return docs, problems


def _reader(part: MIMEPart, name: str, depth: int) -> Reader | None:
    """How sift reads an attachment: a mail while `depth` lasts, a file it reads whole; None for others."""
    if _attached_mail(part) is not None or extension(name) in EMAIL:
        return partial(_email, depth=depth - 1) if depth else None
    return READERS[extension(name)] if extension(name) in ATTACHED else None


def _content(part: MIMEPart, name: str) -> bytes:
    """An attachment's bytes: a mail attached as message/rfc822 as that mail's, a text file's as UTF-8."""
    if (mail := _attached_mail(part)) is not None:
        return mail.as_bytes()
    raw = _payload(part)
    return _as_utf8(raw, part.get_content_charset()) if extension(name) in TEXT else raw


def _payload(part: MIMEPart) -> bytes:
    """A part's bytes, its transfer encoding undone; none of a part that holds parts."""
    raw = part.get_payload(decode=True)
    return raw if isinstance(raw, bytes) else b""


def _attached_mail(part: MIMEPart) -> MIMEPart | None:
    """The mail a message/rfc822 attachment holds, or None."""
    return next(part.iter_parts(), None) if part.get_content_type() == "message/rfc822" else None


def _within(key: str, inner: str) -> str:
    """The key of the document keyed `inner` of the attachment keyed `key`: the attachment's own, or one below it."""
    return f"{posixpath.splitext(key)[0]}/{inner}" if inner else key


def _as_utf8(raw: bytes, charset: str | None) -> bytes:
    """A text attachment as UTF-8, as sift reads text files: as it is if it is UTF-8 already, whatever its mail says
    (mail clients call text us-ascii that isn't), else in the charset its mail declares, if Python knows it."""
    if not charset or _utf8(raw):
        return raw
    try:
        return raw.decode(charset, "replace").encode()
    except LookupError:
        return raw


def _utf8(raw: bytes) -> bool:
    try:
        raw.decode()
    except UnicodeDecodeError:
        return False
    return True


def _attachment_name(part: MIMEPart) -> str | None:
    """An attachment's file name without any folders it claims; a mail attached without one named by its subject, or
    "message.eml". None for any other attachment without a name."""
    name = re.split(r"[\\/]", part.get_filename() or "")[-1].strip()
    if name not in ("", ".", ".."):
        return name
    if (mail := _attached_mail(part)) is None:
        return None
    subject = re.sub(r"[\\/]", "-", str(mail.get("subject") or "")).strip(" .")
    return f"{subject or 'message'}.eml"


def _unique(name: str, taken: set[str]) -> str:
    """`name`, or `name (2)` and so on if a name of `taken` is the same in any case; added to `taken`."""
    stem, ext = posixpath.splitext(name)
    key, n = name, 1
    while fold(key) in taken:
        n += 1
        key = f"{stem} ({n}){ext}"
    taken.add(fold(key))
    return key


READERS: dict[str, Reader] = {
    **dict.fromkeys(TEXT, _text),
    **dict.fromkeys(JSONL, _records),
    **dict.fromkeys(DOCUMENTS, _document),
    **dict.fromkeys(EMAIL, _email),
}
"""How sift reads each kind of file, by extension: reader(content, file name, fields) -> (documents, problems),
where fields are the collection's field choices and problems say what was skipped and why."""


def extension(name: str) -> str:
    """A file name's extension, lowercase, by which sift picks its reader; "" for none, as for `.md`, a name that
    starts with a dot."""
    return posixpath.splitext(name)[1].lower()


MOST_TEXT = 1_000_000_000  # bytes SQLite stores as one value, as it is built (SQLITE_MAX_LENGTH): a document's text


def load(path: str | Path, fields: Fields | None = None) -> tuple[list[Doc], list[str]] | None:
    """Read one file into documents with the reader for its extension (runs in worker processes). Returns
    (documents, problems), or None if the file can't be read. A text file of more than a document's text may be is
    left out unread: read, cut into passages and handed over, the store refused it, on every run."""
    path = Path(path)
    try:
        with path.open("rb") as file:
            size = os.fstat(file.fileno()).st_size
            if size > MOST_TEXT and extension(path.name) in TEXT:
                most = f"a document holds at most {MOST_TEXT:,} bytes of text"
                return [], [f"skipped ({size:,} bytes: {most}; split the file)"]
            raw = file.read()
    except OSError:
        return None
    return READERS[extension(path.name)](raw, path.name, fields or {})
