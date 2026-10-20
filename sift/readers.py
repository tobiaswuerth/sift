"""Reading files into documents, one reader per kind of file (READERS, by extension): Markdown and text files are
one document each, with YAML front matter as metadata; JSON Lines files one per line; PDF, office files, e-books and
web pages one each, converted to Markdown by Xberg. Every document's text is cut into chunks with their line numbers.

Readers run in worker processes while indexing, so they read from bytes and return plain data (`Doc`).
"""

import asyncio
import hashlib
import json
import re
from collections import namedtuple
from pathlib import Path

from . import console

TEXT = (".md", ".markdown", ".txt")
JSONL = (".jsonl", ".ndjson")
DOCUMENTS = (".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".rtf", ".epub", ".html", ".htm")  # by Xberg
CHUNK_CHARS, OVERLAP_CHARS = 3000, 300  # ~750 tokens, well inside the models' 1024-token window


FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(?:(.*?)\r?\n)??---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
LIST_ITEM = re.compile(r"[ \t]*-[ \t]+(.*)")
HEADING = re.compile(r" {0,3}#{1,6}[ \t]")  # how a Markdown (ATX) heading starts
FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")  # how a fenced code block starts and ends
LINE = re.compile(r"[^\n]*\n|[^\n]+")  # only \n ends a line, as in editors; str.splitlines knows eight more


def _value(text):
    """A front-matter value: JSON if it parses (lists, numbers, quoted strings), else plain text."""
    try:
        return json.loads(text, parse_constant=str)  # NaN/Infinity stay strings: SQLite's JSON rejects them
    except ValueError:
        return text.strip("'\"")


def _front_matter(block):
    """Flat `key: value` lines, and lists as `- item` lines below a key without a value, as Obsidian writes them."""
    meta, listing = {}, None
    for line in block.splitlines():
        if listing and (item := LIST_ITEM.match(line)):
            meta[listing] = [*(meta[listing] or []), _value(item.group(1).strip())]
            continue
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key[0].isspace() and key[0] != "#":  # not nested, not a comment
            key, value = key.strip(), value.strip()
            meta[key] = _value(value) if value else None
            listing = None if value else key
    return meta


def _heading(text):
    """The first Markdown heading outside fenced code blocks, or None."""
    fence = None
    for line in text.split("\n"):
        if m := FENCE.match(line):
            mark = m.group(1)
            if not fence:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence) and not line[m.end() :].strip():
                fence = None
        elif not fence and HEADING.match(line):
            title = line.strip().lstrip("#").strip()
            if (bare := title.rstrip("#")) != title and (not bare or bare[-1] in " \t"):
                title = bare.strip()  # a closing sequence: "## Title ##", but not "# C#"
            if title:
                return title
    return None


def parse(text, fallback_title):
    """Split off YAML front matter. Returns title, meta, body and the body's first line number."""
    meta, first_line = {}, 1
    m = FRONT_MATTER.match(text)
    if m:
        meta = _front_matter(m.group(1) or "")
        first_line += text.count("\n", 0, m.end())
        text = text[m.end() :]
    title = meta.get("title") if isinstance(meta.get("title"), str) and meta["title"].strip() else None
    return title or _heading(text) or fallback_title, meta, text, first_line


def chunk(text, first_line=1):
    """Split text into pieces of at most CHUNK_CHARS on line boundaries, preferring headings, with a small overlap.

    Yields (first line, last line, position, length): each piece is text[position:position + length], stripped.
    """
    lines, pos = [], 0  # (line number, position, line)
    for n, line in enumerate(LINE.findall(text), first_line):
        while len(line) > CHUNK_CHARS:  # break overlong lines at whitespace
            cut = line.rfind(" ", 0, CHUNK_CHARS) + 1 or CHUNK_CHARS
            lines.append((n, pos, line[:cut]))
            pos, line = pos + cut, line[cut:]
        lines.append((n, pos, line))
        pos += len(line)

    pieces, buf, size = [], [], 0
    for n, at, line in lines:
        heading = line.startswith("#")
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


def decode(raw):
    """A text file's content: UTF-16 after a UTF-16 byte order mark (Windows PowerShell 5.1 writes those),
    otherwise UTF-8 with invalid bytes replaced."""
    return raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig", "replace")


Doc = namedtuple("Doc", "key digest title meta text chunks")
"""A document read from a file: key is "" for a whole file, the record id for a JSON Lines record; chunks are
(first line, last line, position, length) spans of text."""

ID_FIELDS = ("id", "_id")
TITLE_FIELDS = ("title", "name", "headline", "subject")
TEXT_FIELDS = ("text", "content", "body", "abstract", "description", "summary")


def _digest(raw):
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _as_text(value):
    """A field's text: a string, or a list of strings joined as paragraphs; "" for anything else."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n\n".join(v.strip() for v in value if v.strip())
    return ""


def _record(obj, line, fields):
    """One JSON object as a Doc, or None if it has no text. Without chosen fields, the text is the first of
    TEXT_FIELDS, else the longest string; the title one of TITLE_FIELDS; the id one of ID_FIELDS, else the line
    number. Every field but the text is metadata, as front matter is for Markdown."""
    id_key = fields.get("id") or next((k for k in ID_FIELDS if k in obj), None)
    key = obj.get(id_key)
    key = str(line) if key is None or key == "" else str(key)
    title_key = fields.get("title") or next((k for k in TITLE_FIELDS if _as_text(obj.get(k))), None)
    text_keys = fields.get("text") or [next((k for k in TEXT_FIELDS if _as_text(obj.get(k))), None)]
    if text_keys == [None]:  # no known text field: the longest string is the text
        strings = [(len(v), k) for k, v in obj.items() if k not in (id_key, title_key) and isinstance(v, str)]
        text_keys = [max(strings)[1]] if strings else []
    body = "\n\n".join(t for k in text_keys if (t := _as_text(obj.get(k))))
    if not body:
        return None
    title = _as_text(obj.get(title_key)) if title_key else ""
    text = f"# {title}\n\n{body}" if title else body
    meta = {k: v for k, v in obj.items() if k not in text_keys}
    return Doc(key, "", title or _heading(body) or key, meta, text, list(chunk(text)))


def _text(raw, name, fields):
    """A Markdown or text file: one document, with its YAML front matter as metadata."""
    text = decode(raw).replace("\r\n", "\n")  # same chunks on every OS
    title, meta, body, first_line = parse(text, Path(name).stem)
    offset = len(text) - len(body)  # the chunks are spans of the whole file, front matter included
    chunks = [(first, last, offset + pos, length) for first, last, pos, length in chunk(body, first_line)]
    return [Doc("", _digest(raw), title, meta, text, chunks)], []


def _records(raw, name, fields):
    """A JSON Lines file: one document per line, and the lines skipped with their reason."""
    docs, problems, seen = [], [], set()
    for n, line in enumerate(decode(raw).split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line, parse_constant=str)  # NaN/Infinity stay strings: SQLite's JSON rejects them
        except ValueError:
            problems.append(f"line {n}: not JSON")
            continue
        if not isinstance(obj, dict):
            problems.append(f"line {n}: not an object")
            continue
        doc = _record(obj, n, fields)
        if doc is None:
            problems.append(f"line {n}: no text")
        elif doc.key in seen:
            problems.append(f"line {n}: id {doc.key} used twice")
        else:
            seen.add(doc.key)
            docs.append(doc._replace(digest=_digest(line.encode())))
    if problems:
        shown = "; ".join(problems[:5]) + ("; ..." if len(problems) > 5 else "")
        problems = [f"{console.count(len(problems), 'line')} skipped ({shown})"]
    return docs, problems


PROPERTIES = {  # the document properties sift keeps as metadata: Xberg's name, sift's name
    "title": "title",
    "authors": "authors",
    "subject": "subject",
    "keywords": "keywords",
    "language": "language",
    "created_at": "created",
    "modified_at": "modified",
}


def _document(raw, name, fields):
    """A PDF, office file, e-book or web page, converted to Markdown by Xberg: one document, with its properties
    as metadata. Without OCR a scan has no text."""
    import xberg  # only indexing documents needs it
    from xberg.options import ConversionOptions, SecurityLimits

    config = xberg.ExtractionConfig(
        output_format="markdown",
        use_cache=False,  # sift keeps what it needs in the index
        disable_ocr=True,  # slow, and needs Tesseract or a model
        html_options=ConversionOptions(extract_metadata=False),  # it puts the <head> as broken YAML before the text
        # Office XML compresses at up to about 1000:1, the most deflate can do, which Xberg's default of 100:1 took
        # for zip bombs. Its caps on unpacked size, 100 MB a file and 500 MB in all, still stop those.
        security_limits=SecurityLimits(max_compression_ratio=10_000),
    )
    try:
        result = asyncio.run(xberg.extract(xberg.ExtractInput(kind="bytes", bytes=raw, filename=name), config))
        (doc,) = result.results  # one per input
    except Exception as error:  # a damaged file can fail in many ways; it must not stop indexing the others
        return [], [f"skipped ({error})"]
    if not doc.content.strip():
        return [], ["skipped (no text in it: a scan? sift runs no OCR)"]
    meta = {ours: value for theirs, ours in PROPERTIES.items() if (value := getattr(doc.metadata, theirs))}
    title = meta.get("title", "").strip() or _heading(doc.content) or Path(name).stem
    return [Doc("", _digest(raw), title, meta, doc.content, list(chunk(doc.content)))], []


READERS = {**dict.fromkeys(TEXT, _text), **dict.fromkeys(JSONL, _records), **dict.fromkeys(DOCUMENTS, _document)}
"""How sift reads each kind of file, by extension: reader(content, file name, fields) -> (documents, problems),
where fields are the collection's field choices and problems say what was skipped and why."""


def load(path, fields=None):
    """Read one file into documents with the reader for its extension (runs in worker processes). Returns
    (documents, problems), or None if the file can't be read."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return READERS[path.suffix.lower()](raw, path.name, fields or {})
