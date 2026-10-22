"""Reading files into documents, one reader per kind of file (READERS, by extension): Markdown and text files are
one document each, with YAML front matter as metadata; JSON Lines files one per line; PDF, office files, e-books and
web pages one each, converted to Markdown by Xberg; saved emails one each, and one per attachment sift reads. Every
document's text is cut into chunks with their line numbers.

Readers run in worker processes while indexing, so they read from bytes and return plain data (`Doc`).
"""

import hashlib
import json
import posixpath
import re
import unicodedata
from collections import namedtuple
from pathlib import Path
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


def _value(text):
    """A front-matter value: JSON if it parses (lists, numbers, quoted strings), else a YAML flow list of unquoted
    items, else plain text."""
    try:
        return json.loads(text, parse_constant=str)  # NaN/Infinity stay strings: SQLite's JSON rejects them
    except ValueError:
        items = _flow_list(text)
        return text.strip("'\"") if items is None else items


def _flow_list(text):
    """The items of a YAML flow list such as `[Hans Mueller, "H. Müller"]`, or None if `text` isn't one list. A single
    wikilink, `[[Acme AG]]`, is a link, as Obsidian users mean it, though YAML reads a list in a list."""
    if not (text.startswith("[") and text.endswith("]")) or WIKILINK_VALUE.fullmatch(text):
        return None
    items, depth, quote, start = [], 0, None, 1
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


def _prose(text, first_line=1):
    """The lines of Markdown `text` outside fenced code blocks, with their numbers."""
    fence = None
    for n, line in enumerate(text.split("\n"), first_line):
        if m := FENCE.match(line):
            mark = m.group(1)
            if not fence:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence) and not line[m.end() :].strip():
                fence = None
        elif not fence:
            yield n, line


def _heading(text):
    """The first Markdown heading outside fenced code blocks, or None."""
    for _, line in _prose(text):
        if HEADING.match(line):
            title = line.strip().lstrip("#").strip()
            if (bare := title.rstrip("#")) != title and (not bare or bare[-1] in " \t"):
                title = bare.strip()  # a closing sequence: "## Title ##", but not "# C#"
            if title:
                return title
    return None


def parse(text, fallback_title, title_from="heading"):
    """Split off YAML front matter. Returns title, meta, body and the body's first line number. The title is the
    `title` field, else with title_from "heading" the first heading, else `fallback_title`, the file name."""
    meta, first_line = {}, 1
    m = FRONT_MATTER.match(text)
    if m:
        meta = _front_matter(m.group(1) or "")
        first_line += text.count("\n", 0, m.end())
        text = text[m.end() :]
    title = meta.get("title")
    title = str(title).strip() if isinstance(title, str | int | float) and not isinstance(title, bool) else None
    return title or (title_from == "heading" and _heading(text)) or fallback_title, meta, text, first_line


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


Doc = namedtuple("Doc", "key digest title meta text chunks links", defaults=((),))
"""A document read from a file: key is "" for a whole file, the record id for a JSON Lines record; chunks are
(first line, last line, position, length) spans of text; links are the Links of a Markdown file."""

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
    return [note(text, name, fields)._replace(digest=_digest(raw))], []


def note(text, name, fields=None):
    """The document of a Markdown or text file named `name`, from its text, with the links of a Markdown file. The
    index keeps that text, so a newer sift can read the note again without its file."""
    title, meta, body, first_line = parse(text, Path(name).stem, (fields or {}).get("title_from", "heading"))
    offset = len(text) - len(body)  # the chunks are spans of the whole file, front matter included
    chunks = [(first, last, offset + pos, length) for first, last, pos, length in chunk(body, first_line)]
    if not chunks and (m := FRONT_MATTER.match(text)) and m.group(1):  # front matter alone, as contact notes are
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

Link = namedtuple("Link", "field line kind target key fragment shown")
"""A link in a Markdown file: the front-matter field it is in (None in the text), its line in the file, its kind
("wikilink", "embed" for ![[...]], "markdown"), its target as written (decoded, in a Markdown link) without the
heading or block it points into (`fragment`) and the text it shows, and the target's `key`."""


def fold(name):
    """A name as names compare: case-insensitively, and composed (macOS file names arrive decomposed)."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())


def key(name):
    """What a file name or the last part of a link's target compares as: folded, without a Markdown extension."""
    stem, ext = posixpath.splitext(name)
    return fold(stem if ext.lower() in MARKDOWN else name)


def links(text, offset, first_line):
    """The links of a Markdown file's `text` in order: wikilinks in the front matter, text[:offset], and wikilinks and
    Markdown links in the body, from line `first_line`, outside code."""
    if "[[" not in text and "](" not in text:  # most files of a collection without links, at the cost of a scan
        return ()
    found, field = [], None
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


def _in_line(line, n, field=None):
    """The links in line `n`, in order: in the body all kinds, in the front-matter field `field` wikilinks."""
    line, found = _without_code(line), []
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


def _link(field, n, kind, target, fragment, shown):
    """A Link, or None for one into its own note, to a folder, or to a file sift doesn't read."""
    target = target.strip()
    last = target.rsplit("/", 1)[-1]
    if last in ("", ".", "..") or posixpath.splitext(last)[1].lower() in ATTACHMENTS:
        return None
    return Link(field, n, kind, target, key(last), fragment.strip() or None, shown.strip() or None)


def _without_code(line):
    """The line with its code spans, `code` or ``co`de``, blanked out: a span ends at the next run of as many
    backticks, and a run without one is just backticks."""
    if "`" not in line:
        return line
    runs = [m.span() for m in BACKTICKS.finditer(line)]
    following, last_of_length = [None] * len(runs), {}  # the next run of the same length, found from the end
    for i in range(len(runs) - 1, -1, -1):
        length = runs[i][1] - runs[i][0]
        following[i], last_of_length[length] = last_of_length.get(length), i
    pieces, i, end = [], 0, 0
    while i < len(runs):
        if (j := following[i]) is None:
            i += 1
            continue
        pieces += [line[end : runs[i][0]], " " * (runs[j][1] - runs[i][0])]
        i, end = j + 1, runs[j][1]
    return "".join(pieces) + line[end:]


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
    import asyncio

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


def _email(raw, name, fields):
    """A saved email (.eml): one document, titled by its subject, with its sender, recipients, date and ids as metadata
    (the addresses bare, for filters) and above its text, which is its plain text, else its HTML as Markdown; and one
    document for each attachment sift reads, named by its file name. A damaged mail is skipped, as a document is."""
    from email import message_from_bytes, policy

    try:
        msg = message_from_bytes(raw, policy=policy.default)
        meta, shown = _mail_headers(msg)
        if names := [n for part in msg.iter_attachments() if (n := _file_name(part))]:
            meta["attachments"] = names
            shown.append("Attachments: " + ", ".join(names))
        title = str(msg.get("subject") or "").strip() or Path(name).stem
        body, problems = _mail_text(msg, fields)
        text = "\n\n".join(part for part in (f"# {title}", "\n".join(shown), body) if part)
        attached, trouble = _attachments(msg, fields)
    except Exception as error:  # a damaged mail can fail in many ways; it must not stop indexing the others
        return [], [f"skipped ({error})"]
    return [Doc("", _digest(raw), title, meta, text, list(chunk(text))), *attached], problems + trouble


def _mail_headers(msg):
    """An email's metadata, and its headers as they show above its text: From, To, Cc and Date."""
    meta, shown = {}, []
    for key in ("from", "to", "cc"):
        header = msg.get(key)
        if addresses := [a.addr_spec for a in getattr(header, "addresses", ()) if a.addr_spec]:
            meta[key] = addresses[0] if key == "from" else addresses
            shown.append(f"{key.capitalize()}: {header}")
    if date := msg.get("date"):
        when = getattr(date, "datetime", None)  # None for a date that doesn't parse: then as written
        meta["date"] = when.isoformat() if when else str(date)
        shown.append(f"Date: {meta['date']}")
    for key in ("message-id", "in-reply-to"):
        if value := str(msg.get(key) or "").strip():
            meta[key.replace("-", "_")] = value
    return meta, shown


def _mail_text(msg, fields):
    """An email's text: its plain part, else its HTML part as Markdown, "" if it has none; and what went wrong."""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return "", []
    try:
        content = part.get_content()
    except (LookupError, UnicodeError):  # a charset Python doesn't know: as UTF-8, with what doesn't decode replaced
        content = decode(part.get_payload(decode=True) or b"")
    if part.get_content_subtype() != "html":
        return content.replace("\r\n", "\n").strip(), []
    docs, problems = _document(content.encode(), "mail.html", fields)
    return (docs[0].text.strip() if docs else ""), [f"its HTML: {problem}" for problem in problems]


ATTACHED = (*TEXT, *DOCUMENTS)  # the attachments of an email sift reads: whole files; no records, no mail in mail


def _attachments(msg, fields):
    """The documents of an email's attachments sift reads, each named by its file name, made unique; and what had to
    be skipped of them."""
    docs, problems, taken = [], [], set()
    for part in msg.iter_attachments():
        name = _file_name(part)
        if not (name and name.lower().endswith(ATTACHED)):
            continue
        found, trouble = READERS[Path(name).suffix.lower()](part.get_payload(decode=True) or b"", name, fields)
        key = _unique(name, taken)
        docs += [doc._replace(key=key) for doc in found]
        problems += [f"{name}: {problem}" for problem in trouble]
    return docs, problems


def _file_name(part):
    """An attachment's file name without any folders it claims, or None if it has none."""
    name = re.split(r"[\\/]", part.get_filename() or "")[-1].strip()
    return name if name not in ("", ".", "..") else None


def _unique(name, taken):
    """`name`, or `name (2)` and so on if a name of `taken` is the same in any case; added to `taken`."""
    stem, ext = posixpath.splitext(name)
    key, n = name, 1
    while fold(key) in taken:
        n += 1
        key = f"{stem} ({n}){ext}"
    taken.add(fold(key))
    return key


READERS = {
    **dict.fromkeys(TEXT, _text),
    **dict.fromkeys(JSONL, _records),
    **dict.fromkeys(DOCUMENTS, _document),
    **dict.fromkeys(EMAIL, _email),
}
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
