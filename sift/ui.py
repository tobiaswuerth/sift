"""`sift ui`: the index in a web browser, to search, browse and read the sources yourself.

`sift serve` serves it under /ui: one page of three panes, the filters and folders, the results, and the document
they lead to. htmx swaps a pane in place; every view has its own address, so links, reloading and the back button
work. The server renders everything, so it is tested as the rest of sift is, and needs no build step."""

import re
import socket
import sqlite3
import time
import urllib.error
import urllib.request
import webbrowser
from collections import Counter
from dataclasses import dataclass, fields, replace
from itertools import islice
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from markupsafe import Markup, escape

from . import console, readers, store
from .search import STOPWORDS, line_span

WEB = Path(__file__).parent / "web"
PREFIX = "/ui/"
MODES = {  # the searches, by name: their label and what they do
    "search": ("Keyword", "Exact words (BM25): fast, as you type"),
    "vsearch": ("Meaning", "Closeness in meaning: press Enter"),
    "query": ("Hybrid", "Keywords and meaning, reranked: best for questions, press Enter"),
}
PAGE = 30  # results per page, and how many more "More" shows
MOST = 500  # results one page can ask for
FACET_VALUES = 8  # values shown per front-matter field, the most common first
SHORT = 60  # characters a front-matter value may have to filter by it
SNIPPET = 260  # characters of a passage shown in the results
READER_LINES = 2000  # lines of a document the reader shows at once: a long log would make a page of megabytes
LOCAL_HOSTS = ("127.0.0.1", "localhost")
STARTUP_SECONDS = 600  # how long `sift ui` waits for the server it started: loading vectors can take minutes
ERRORS = (LookupError, ValueError, ConnectionError, sqlite3.Error)  # what a view shows, as the MCP tools say it


@dataclass(frozen=True)
class View:
    """What the page shows, as its address says: the search, the folder or filters, and the document read."""

    q: str = ""
    mode: str = "search"
    c: str = ""  # collection
    path: str = ""  # folder in it, e.g. "2024/"
    f: tuple = ()  # front-matter filters, "field=value"
    sort: str = ""
    k: int = PAGE
    similar: str = ""  # a document whose like to list
    doc: str = ""  # the document in the reader
    lines: str = ""  # its lines to mark
    at: int = 0  # the first of its lines to show; 0: a little before the marked ones

    @classmethod
    def of(cls, params):
        k, at, path = params.get("k", ""), params.get("at", ""), params.get("path", "").strip("/")
        return cls(
            q=params.get("q", "").strip(),
            mode=params.get("mode") if params.get("mode") in MODES else "search",
            c=params.get("c", ""),
            path=path and path + "/",  # "2024", "/2024/": the folder 2024/
            f=tuple(params.getlist("f")),
            sort=params.get("sort", ""),
            k=min(int(k), MOST) if re.fullmatch("[0-9]+", k) and int(k) > 0 else PAGE,
            similar=params.get("similar", ""),
            doc=params.get("doc", ""),
            lines=params.get("lines", ""),
            at=int(at) if re.fullmatch("[0-9]+", at) else 0,
        )

    def href(self, **changes):
        """The address of this view with `changes`; values left at their defaults stay out of it."""
        view, default = replace(self, **changes), View()
        pairs = [
            (name, value)
            for name in (field.name for field in fields(View))
            if getattr(view, name) != getattr(default, name)
            for value in (view.f if name == "f" else [getattr(view, name)])
        ]
        return PREFIX + ("?" + urlencode(pairs) if pairs else "")

    def filters(self):
        """The filters as searches take them: the values of one field stand for any of them."""
        found = {}
        for pair in self.f:
            key, equals, value = pair.partition("=")
            if not equals:
                raise ValueError(f"filter {pair!r} must be field=value")
            found.setdefault(key, []).extend(_forms(value))
        return found

    def filtered(self, key, value):
        """This view with the filter key=value added, or removed if it is there; on the first page."""
        pair = f"{key}={value}"
        return self.href(f=tuple(p for p in self.f if p != pair) if pair in self.f else (*self.f, pair), k=PAGE)


def _forms(text):
    """A filter value from the address in each form the front matter may hold it: as text, and as the number or yes
    or no it reads as. "2024" is a number in one document, "007" text in another."""
    if re.fullmatch(r"-?[0-9]+", text):
        return [text, int(text)]
    if re.fullmatch(r"-?[0-9]+\.[0-9]+", text):
        return [text, float(text)]
    return [text, text == "true"] if text in ("true", "false") else [text]


def _shown(value):
    """A front-matter value as the page shows it, and as a filter names it: yes and no as JSON writes them."""
    return ("true" if value else "false") if isinstance(value, bool) else value


def _glob(path):
    """A folder's path as a glob that matches what is inside it: [, * and ? stand for themselves."""
    return re.sub(r"[\[*?]", r"[\g<0>]", path) + "*"


def _pattern(query):
    """A regex for the words of a query, any ending, to mark them; None for a query without any."""
    terms = {w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS | {"and", "or", "not", "near"}}
    terms = sorted((t for t in terms if len(t) > 1), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(map(re.escape, terms)) + r")\w*", re.IGNORECASE) if terms else None


def _marked(text, pattern):
    """Text as HTML, its matches of `pattern` in <mark>. Escaped piece by piece, so a match can't land inside an
    escape such as &amp;."""
    if pattern is None:
        return escape(text)
    out, last = [], 0
    for m in pattern.finditer(text):
        out += [escape(text[last : m.start()]), Markup("<mark>"), escape(m[0]), Markup("</mark>")]
        last = m.end()
    return Markup("").join([*out, escape(text[last:])])


def _snippet(text, pattern, title=""):
    """A passage on one line, about SNIPPET characters from a little before its first match, marked; without
    Markdown's heading marks, and without the title the results show anyway."""
    text = re.sub(r"(?<!\S)#{1,6}\s", "", " ".join(text.split()))
    text = text[len(title) :].lstrip() if title and text.startswith(title) else text
    match = pattern and pattern.search(text)
    start = max(0, match.start() - SNIPPET // 4) if match else 0
    piece = text[start : start + SNIPPET]
    return _marked(("…" if start else "") + piece + ("…" if start + SNIPPET < len(text) else ""), pattern)


def _short(value):
    return isinstance(value, str | int | float) and not isinstance(value, bool) and len(str(value)) <= SHORT


def _link(value):
    return isinstance(value, str) and re.match(r"https?://\S+$", value) is not None


def _values(value):
    return value if isinstance(value, list) else [value]


def _facets(hits, view):
    """For each front-matter field of the hits, its most common short values with their counts, as filters: first the
    fields whose values most hits share. A field whose values all differ, such as an id or a link, helps nobody
    filter, unless it has few."""
    counts = {}
    for hit in hits:
        for key, value in hit.get("fields", {}).items():
            for one in map(_shown, _values(value)):
                if _short(one) and not _link(one):
                    counts.setdefault(key, Counter())[one] += 1
    facets = []
    for key, counter in sorted(counts.items(), key=lambda kc: (-max(kc[1].values()), kc[0])):  # the most shared first
        if len(counter) > FACET_VALUES and max(counter.values()) == 1:
            continue
        values = [
            {"value": value, "count": n, "href": view.filtered(key, value), "on": f"{key}={value}" in view.f}
            for value, n in counter.most_common(FACET_VALUES)
        ]
        facets.append((key, values))
    return facets


def _collections(db):
    """The collections and their number of documents, by name."""
    sql = """SELECT co.name, count(d.id) FROM collections co LEFT JOIN docs d ON d.collection = co.id
             GROUP BY co.id ORDER BY co.name"""
    return db.execute(sql).fetchall()


def _crumbs(view):
    """The collection and folders the view is in, each a link to browse it."""
    if not view.c:
        return []
    crumbs, path = [(view.c, view.href(path="", q="", similar="", f=(), k=PAGE))], ""
    for part in view.path.strip("/").split("/") if view.path else []:
        path += part + "/"
        crumbs.append((part, view.href(path=path, q="", similar="", f=(), k=PAGE)))
    return crumbs


def _scored(hits, view, pattern):
    """The hits as the results list shows them: a link to read each, its passage shortened, its score as a bar."""
    scores = [h["score"] for h in hits if h.get("score") is not None]
    low, high = (min(scores), max(scores)) if scores else (0, 0)
    for hit in hits:
        hit["href"] = view.href(doc=hit["id"], lines=hit.get("lines", ""), at=0)
        hit["snippet"] = _snippet(hit.get("text", ""), pattern, hit["title"])
        if "score" in hit:
            hit["bar"] = round((hit["score"] - low) / (high - low), 3) if high > low else 1
    return hits


def _explore(index, view, pattern):
    """The filters, folders and results of a view."""
    options = {
        "collection": view.c or None,
        "path": view.path and _glob(view.path),
        "fields": ["*"],
        "sort": view.sort or None,
    }
    context = {"kind": "overview", "hits": [], "folders": [], "crumbs": _crumbs(view), "error": None}
    try:
        options["filters"] = view.filters()
        if view.q:  # typed while documents like one are listed: a search
            context |= {"kind": "results", "hits": getattr(index, view.mode)(view.q, view.k, **options)}
        elif view.similar:
            context |= {"kind": "similar", "hits": index.similar(view.similar, view.k, **options)}
            context["similar_title"] = index.document(view.similar)["title"]
        elif view.c or view.f:
            context |= {"kind": "documents", "hits": index.browse(view.k, **options)}
            if view.c:
                context["folders"] = [
                    (name, view.href(path=view.path + name + "/", k=PAGE)) for name in index.folders(view.c, view.path)
                ]
    except ERRORS as e:
        context["error"] = str(e)
    hits = _scored(context["hits"], view, pattern)
    sortable = sorted({key for hit in hits for key in hit.get("fields", {})} | ({view.sort.lstrip("-")} - {""}))
    return context | {
        "facets": _facets(hits, view),
        "chips": [(pair.replace("=", ": ", 1), view.filtered(*pair.split("=", 1))) for pair in view.f if "=" in pair],
        "more": view.href(k=view.k + PAGE) if len(hits) == view.k and view.k < MOST else None,
        "sortable": sortable,
    }


def _lines(text, span, pattern, at):
    """READER_LINES lines of a document, from line `at` on, for the reader: number, text marked, whether in the hit,
    and what kind of line it is."""
    rows = []
    for n, found in enumerate(islice(readers.LINE.finditer(text), at - 1, at - 1 + READER_LINES), at):
        line, kind = found[0].rstrip("\n"), "t" if found[0].lstrip().startswith("|") else ""
        if m := re.match(r"(#{1,6}) ", line):  # a heading, shown as one rather than with its marks
            line, kind = line[m.end() :], "h" + str(min(len(m[1]), 3))
        rows.append({"n": n, "text": _marked(line, pattern), "hit": span and span[0] <= n <= span[1], "kind": kind})
    return rows


def _field(view, key, value):
    """A front-matter value in the reader: a link opens its page, a short value filters by it, others are text."""
    value = _shown(value)
    if _link(value):
        return {"value": value, "link": value}
    return {"value": value, "href": view.filtered(key, value) if _short(value) else None}


def _reader(index, view, pattern):
    """The document a view reads, with its front matter as filters; None if it reads none."""
    if not view.doc:
        return None
    try:
        span = line_span(view.lines) if view.lines else None
        document = index.document(view.doc, fields=["*"])
        path = index.file(document["id"])
    except ERRORS as e:
        return {"error": str(e)}
    fields_ = [(key, [_field(view, key, v) for v in _values(value)]) for key, value in document["fields"].items()]
    text = document["text"]
    total = text.count("\n") + (not text.endswith("\n") and text != "")
    at = min(view.at or (max(1, span[0] - READER_LINES // 10) if span else 1), max(1, total))
    end = min(total, at + READER_LINES - 1)
    return {
        "id": document["id"],
        "title": document["title"],
        "fields": fields_,
        "lines": _lines(text, span, pattern, at),
        "at": at,
        "part": f"lines {at:,} to {end:,} of {total:,}" if at > 1 or end < total else None,
        "earlier": view.href(at=max(1, at - READER_LINES)) if at > 1 else None,
        "later": view.href(at=end + 1) if end < total else None,
        "file": PREFIX + "file?" + urlencode({"id": document["id"]}) if path and path.is_file() else None,
        "similar": view.href(similar=document["id"], q="", k=PAGE),
    }


def app(index, host="127.0.0.1"):
    """The UI's ASGI app over `index`, to serve under /ui. Served on a local address, it answers only requests that
    name a local host: a web page could otherwise reach it under its own domain (DNS rebinding) and read the
    documents. Its handlers are async, so searches run on the event loop's thread, as PyTorch needs."""
    from jinja2 import Environment, FileSystemLoader
    from starlette.applications import Starlette
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from starlette.responses import FileResponse, HTMLResponse, PlainTextResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles

    env = Environment(loader=FileSystemLoader(WEB), autoescape=True, trim_blocks=True, lstrip_blocks=True)
    env.filters["num"] = lambda n: f"{n:,}"
    env.globals["PAGE"] = PAGE
    counted = {}  # the collections, by the state of the index: counting millions of documents per keystroke is slow

    def collections():
        # Removing documents changes the version; adding them, the newest document's id.
        version = store.get_meta(index.db, "version"), index.db.execute("SELECT max(id) FROM docs").fetchone()[0]
        if version not in counted:
            counted.clear()
            counted[version] = _collections(index.db)
        return counted[version]

    async def page(request):
        view = View.of(request.query_params)
        part = request.headers.get("hx-target") if "hx-history-restore-request" not in request.headers else None
        pattern = _pattern(view.q)
        context = {
            "view": view,
            "modes": MODES,
            "oob": part == "explore",  # the form's parts come along, to show the view
            "typed": request.headers.get("hx-trigger") == "search",  # the box holds what is being typed: it stays
            "title": f"{view.q} · sift" if view.q else "sift",
            "collections": collections(),
        }
        if part != "reader":
            context |= _explore(index, view, pattern)
        if part != "explore":
            context["reader"] = _reader(index, view, pattern)
        name = {"explore": "explore.html", "reader": "reader.html"}.get(part, "page.html")
        return HTMLResponse(env.get_template(name).render(context))

    async def file(request):
        try:
            path = index.file(request.query_params.get("id", ""))
        except LookupError:
            path = None
        if path is None or not path.is_file():
            return PlainTextResponse("no such file", status_code=404)
        response = FileResponse(path, filename=path.name, content_disposition_type="inline")
        # A file runs no scripts as the UI, where a saved web page could read the whole index. Chromium's PDF viewer
        # refuses sandboxed documents, and runs none of a PDF's scripts as the page anyway.
        if response.media_type != "application/pdf":
            response.headers["Content-Security-Policy"] = "sandbox"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    routes = [Route("/", page), Route("/file", file), Mount("/static", StaticFiles(directory=WEB / "static"))]
    ui = Starlette(routes=routes)
    return TrustedHostMiddleware(ui, allowed_hosts=list(LOCAL_HOSTS)) if host in LOCAL_HOSTS else ui


def address(host, port):
    """Where to reach a server listening on `host`: this machine's own address if it listens on all of them, and
    IPv6 addresses in brackets."""
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def open_in_browser(host="127.0.0.1", port=8765):
    """`sift ui`: open the UI of the `sift serve` on this port in the browser, starting the server if none runs."""
    url = address(host, port) + PREFIX
    try:
        urllib.request.urlopen(url, timeout=5).close()
    except urllib.error.HTTPError as e:
        e.close()  # it holds the answer open: a ResourceWarning on Python 3.14+ when left to the garbage collector
        raise SystemExit(
            f"error: what answers on port {port} is no sift with a UI: an older `sift serve`? Restart it with this "
            "version of sift, or choose another port with --port"
        ) from None
    except OSError:  # nobody listens: serve here, and open the page once the server answers
        import threading

        from . import server

        threading.Thread(target=_open_when_up, args=(url,), daemon=True).start()
        server.serve(host, port)
        return
    _open(url)


def _open(url):
    console.say(f"sift ui on {url}")
    webbrowser.open(url)


def _open_when_up(url, wait=STARTUP_SECONDS):
    """Open the page once something listens on its port, within `wait` seconds."""
    deadline, parts = time.monotonic() + wait, urlsplit(url)
    while time.monotonic() < deadline:
        try:
            socket.create_connection((parts.hostname, parts.port), timeout=0.2).close()
        except OSError:
            time.sleep(0.1)
            continue
        _open(url)
        return
