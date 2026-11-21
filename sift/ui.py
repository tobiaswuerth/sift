"""`sift ui`: the index in a web browser, to search, browse and read the sources yourself.

`sift serve` serves it under /ui: one page of three panes, the filters and folders, the results, and the document
they lead to. htmx swaps a pane in place; every view has its own address, so links, reloading and the back button
work. The server renders everything, so it is tested as the rest of sift is, and needs no build step."""

import ipaddress
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.request
import webbrowser
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit

from markupsafe import Markup, escape

from . import console, jobs, readers, store
from .filters import each
from .hits import Hit, line_span
from .queries import words

if TYPE_CHECKING:
    from starlette.datastructures import QueryParams
    from starlette.types import ASGIApp, Receive, Scope, Send

    from .catalog import Collection
    from .search import Index

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
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
HOST_NAME = re.compile(r"[a-z0-9_-]+(\.[a-z0-9_-]+)*")  # a host's name, as a request's Host header holds it
STARTUP_SECONDS = 600  # how long `sift ui` waits for the server it started: loading models and codes takes a while
PROBE_SECONDS = 5  # how long `sift ui` waits for a server on its port to answer, before it takes it for busy
# A page and its parts share an address, told apart by the request's headers: a browser that kept one showed a part
# alone, without the page around it, on the way back to the address. Nor do the documents' texts belong in its cache.
UNKEPT = {"Cache-Control": "no-store"}
FRAMED_BY_NONE = {"Content-Security-Policy": "frame-ancestors 'none'"}  # no other site shows the UI in a frame
ERRORS = (LookupError, ValueError, ConnectionError, sqlite3.Error)  # what a view shows, as the MCP tools say it
BUGS = (KeyError, IndexError)  # LookupErrors too, but never what sift says of a view: the server's error page, logged

Context = dict[str, Any]  # what a template is rendered with


@dataclass(frozen=True)
class View:
    """What the page shows, as its address says: the search, the folder or filters, and the document read."""

    q: str = ""
    mode: str = "search"
    c: str = ""  # collection
    path: str = ""  # folder in it, e.g. "2024/"
    f: tuple[str, ...] = ()  # front-matter filters, "field=value"
    sort: str = ""
    k: int = PAGE
    similar: str = ""  # a document whose like to list
    to: str = ""  # a note: only documents linking to it
    by: str = ""  # a note: only documents it links to
    doc: str = ""  # the document in the reader
    lines: str = ""  # its lines to mark
    at: int = 0  # the first of its lines to show; 0: a little before the marked ones

    @classmethod
    def of(cls, params: "QueryParams") -> "View":
        k, at, path = _count(params.get("k", "")), _count(params.get("at", "")), params.get("path", "").strip("/")
        mode = params.get("mode", "")
        return cls(
            q=params.get("q", "").strip(),
            mode=mode if mode in MODES else "search",
            c=params.get("c", ""),
            path=path and path + "/",  # "2024", "/2024/": the folder 2024/
            f=tuple(params.getlist("f")),
            sort=params.get("sort", ""),
            k=min(k, MOST) if k else PAGE,
            similar=params.get("similar", ""),
            to=params.get("to", ""),
            by=params.get("by", ""),
            doc=params.get("doc", ""),
            lines=params.get("lines", ""),
            at=at or 0,
        )

    def href(self, **changes: Any) -> str:
        """The address of this view with `changes`; values left at their defaults stay out of it."""
        view, default = replace(self, **changes), View()
        pairs = [
            (name, value)
            for name in (field.name for field in fields(View))
            if getattr(view, name) != getattr(default, name)
            for value in (view.f if name == "f" else [getattr(view, name)])
        ]
        return PREFIX + ("?" + urlencode(pairs) if pairs else "")

    def filters(self) -> dict[str, list[str]]:
        """The filters as searches take them: the values of one field stand for any of them, each in whatever form
        the front matter holds it, a number or text."""
        found: dict[str, list[str]] = {}
        for pair in self.f:
            key, equals, value = pair.partition("=")
            if not equals:
                raise ValueError(f"filter {pair!r} must be field=value")
            found.setdefault(key, []).append(value)
        return found

    def filtered(self, key: str, value: object) -> str:
        """This view with the filter key=value added, or removed if it is there; on the first page."""
        pair = f"{key}={value}"
        return self.href(f=tuple(p for p in self.f if p != pair) if pair in self.f else (*self.f, pair), k=PAGE)


def _count(text: str) -> int | None:
    """A count from the address, or None for anything else: one too long for int() to read, too."""
    return int(text) if re.fullmatch("[0-9]{1,9}", text) else None


def _shown(value: Any) -> Any:
    """A front-matter value as the page shows it, and as a filter names it: yes and no as JSON writes them."""
    return ("true" if value else "false") if isinstance(value, bool) else value


def _glob(path: str) -> str:
    """A folder's path as a glob that matches what is inside it: [, * and ? stand for themselves."""
    return re.sub(r"[\[*?]", r"[\g<0>]", path) + "*"


def _pattern(query: str) -> re.Pattern[str] | None:
    """A regex for the words of a query, as searches take them (queries.words), any ending, to mark them; None for a
    query without any."""
    terms = sorted(set(words(query)) - {"not", "near"}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(map(re.escape, terms)) + r")\w*", re.IGNORECASE) if terms else None


def _marked(text: str, pattern: re.Pattern[str] | None) -> Markup:
    """Text as HTML, its matches of `pattern` in <mark>. Escaped piece by piece, so a match can't land inside an
    escape such as &amp;."""
    if pattern is None:
        return escape(text)
    out: list[Markup] = []
    last = 0
    for m in pattern.finditer(text):
        out += [escape(text[last : m.start()]), Markup("<mark>"), escape(m[0]), Markup("</mark>")]
        last = m.end()
    return Markup("").join([*out, escape(text[last:])])


def _snippet(text: str, pattern: re.Pattern[str] | None, title: str = "") -> Markup:
    """A passage on one line, about SNIPPET characters from a little before its first match, marked; without
    Markdown's heading marks, and without the title the results show anyway."""
    text = re.sub(r"(?<!\S)#{1,6}\s", "", " ".join(text.split()))
    text = text[len(title) :].lstrip() if title and text.startswith(title) else text
    match = pattern and pattern.search(text)
    start = max(0, match.start() - SNIPPET // 4) if match else 0
    piece = text[start : start + SNIPPET]
    return _marked(("…" if start else "") + piece + ("…" if start + SNIPPET < len(text) else ""), pattern)


def _short(value: object) -> bool:
    return isinstance(value, str | int | float) and not isinstance(value, bool) and len(str(value)) <= SHORT


def _link(value: object) -> bool:
    return isinstance(value, str) and re.match(r"https?://\S+$", value) is not None


def _facets(hits: Iterable[Hit], view: View) -> list[tuple[str, list[dict[str, Any]]]]:
    """For each front-matter field of the hits, its most common short values with their counts, as filters: first the
    fields whose values most hits share. A field whose values all differ, such as an id or a link, helps nobody
    filter, unless it has few."""
    counts: dict[str, Counter[Any]] = {}
    for hit in hits:
        for key, value in hit.get("fields", {}).items():
            for one in map(_shown, each(value)):
                if _short(one) and not _link(one):
                    counts.setdefault(key, Counter())[one] += 1
    facets: list[tuple[str, list[dict[str, Any]]]] = []
    for key, counter in sorted(counts.items(), key=lambda kc: (-max(kc[1].values()), kc[0])):  # the most shared first
        if len(counter) > FACET_VALUES and max(counter.values()) == 1:
            continue
        values = [
            {"value": value, "count": n, "href": view.filtered(key, value), "on": f"{key}={value}" in view.f}
            for value, n in counter.most_common(FACET_VALUES)
        ]
        facets.append((key, values))
    return facets


def _collections(stores: Iterable[tuple["Collection", sqlite3.Connection]]) -> list[tuple[str, int | None, str | None]]:
    """The collections of [(collection, its store)], each as (name, its number of documents as its store keeps it,
    None), or of a store found damaged (name, None, what is wrong and what to do)."""
    listed: list[tuple[str, int | None, str | None]] = []
    for c, db in stores:
        try:
            listed.append((c.name, store.counts(db)["docs"], None))
        except sqlite3.DatabaseError as error:  # one store damaged keeps no page from showing the others
            listed.append((c.name, None, store.damage(c.name, error)))
    return listed


def _crumbs(view: View) -> list[tuple[str, str]]:
    """The collection and folders the view is in, each a link to browse it."""
    if not view.c:
        return []
    crumbs, path = [(view.c, view.href(path="", q="", similar="", f=(), to="", by="", k=PAGE))], ""
    for part in view.path.strip("/").split("/") if view.path else []:
        path += part + "/"
        crumbs.append((part, view.href(path=path, q="", similar="", f=(), to="", by="", k=PAGE)))
    return crumbs


def _scored(hits: list[Hit], view: View, pattern: re.Pattern[str] | None) -> list[Hit]:
    """The hits as the results list shows them: a link to read each, its passage shortened, its score as a bar."""
    scores = [h["score"] for h in hits if h.get("score") is not None]
    low, high = (min(scores), max(scores)) if scores else (0, 0)
    for hit in hits:
        hit["href"] = view.href(doc=hit["id"], lines=hit.get("lines", ""), at=0)
        hit["snippet"] = _snippet(hit.get("text", ""), pattern, hit["title"])
        if "score" in hit:
            hit["bar"] = round((hit["score"] - low) / (high - low), 3) if high > low else 1
    return hits


def _explore(index: "Index", view: View, pattern: re.Pattern[str] | None) -> Context:
    """The filters, folders and results of a view."""
    options: dict[str, Any] = {
        "collection": view.c or None,
        "path": view.path and _glob(view.path),
        "fields": ["*"],
        "sort": view.sort or None,
        "links_to": [view.to] if view.to else None,
        "linked_from": [view.by] if view.by else None,
    }
    context: Context = {
        "kind": "overview",
        "hits": [],
        "folders": [],
        "crumbs": _crumbs(view),
        "error": None,
        "linking": [],
    }
    # One result more than the page tells whether there are more; but a search sorted by a field sorts its k best,
    # which one more would change.
    peek = not (view.sort and (view.q or view.similar))
    ask = view.k + peek
    try:
        options["filters"] = view.filters()
        if view.q:  # typed while documents like one are listed: a search
            context |= {"kind": "results", "hits": getattr(index, view.mode)(view.q, ask, **options)}
        elif view.similar:
            context |= {"kind": "similar", "hits": index.similar(view.similar, ask, **options)}
            context["similar_title"] = index.document(view.similar)["title"]
        elif view.c or view.f or view.to or view.by:
            context |= {"kind": "documents", "hits": index.browse(ask, **options)}
            if view.c:
                context["folders"] = [
                    (name, view.href(path=view.path + name + "/", k=PAGE)) for name in index.folders(view.c, view.path)
                ]
        for note, label, name in ((view.to, "Linking to", "to"), (view.by, "Linked from", "by")):
            if note:
                context["linking"].append((f"{label} {_title(index, note)}", view.href(**{name: ""}, k=PAGE)))
    except BUGS:
        raise
    except ERRORS as e:
        context["error"] = str(e)
    found = context["hits"]
    hits = _scored(found[: view.k], view, pattern)
    more = len(found) > view.k if peek else len(found) == view.k
    sortable = sorted({key for hit in hits for key in hit.get("fields", {})} | ({view.sort.lstrip("-")} - {""}))
    filters = [(pair.replace("=", ": ", 1), view.filtered(*pair.split("=", 1))) for pair in view.f if "=" in pair]
    return context | {
        "hits": hits,
        "facets": _facets(hits, view),
        "chips": context["linking"] + filters,
        "more": view.href(k=view.k + PAGE) if more and view.k < MOST else None,
        "sortable": sortable,
    }


def _lines(text: str, span: tuple[int, int] | None, pattern: re.Pattern[str] | None, at: int) -> list[dict[str, Any]]:
    """READER_LINES lines of a document, from line `at` on, for the reader: number, text marked, whether in the hit,
    and what kind of line it is."""
    rows: list[dict[str, Any]] = []
    for n, found in enumerate(islice(readers.LINE.finditer(text), at - 1, at - 1 + READER_LINES), at):
        line, kind = found[0].rstrip("\n"), "t" if found[0].lstrip().startswith("|") else ""
        if m := re.match(r"(#{1,6}) ", line):  # a heading, shown as one rather than with its marks
            line, kind = line[m.end() :], "h" + str(min(len(m[1]), 3))
        rows.append({"n": n, "text": _marked(line, pattern), "hit": span and span[0] <= n <= span[1], "kind": kind})
    return rows


def _field(view: View, key: str, value: Any) -> dict[str, Any]:
    """A front-matter value in the reader: a link opens its page, a short value filters by it, others are text."""
    value = _shown(value)
    if _link(value):
        return {"value": value, "link": value}
    return {"value": value, "href": view.filtered(key, value) if _short(value) else None}


def _title(index: "Index", note: str) -> str:
    """A note's title, or as named if it names none: a link target, in an address written by hand."""
    try:
        return index.document(note)["title"]
    except LookupError:
        return note


def _reader(index: "Index", view: View, pattern: re.Pattern[str] | None) -> Context | None:
    """The document a view reads, with its front matter as filters, and the notes linking it and that it links;
    None if it reads none."""
    if not view.doc:
        return None
    try:
        span = line_span(view.lines) if view.lines else None
        document = index.document(view.doc, fields=["*"])
        path = index.file(document["id"])
        into, out = index.link_count(document["id"])
    except BUGS:
        raise
    except ERRORS as e:
        return {"error": str(e)}
    listing = {"q": "", "similar": "", "c": "", "path": "", "f": (), "k": PAGE}  # all of them, wherever they are
    fields_ = [(key, [_field(view, key, v) for v in each(value)]) for key, value in document["fields"].items()]
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
        "backlinks": (into, view.href(to=document["id"], by="", **listing)) if into else None,
        "links": (out, view.href(by=document["id"], to="", **listing)) if out else None,
    }


def app(index: "Index", host: str = "127.0.0.1") -> "ASGIApp":
    """The UI's ASGI app over `index`, to serve under /ui. Served on a local address, it answers only requests that
    name a local host: a web page could otherwise reach it under its own domain (DNS rebinding) and read the
    documents. Its pages are made as the MCP tools' calls are run: one at a time, off the event loop (see `jobs`)."""
    from jinja2 import Environment, FileSystemLoader
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles

    env = Environment(loader=FileSystemLoader(WEB), autoescape=True, trim_blocks=True, lstrip_blocks=True)
    env.filters["num"] = lambda n: f"{n:,}"

    def render(view: View, part: str | None, headers: Mapping[str, str]) -> str:
        pattern = _pattern(view.q)
        context: Context = {
            "view": view,
            "modes": MODES,
            "oob": part == "explore",  # the form's parts come along, to show the view
            "typed": headers.get("hx-trigger") == "search",  # the box holds what is being typed: it stays
            "title": f"{view.q} · sift" if view.q else "sift",
            "collections": _collections([(c, index.catalog.store(c)) for c in index.active().values()]),
            "PAGE": PAGE,  # in the context, as included templates see it: jinja2 types its globals as its own alone
        }
        if part != "reader":
            context |= _explore(index, view, pattern)
        if part != "explore":
            context["reader"] = _reader(index, view, pattern)
        templates: dict[str | None, str] = {"explore": "explore.html", "reader": "reader.html"}
        name = templates.get(part, "page.html")
        return env.get_template(name).render(context)

    async def page(request: Request) -> HTMLResponse:
        view = View.of(request.query_params)
        part = request.headers.get("hx-target") if "hx-history-restore-request" not in request.headers else None
        return HTMLResponse(await jobs.run(render, view, part, request.headers), headers=UNKEPT | FRAMED_BY_NONE)

    def located(id: str) -> Path | None:
        """The file of the document `id`, if it is there: looked for here, off the event loop, which a share that
        doesn't answer would hold for as long as the system waits."""
        try:
            path = index.file(id)
        except LookupError:
            return None
        return path if path is not None and path.is_file() else None

    async def file(request: Request) -> Response:
        path = await jobs.run(located, request.query_params.get("id", ""))
        if path is None:
            return PlainTextResponse("no such file", status_code=404)
        response = FileResponse(path, filename=path.name, content_disposition_type="inline")
        # A file runs no scripts as the UI, where a saved web page could read the whole index. Chromium's PDF viewer
        # refuses sandboxed documents, and runs none of a PDF's scripts as the page anyway.
        if response.media_type != "application/pdf":
            response.headers["Content-Security-Policy"] = "sandbox"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    routes = [Route("/", page), Route("/file", file), Mount("/static", StaticFiles(directory=WEB / "static"))]
    ui = _no_other_site(Starlette(routes=routes))
    return _local_only(ui) if host in LOCAL_HOSTS else ui


def _no_other_site(app: "ASGIApp") -> "ASGIApp":
    """`app`, refusing what another site's page has the browser fetch in the background or show in a frame: a search,
    which holds the server's one call at a time and the GPU. Browsers say who asks (Fetch Metadata). A navigation of
    the whole tab to the UI goes on, a link followed or a page sending the browser there, which the user sees; so
    does a client that is no browser."""
    from starlette.responses import PlainTextResponse

    async def checked(scope: "Scope", receive: "Receive", send: "Send") -> None:
        headers = dict(scope.get("headers", ()))
        followed = headers.get(b"sec-fetch-mode") == b"navigate" and headers.get(b"sec-fetch-dest") == b"document"
        if scope["type"] == "http" and headers.get(b"sec-fetch-site") == b"cross-site" and not followed:
            await PlainTextResponse("refused: another site's page asked", status_code=403)(scope, receive, send)
            return
        await app(scope, receive, send)

    return checked


def _local_only(app: "ASGIApp") -> "ASGIApp":
    """`app`, answering only requests whose Host names a local host."""
    return _hosts_only(app, lambda name: name in LOCAL_HOSTS)


def host_name(name: str) -> str:
    """A name a server answers to beside its own (`--allow-host`), as a request's Host names it: in lower case,
    without a dot at its end, and one beyond ASCII as browsers send it (IDNA). An error for what is no name: an
    address needs no allowing, and a port or a path is no part of a name."""
    try:
        encoded = name.strip().lower().removesuffix(".").encode("idna").decode("ascii")
    except UnicodeError:
        encoded = ""
    if not HOST_NAME.fullmatch(encoded):
        raise ValueError(
            "--allow-host takes a name this machine is reached by, such as nas or notes.example, without a port or a "
            f"path: got {name!r}"
        )
    return encoded


def reached_only(app: "ASGIApp", allowed: Iterable[str] = ()) -> "ASGIApp":
    """`app`, served on an address other machines reach, answering only requests that name this machine by an
    address, by its own name (or that name's .local), by a local name, or by one `allowed` (`--allow-host`): a page of
    another site whose name leads to this machine's address (DNS rebinding) is refused; an address is no such name."""
    me = socket.gethostname().lower()
    names = {*LOCAL_HOSTS, me, f"{me}.local", *map(host_name, allowed)}

    def known(name: str) -> bool:
        try:
            ipaddress.ip_address(name)
        except ValueError:
            return name.lower().removesuffix(".") in names
        return True

    refusal = (
        "This sift answers to its machine's addresses and names alone: `sift serve --allow-host NAME` adds a name it "
        "is reached by."
    )
    return _hosts_only(app, known, refusal)


def _hosts_only(app: "ASGIApp", known: Callable[[str], bool], refusal: str = "Invalid host header") -> "ASGIApp":
    """`app`, answering only requests whose Host names a host `known` says it is, an IPv6 one in brackets too
    ("[::1]:8765"), which Starlette's TrustedHostMiddleware cuts at its first colon; any other gets `refusal`."""
    from starlette.responses import PlainTextResponse

    async def checked(scope: "Scope", receive: "Receive", send: "Send") -> None:
        host = dict(scope.get("headers", ())).get(b"host", b"").decode("latin-1")
        bracketed = host.startswith("[") and "]" in host
        name = host[1 : host.find("]")] if bracketed else host.partition(":")[0]
        if scope["type"] != "lifespan" and not known(name):
            await PlainTextResponse(refusal, status_code=400)(scope, receive, send)
            return
        await app(scope, receive, send)

    return checked


def address(host: str, port: int) -> str:
    """Where to reach a server listening on `host`: this machine's own address if it listens on all of them, and
    IPv6 addresses in brackets."""
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def open_in_browser(host: str = "127.0.0.1", port: int = 8765, allowed: Sequence[str] = ()) -> None:
    """`sift ui`: open the UI of the `sift serve` on this port in the browser, starting the server if none runs, with
    the names `allowed` (see reached_only)."""
    url = address(host, port) + PREFIX
    # Asked itself, not through a proxy set for the web, which knows no server of this machine.
    directly = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        directly.open(url, timeout=PROBE_SECONDS).close()
    except urllib.error.HTTPError as e:
        e.close()  # it holds the answer open: a ResourceWarning on Python 3.14+ when left to the garbage collector
        raise SystemExit(
            f"error: what answers on port {port} is no sift with a UI: an older `sift serve`? Restart it with this "
            "version of sift, or choose another port with --port"
        ) from None
    except TimeoutError:  # it listens and answers nobody: a sift whose models run, on the thread that answers
        console.say(f"the server on port {port} is busy: its page loads once it answers")
    except OSError:  # nobody listens: serve here, and open the page once the server answers
        import threading

        from . import server

        threading.Thread(target=_open_when_up, args=(url,), daemon=True).start()
        server.serve(host, port, allowed=allowed)
        return
    _open(url)


def _open(url: str) -> None:
    console.say(f"sift ui on {url}")
    webbrowser.open(url)


def _open_when_up(url: str, wait: float = STARTUP_SECONDS) -> None:
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
