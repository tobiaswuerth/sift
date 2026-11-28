from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from sift.api.browsing import Browsed
from sift.api.errors import SiftError
from sift.api.links import Links
from sift.api.search import Hit, SearchResult
from sift.api.wire import Json
from sift.interfaces.acts.acting import answered
from sift.interfaces.web.html import Html, joined, tag
from sift.interfaces.web.marking import Sought, snippet, sought
from sift.interfaces.web.view import MODES, MOST_SHOWN, PAGE, PageView, filtered, filters_of, folder_glob, href

FACET_VALUES = 8
SHORT = 60
WELCOME = "Search the collections, or pick one to browse it."


class PageReads(Protocol):
    def read(self, name: str, data: Json) -> object: ...


@dataclass(frozen=True, slots=True)
class Explored:
    main: Html
    side: Html


def refusal(message: str, remedy: str | None) -> Html:
    told = tag("p", {"class": "fix"}, f"fix: {remedy}") if remedy else ""
    return tag("div", {"class": "error", "role": "alert"}, tag("p", None, message), told)


def _options(view: PageView) -> dict[str, Json]:
    options: dict[str, Json] = {"k": view.k, "fields": ["*"], "filters": cast("Json", filters_of(view))}
    if view.c:
        options["collections"] = [view.c]
    if view.path:
        options["path"] = folder_glob(view.path)
    if view.sort:
        options["sort"] = view.sort
    if view.to:
        options["links_to"] = [view.to]
    if view.by:
        options["linked_from"] = [view.by]
    return options


def _short(value: object) -> bool:
    return isinstance(value, str | int | float) and not isinstance(value, bool) and len(str(value)) <= SHORT


def _shown(value: object) -> object:
    return ("true" if value else "false") if isinstance(value, bool) else value


def _each(value: object) -> list[object]:
    return list(cast("Sequence[object]", value)) if isinstance(value, list | tuple) else [value]


def _counted(hits: Iterable[Hit]) -> dict[str, Counter[object]]:
    counts: dict[str, Counter[object]] = {}
    for hit in hits:
        for key, value in (hit.fields or {}).items():
            for one in map(_shown, _each(value)):
                if _short(one) and not str(one).startswith(("http://", "https://")):
                    counts.setdefault(key, Counter())[one] += 1
    return counts


def _facets(hits: Sequence[Hit], view: PageView) -> Html:
    shown = list[Html]()
    for key, counter in sorted(_counted(hits).items(), key=lambda pair: (-max(pair[1].values()), pair[0])):
        if len(counter) > FACET_VALUES and max(counter.values()) == 1:
            continue
        values = [
            tag(
                "li",
                {"class": "on" if f"{key}={value}" in view.f else None},
                tag("a", {"href": filtered(view, key, value)}, str(value)),
                tag("span", {"class": "count"}, n),
            )
            for value, n in counter.most_common(FACET_VALUES)
        ]
        shown.append(tag("section", {"class": "facet"}, tag("h3", None, key), tag("ul", None, values)))
    return joined(shown)


def chips(view: PageView) -> Html:
    found = [
        tag(
            "a",
            {"class": "chip", "href": filtered(view, *pair.split("=", 1))},
            pair.replace("=", ": ", 1),
            " \N{MULTIPLICATION SIGN}",
        )
        for pair in view.f
    ]
    for note, label, name in ((view.to, "linking to", "to"), (view.by, "linked from", "by")):
        if note:
            found.append(
                tag("a", {"class": "chip", "href": href(view, **{name: ""})}, f"{label} {note} \N{MULTIPLICATION SIGN}")
            )
    return tag("div", {"class": "chips"}, found) if found else Html("")


def _hit(view: PageView, hit: Hit, terms: Sought) -> Html:
    lines = f"{hit.first_line}-{hit.last_line}"
    link = tag("a", {"href": f"{href(view, doc=hit.id, lines=lines, at=0)}#l{hit.first_line}"}, hit.title)
    shown = snippet(hit.text or "", terms, hit.title)
    return tag("li", {"class": "hit"}, tag("h2", None, link), tag("p", {"class": "id"}, hit.id), tag("p", None, shown))


def _more(view: PageView, *, more: bool) -> Html:
    if not more or view.k >= MOST_SHOWN:
        return Html("")
    return tag("a", {"class": "more", "href": href(view, k=min(view.k + PAGE, MOST_SHOWN))}, "More")


def _hits(view: PageView, result: SearchResult, heading: str) -> Explored:
    terms = sought(view.q)
    notes = [
        tag("p", {"class": "note"}, f"{each.collection}: {each.message}" if each.collection else each.message)
        for each in (*result.unsearched, *result.left_out)
    ]
    listed = (
        tag("ol", {"class": "hits"}, [_hit(view, hit, terms) for hit in result.hits])
        if result.hits
        else tag("p", None, "Nothing found.")
    )
    main = joined([tag("p", {"class": "heading"}, heading), notes, listed, _more(view, more=result.more)])
    return Explored(main, _facets(result.hits, view))


def _searched(view: PageView, reads: PageReads) -> Explored:
    label, act = MODES[view.mode]
    result = answered(reads.read(act, {**_options(view), "query": view.q}), SearchResult)
    return _hits(view, result, f"{label}: {view.q}")


def _alike(view: PageView, reads: PageReads) -> Explored:
    result = answered(reads.read("similar", {**_options(view), "id": view.similar}), SearchResult)
    return _hits(view, result, f"Like {view.similar}")


def _linking(view: PageView, reads: PageReads) -> Explored:
    note, direction = (view.to, "in") if view.to else (view.by, "out")
    links = answered(reads.read("links", {"id": note, "direction": direction, "k": view.k}), Links)
    if direction == "in":
        rows = [
            tag(
                "li",
                None,
                tag("a", {"href": href(view, doc=each.id, lines="", at=0)}, each.title),
                tag("p", {"class": "id"}, each.id),
            )
            for each in links.into or ()
        ]
        heading = f"Linking to {links.title}"
    else:
        rows = [
            tag(
                "li",
                None,
                tag("a", {"href": href(view, doc=each.id, lines="", at=0)}, each.title or each.target)
                if each.id
                else each.target,
            )
            for each in links.out or ()
        ]
        heading = f"Linked from {links.title}"
    listed = tag("ul", {"class": "hits"}, rows) if rows else tag("p", None, "None.")
    return Explored(joined([tag("p", {"class": "heading"}, heading), listed, _more(view, more=links.more)]), Html(""))


def _crumbs(view: PageView, browsed: Browsed) -> Html:
    found, folder = [tag("a", {"href": href(view, path="", k=PAGE)}, browsed.collection), "/"], ""
    for part in (each for each in browsed.folder.split("/") if each):
        folder += f"{part}/"
        found += [tag("a", {"href": href(view, path=folder, k=PAGE)}, part), "/"]
    return joined(found)


def _browsed(view: PageView, reads: PageReads) -> Explored:
    browsed = answered(reads.read("browse", {"collection": view.c, "folder": view.path, "k": view.k}), Browsed)
    folders = [
        tag(
            "li",
            {"class": "folder"},
            tag("a", {"href": href(view, path=f"{browsed.folder}{name}/", k=PAGE)}, f"{name}/"),
        )
        for name in browsed.folders
    ]
    documents = [
        tag(
            "li",
            None,
            tag("a", {"href": href(view, doc=each.id, lines="", at=0)}, each.title),
            tag("p", {"class": "id"}, each.id),
        )
        for each in browsed.documents
    ]
    heading = tag("p", {"class": "heading"}, _crumbs(view, browsed))
    main = joined(
        [
            heading,
            tag("ul", {"class": "folders"}, folders) if folders else "",
            tag("ul", {"class": "hits"}, documents),
            _more(view, more=browsed.more),
        ]
    )
    return Explored(main, Html(""))


def explored(view: PageView, reads: PageReads) -> Explored:
    try:
        if view.q:
            return _searched(view, reads)
        if view.similar:
            return _alike(view, reads)
        if view.to or view.by:
            return _linking(view, reads)
        if view.c:
            return _browsed(view, reads)
    except SiftError as error:
        return Explored(refusal(error.message, error.remedy), Html(""))
    return Explored(tag("p", {"class": "welcome"}, WELCOME), Html(""))
