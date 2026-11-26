from sift.api.collections import Counted, Status
from sift.api.errors import SiftError
from sift.interfaces.acts.acting import answered
from sift.interfaces.web.explore import PageReads, chips, explored, refusal
from sift.interfaces.web.html import Html, joined, tag
from sift.interfaces.web.reader import reader
from sift.interfaces.web.view import MODES, PAGE, PREFIX, PageView, href

DOCTYPE = Html("<!doctype html>")
NAME = "sift"


def _collections(view: PageView, reads: PageReads) -> Html:
    try:
        status = answered(reads.read("status", {}), Status)
    except SiftError as error:
        return tag("section", {"class": "collections"}, refusal(error.message, error.remedy))
    rows = list[Html]()
    for one in status.collections:
        held = f"{one.contents.documents:,}" if isinstance(one.contents, Counted) else one.contents.message
        browse = href(view, c=one.name, path="", q="", similar="", f=(), to="", by="", k=PAGE)
        rows.append(
            tag(
                "li",
                {"class": "on" if one.name == view.c else None},
                tag("a", {"href": browse}, one.name),
                tag("span", {"class": "count"}, held),
            )
        )
    return tag("section", {"class": "collections"}, tag("h3", None, "Collections"), tag("ul", None, rows))


def _form(view: PageView) -> Html:
    modes = [tag("option", {"value": mode, "selected": mode == view.mode}, label) for mode, (label, _) in MODES.items()]
    kept = [
        tag("input", {"type": "hidden", "name": name, "value": value})
        for name, value in (("c", view.c), ("path", view.path), ("sort", view.sort), ("to", view.to), ("by", view.by))
        if value
    ]
    filters = [tag("input", {"type": "hidden", "name": "f", "value": pair}) for pair in view.f]
    query = tag("input", {"type": "search", "name": "q", "value": view.q, "placeholder": "Search", "autofocus": True})
    return tag(
        "form",
        {"class": "search", "action": PREFIX, "method": "get", "role": "search"},
        tag("a", {"class": "home", "href": PREFIX}, NAME),
        query,
        tag("select", {"name": "mode", "aria-label": "how to search"}, modes),
        kept,
        filters,
        tag("button", {"type": "submit"}, "Search"),
    )


def page(view: PageView, reads: PageReads) -> Html:
    found = explored(view, reads)
    title = f"{view.q} \N{MIDDLE DOT} {NAME}" if view.q else NAME
    head = tag(
        "head",
        None,
        tag("meta", {"charset": "utf-8"}),
        tag("meta", {"name": "viewport", "content": "width=device-width, initial-scale=1"}),
        tag("title", None, title),
        tag("link", {"rel": "stylesheet", "href": f"{PREFIX}/style.css"}),
    )
    side = tag("aside", {"class": "side"}, _collections(view, reads), found.side)
    results = tag("section", {"class": "results"}, chips(view), found.main)
    body = tag("body", None, tag("header", None, _form(view)), tag("main", None, side, results, reader(view, reads)))
    return joined([DOCTYPE, tag("html", {"lang": "en"}, head, body)])
