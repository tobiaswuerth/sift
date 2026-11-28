import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast
from urllib.parse import urlencode

PREFIX = "/ui"
PAGE = 30
MOST_SHOWN = 500
MOST_FILTERS = 32
MOST_PATH_CHARACTERS = 4_096
MOST_FOLDERS_DEEP = 256
MODES = MappingProxyType(
    {"keyword": ("Keywords", "search"), "semantic": ("Meaning", "vsearch"), "hybrid": ("Hybrid", "query")}
)
COUNT = re.compile("[0-9]{1,9}")
GLOB_SPECIAL = re.compile(r"[\[*?]")


@dataclass(frozen=True, slots=True)
class PageView:
    q: str = ""
    mode: str = "keyword"
    c: str = ""
    path: str = ""
    f: tuple[str, ...] = ()
    sort: str = ""
    k: int = PAGE
    similar: str = ""
    to: str = ""
    by: str = ""
    doc: str = ""
    lines: str = ""
    at: int = 0


DEFAULT = PageView()


def _count(text: str) -> int:
    return int(text) if COUNT.fullmatch(text) else 0


def view_of(pairs: Sequence[tuple[str, str]]) -> PageView:
    given = dict(pairs)
    parts = [part for part in given.get("path", "")[:MOST_PATH_CHARACTERS].split("/") if part]
    path = "/".join(parts[:MOST_FOLDERS_DEEP])
    mode = given.get("mode", "")
    return PageView(
        q=given.get("q", "").strip(),
        mode=mode if mode in MODES else DEFAULT.mode,
        c=given.get("c", ""),
        path=f"{path}/" if path else "",
        f=tuple(value for key, value in pairs if key == "f" and "=" in value)[:MOST_FILTERS],
        sort=given.get("sort", ""),
        k=min(_count(given.get("k", "")), MOST_SHOWN) or PAGE,
        similar=given.get("similar", ""),
        to=given.get("to", ""),
        by=given.get("by", ""),
        doc=given.get("doc", ""),
        lines=given.get("lines", ""),
        at=_count(given.get("at", "")),
    )


def href(view: PageView, **changes: object) -> str:
    changed = replace(view, **changes)
    pairs: list[tuple[str, object]] = []
    for field in dataclasses.fields(PageView):
        value = cast("object", getattr(changed, field.name))
        if value != getattr(DEFAULT, field.name):
            values = cast("tuple[object, ...]", value) if field.name == "f" else (value,)
            pairs.extend((field.name, each) for each in values)
    return PREFIX + ("?" + urlencode(pairs) if pairs else "")


def filtered(view: PageView, key: str, value: object) -> str:
    pair = f"{key}={value}"
    kept = tuple(each for each in view.f if each != pair)
    return href(view, f=kept if pair in view.f else (*view.f, pair), k=PAGE)


def folder_glob(folder: str) -> str:
    return GLOB_SPECIAL.sub(lambda found: f"[{found[0]}]", folder) + "*"


def filters_of(view: PageView) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = {}
    for pair in view.f:
        key, _, value = pair.partition("=")
        grouped.setdefault(key, []).append(value)
    return [{"field": key, "operator": "=", "values": values} for key, values in grouped.items()]
