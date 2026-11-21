"""What a search hands back: its options checked, and its hits, dicts as search.Index._hits makes them, ordered, cut to
those that stand out, one per work, and shown with what was asked for. None of it reads a store."""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from .filters import json_path

Hit = dict[str, Any]  # "id", "title", "lines", "score", "meta" (JSON), "text", "chunk"; "also" and "partial" if any

MOST_HITS = 10_000  # hits a search returns at most: vsearch(k=10**8) read every vector of every collection, and
# search(k=10**8) scored every posting of its words
ADAPTIVE_SHARE = 0.9  # of the gaps between neighbouring scores, the first, where the largest is looked for
ADAPTIVE_BUFFER = 5  # hits kept after the largest gap, as the paper keeps: of 10 or 50 vsearch hits for 2,000 arXiv
# titles, 6 on average, with the paper sought in 99.2 to 99.6% of the cases it was among them (benchmarks.md)
DOI_LINKS = tuple(f"{scheme}{host}doi.org/" for scheme in ("https://", "http://") for host in ("", "www.", "dx."))
DOI_PREFIXES = (*DOI_LINKS, "doi:")


@dataclass(frozen=True)
class Options:
    """What every search takes beside what it looks for and k. Where it looks: one `collection`, or every active one;
    the documents whose path matches the glob `path`, that pass the `filters` on their front matter, that link one of
    the notes `links_to` names, and that one of `linked_from` links. And how its hits come: the best passage of each
    document, or with `all_passages` every passage; with the front-matter `fields` asked for (["*"] for all);
    `compact`, without their text; by `sort`, a front-matter field that orders the k found instead of their score
    ("-created": newest first); and with `adaptive`, k at most, those before the largest drop in score and a few more
    (see standing_out)."""

    collection: str | None = None
    path: str | None = None
    filters: dict[str, Any] | None = None
    fields: Any = None
    compact: bool = False
    all_passages: bool = False
    sort: Any = None
    links_to: Any = None
    linked_from: Any = None
    adaptive: bool = False

    @property
    def where(self) -> tuple[str | None, str | None, dict[str, Any] | None, Any, Any]:
        """Where the search looks, as search.Index._where takes it."""
        return self.collection, self.path, self.filters, self.links_to, self.linked_from

    def checked(self) -> Self:
        """These options, their `fields` and `sort` checked."""
        check_fields(self.fields)
        check_sort(self.sort)
        return self

    def shown(self, hits: list[Hit]) -> list[Hit]:
        """Ranked hits as these options return them (see present)."""
        return present(standing_out(hits) if self.adaptive else hits, self.fields, self.compact, self.sort)


def check_k(k: object, most: int = MOST_HITS) -> int:
    """k as asked, at most `most`."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return min(k, most)


def check_budget(budget: object) -> float | None:
    if budget is None:
        return None
    if isinstance(budget, bool) or not isinstance(budget, int | float) or not budget > 0:
        raise ValueError(f"budget must be a positive number of seconds, got {budget!r}")
    return budget


def check_fields(fields: Any) -> Any:
    if fields is not None and (isinstance(fields, str) or not all(isinstance(f, str) and f for f in fields)):
        raise ValueError(f"fields must be a list of front-matter field names, or ['*'] for all, got {fields!r}")
    return fields


def check_sort(sort: Any) -> Any:
    if sort is not None:
        try:
            json_path(sort.removeprefix("-") if isinstance(sort, str) else sort)
        except ValueError:
            raise ValueError(
                f"sort must be a front-matter field, with - first to sort descending, e.g. '-created'; got {sort!r}"
            ) from None
    return sort


def line_span(lines: object) -> tuple[int, int]:
    """First and last line of a range such as "17-26", or of a single line: "17" or 17."""
    if isinstance(lines, int) and not isinstance(lines, bool):
        lines = str(lines)
    m = re.fullmatch(r"([1-9]\d*)(?:-([1-9]\d*))?", lines) if isinstance(lines, str) else None
    if not m or int(m[2] or m[1]) < int(m[1]):
        raise ValueError(f"lines must be a line number or a range such as '17-26', got {lines!r}")
    return int(m[1]), int(m[2] or m[1])


def pick(meta: str, fields: list[str]) -> dict[str, Any]:
    """The front-matter `fields` asked for that a document has, from its meta (JSON); ["*"]: all but the title."""
    held: dict[str, Any] = json.loads(meta)
    return {
        key: held[key] for key in ([key for key in held if key != "title"] if "*" in fields else fields) if key in held
    }


def by_field(hits: list[Hit], sort: str) -> list[Hit]:
    """Hits ordered by the front-matter field `sort` names, descending with "-" first: numbers before text, after it
    descending, as SQLite orders them (see search.Index.browse), and hits without the field last, in their order."""
    field = sort.removeprefix("-")
    valued: list[tuple[Any, Hit]] = [(json.loads(hit["meta"]).get(field), hit) for hit in hits]
    have = [(value, hit) for value, hit in valued if value is not None]
    have.sort(key=lambda vh: (0, vh[0]) if isinstance(vh[0], int | float) else (1, str(vh[0])), reverse=sort[0] == "-")
    return [hit for _, hit in have] + [hit for value, hit in valued if value is None]


def present(hits: list[Hit], fields: list[str] | None, compact: bool, sort: str | None) -> list[Hit]:
    """Hits as returned: ordered by `sort` if given, with the front-matter `fields` asked for, without text if
    compact, the other copies of the same work under "also", and "partial" if found by a search out of time."""
    out: list[Hit] = []
    for hit in by_field(hits, sort) if sort else hits:
        shown = {key: hit[key] for key in ("id", "title", "lines", "score")}
        if fields:
            shown["fields"] = pick(hit["meta"], fields)
        if not compact:
            shown["text"] = hit["text"]
        if hit.get("also"):
            shown["also"] = hit["also"]
        if hit.get("partial"):
            shown["partial"] = True
        out.append(shown)
    return out


def doi_of(meta: str | None) -> str | None:
    """A document's DOI, from its front-matter field `doi`, as DOIs compare: in any case, without the prefix of a link
    or "doi:"; None if it has none, or one that is no DOI (they all start with "10.")."""
    value: Any = json.loads(meta).get("doi") if meta else None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    for prefix in DOI_PREFIXES:
        value = value.removeprefix(prefix).strip()  # "doi: 10...."
    return value if value.startswith("10.") else None


def one_per_work(hits: list[Hit]) -> list[Hit]:
    """Ranked hits, one per work: a hit with the DOI of a better ranked one goes, as several collections hold the same
    paper, and the better ranked one names it under "also"."""
    kept: list[Hit] = []
    first: dict[str, Hit] = {}
    for hit in hits:
        doi = doi_of(hit["meta"])
        if doi is None or doi not in first:
            kept.append(hit)
            if doi is not None:
                first[doi] = hit
        else:
            first[doi].setdefault("also", []).append(hit["id"])
    return kept


def standing_out(hits: list[Hit]) -> list[Hit]:
    """Hits, best first, up to the largest drop in score among the first ADAPTIVE_SHARE of their gaps, and
    ADAPTIVE_BUFFER more: Adaptive-k (Taguchi et al. 2025, arXiv 2506.08479), the few that stand out instead of k."""
    if len(hits) < 3:
        return hits
    scores = np.array([hit["score"] for hit in hits], np.float64)
    gaps = scores[:-1] - scores[1:]
    considered = gaps[: math.ceil(ADAPTIVE_SHARE * len(gaps))]
    if not considered.max() > 0:  # no drop at all: none stands out
        return hits
    return hits[: int(considered.argmax()) + 1 + ADAPTIVE_BUFFER]
