import re
from collections.abc import Callable, Mapping, Sequence
from urllib.parse import unquote

from sift.api.fields import FieldValue

DOI_FIELD = "doi"
DOI = re.compile(r"(?<![0-9])10\.[0-9]+(?:\.[0-9]+)*/\S+")
DOI_END = ".,;:)]}>\"'"
MOST_DOI = 1_000


def work_of(doi: object) -> str | None:
    found = DOI.search(unquote(doi[:MOST_DOI])) if isinstance(doi, str) else None
    return found.group().lower().rstrip(DOI_END) if found else None


def work_of_fields(fields: Mapping[str, FieldValue]) -> str | None:
    return work_of(fields.get(DOI_FIELD))


def one_per_work[T](ranked: Sequence[T], work: Callable[[T], str | None], copied: Callable[[T, T], T]) -> list[T]:
    kept: list[T] = []
    first: dict[str, int] = {}
    for each in ranked:
        found = work(each)
        if found is not None and found in first:
            kept[first[found]] = copied(kept[first[found]], each)
            continue
        if found is not None:
            first[found] = len(kept)
        kept.append(each)
    return kept
