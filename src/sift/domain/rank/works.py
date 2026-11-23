from collections.abc import Callable, Mapping, Sequence

from sift.api.fields import FieldValue

DOI_FIELD = "doi"
DOI_LINKS = tuple(f"{scheme}{host}doi.org/" for scheme in ("https://", "http://") for host in ("", "www.", "dx."))
DOI_PREFIXES = (*DOI_LINKS, "doi:")


def work_of(doi: object) -> str | None:
    if not isinstance(doi, str):
        return None
    value = doi.strip().lower()
    for prefix in DOI_PREFIXES:
        value = value.removeprefix(prefix).strip()
    return value if value.startswith("10.") else None


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
