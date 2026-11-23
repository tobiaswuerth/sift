from collections.abc import Callable, Sequence


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
