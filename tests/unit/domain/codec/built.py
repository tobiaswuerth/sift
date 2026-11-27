from collections.abc import Iterable, Sequence

import numpy as np
from hypothesis import strategies as st
from numpy.typing import NDArray

from sift.domain.codec.segments import WrittenSegment, building, written
from sift.domain.codec.shape import SHAPE, Shape
from tests.coded_lists import packed

SMALL = Shape(range_shift=2, block=2, long=3, lengths_shift=3)
EVERY_TOKEN = 1 << 62
SEGMENT = 7

type Columns = tuple[list[str], ...]
words = st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), max_size=6)
chunks = st.lists(st.tuples(words, words, words), min_size=1, max_size=25)


def of_bodies(each: Iterable[Sequence[str]]) -> list[Columns]:
    return [(list[str](), [*words], list[str]()) for words in each]


def short_lists(made: WrittenSegment) -> dict[str, tuple[bytes, bytes]]:
    entries = (each for _, _, lists, places in made.filed.packs for each in packed(lists, places))
    return {term: (coded, placed) for term, _, coded, placed in entries if coded}


def segment(ids: NDArray[np.int64], columns: Sequence[Columns], average: float, shape: Shape = SHAPE) -> WrittenSegment:
    built = building(EVERY_TOKEN)
    fields = [(" ".join(title), " ".join(meta)) for title, _, meta in columns]
    assert built.add(ids, range(len(columns)), fields, [" ".join(body) for _, body, _ in columns]) == len(columns)
    return written(built, SEGMENT, average, shape)
