import json
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids
from sift.domain.codec.segments import counted, unpacked
from sift.domain.errors import DamagedError
from sift.storage.columns import columns
from sift.storage.database import Count, Reading, batches, first, rows

MOST_POINT = 0x10FFFF
SURROGATES_START, SURROGATES_END = 0xD800, 0xE000
WITHIN = """SELECT lists FROM keyword_packs WHERE segment = :segment AND first >= coalesce(
                (SELECT max(first) FROM keyword_packs WHERE segment = :segment AND first <= :low), :low)"""
PACKS_WITHIN = (f"{WITHIN} ORDER BY first", f"{WITHIN} AND first < :high ORDER BY first")
PACKS_AT_ONCE = 64
GONE = "SELECT term, sum(chunks) FROM keyword_gone WHERE term IN (SELECT value FROM json_each(?)) GROUP BY term"
GONE_WITHIN = (
    "SELECT term, sum(chunks) FROM keyword_gone WHERE term >= :low GROUP BY term",
    "SELECT term, sum(chunks) FROM keyword_gone WHERE term >= :low AND term < :high GROUP BY term",
)
PACK_OF = """SELECT s.id, t.value,
                    (SELECT p.rowid FROM keyword_packs p WHERE p.segment = s.id AND p.first <= t.value
                     ORDER BY p.first DESC LIMIT 1)
             FROM keyword_served s, json_each(?) t"""
RANGES = "SELECT ranges FROM keyword_ranges WHERE segment = ? AND term = ?"
PACKS = "SELECT rowid, segment, lists FROM keyword_packs WHERE rowid IN (SELECT value FROM json_each(?))"
PACK_PLACED = """SELECT lists, places FROM keyword_packs WHERE segment = ? AND first <= ?
                 ORDER BY first DESC LIMIT 1"""


class SegmentRow(NamedTuple):
    id: int
    first: int
    last: int
    chunks: int
    average: float


class Part(NamedTuple):
    segment: int
    term: str
    chunks: int
    ranged: int
    data: bytes


class PackOf(NamedTuple):
    segment: int
    term: str
    pack: int | None


class PackLists(NamedTuple):
    id: int
    segment: int
    lists: bytes


class PackPlaced(NamedTuple):
    lists: bytes
    places: bytes


class RangesRow(NamedTuple):
    ranges: bytes


class BlockRow(NamedTuple):
    number: int
    postings: bytes


class PositionsRow(NamedTuple):
    number: int
    positions: bytes


class Frequency(NamedTuple):
    term: str
    chunks: int


class PackHeads(NamedTuple):
    lists: bytes


class LengthsRow(NamedTuple):
    block: int
    lengths: bytes


def segments(transaction: Reading) -> list[SegmentRow]:
    sql = "SELECT id, first, last, chunks, average FROM keyword_served ORDER BY first"
    return rows(transaction, SegmentRow, sql)


class IdsRow(NamedTuple):
    ids: str


def dead_within(transaction: Reading, low: int, high: int) -> Ids:
    sql = """SELECT json_group_array(place) FROM
             (SELECT place FROM keyword_dead WHERE place BETWEEN ? AND ? ORDER BY place)"""
    (found,) = rows(transaction, IdsRow, sql, (low, high))
    return np.array(json.loads(found.ids), np.int64)


def chunks_placed(transaction: Reading, places: Sequence[int]) -> tuple[Ids, Ids, Ids]:
    sql = "SELECT place, id, document FROM chunks WHERE place IN (SELECT value FROM json_each(?)) ORDER BY place"
    found, chunks, documents = columns(transaction, sql, (json.dumps(list(places)),), 3)
    return found, chunks, documents


def gone(transaction: Reading, terms: Sequence[str]) -> dict[str, int]:
    return {found.term: found.chunks for found in rows(transaction, Frequency, GONE, (json.dumps(list(terms)),))}


def _after(prefix: str) -> str | None:
    kept = prefix.rstrip(chr(MOST_POINT))
    if not kept:
        return None
    following = ord(kept[-1]) + 1
    return kept[:-1] + chr(SURROGATES_END if SURROGATES_START <= following < SURROGATES_END else following)


def _counted_within(transaction: Reading, low: str, high: str | None) -> Iterator[list[tuple[str, int]]]:
    for segment in segments(transaction):
        within = {"segment": segment.id, "low": low, "high": high}
        for packs in batches(transaction, PackHeads, PACKS_WITHIN[high is not None], within, PACKS_AT_ONCE):
            yield counted([pack.lists for pack in packs], low, high)


def prefixed(transaction: Reading, prefix: str, most: int) -> dict[str, int]:
    high = _after(prefix)
    deleted = rows(transaction, Frequency, GONE_WITHIN[high is not None], {"low": prefix, "high": high})
    held: dict[str, int] = {}
    for found in _counted_within(transaction, prefix, high):
        for term, chunks in found:
            held[term] = held.get(term, 0) + chunks
        if len(held) - len(deleted) > most:
            break
    for each in deleted:
        held[each.term] = held.get(each.term, 0) - each.chunks
    return {term: chunks for term, chunks in sorted(held.items()) if chunks > 0}


def _packed(transaction: Reading, terms: Sequence[str]) -> dict[int, list[str]]:
    held: dict[int, list[str]] = {}
    for found in rows(transaction, PackOf, PACK_OF, (json.dumps(sorted(set(terms))),)):
        if found.pack is not None:
            held.setdefault(found.pack, []).append(found.term)
    return held


def _ranged(transaction: Reading, segment: int, term: str) -> bytes:
    found = first(transaction, RangesRow, RANGES, (segment, term))
    if found is None:
        raise DamagedError("the ranges of a long list are missing from the keyword index")
    return found.ranges


def parts(transaction: Reading, terms: Sequence[str]) -> list[Part]:
    asked = _packed(transaction, terms)
    found: list[Part] = []
    for pack in rows(transaction, PackLists, PACKS, (json.dumps(list(asked)),)):
        for held in unpacked(pack.lists, None, asked[pack.id]):
            if held.postings is None:
                ranges = _ranged(transaction, pack.segment, held.term)
                found.append(Part(pack.segment, held.term, held.chunks, 1, ranges))
            else:
                found.append(Part(pack.segment, held.term, held.chunks, 0, held.postings))
    return sorted(found, key=lambda part: (part.segment, part.term))


def blocks(transaction: Reading, segment: int, term: str, numbers: Sequence[int]) -> list[BlockRow]:
    sql = """SELECT number, postings FROM keyword_blocks
             WHERE segment = ? AND term = ? AND number IN (SELECT value FROM json_each(?)) ORDER BY number"""
    return rows(transaction, BlockRow, sql, (segment, term, json.dumps(list(numbers))))


def positions(transaction: Reading, segment: int, term: str, numbers: Sequence[int] | None) -> list[PositionsRow]:
    if numbers is None:
        pack = first(transaction, PackPlaced, PACK_PLACED, (segment, term))
        held = [] if pack is None else unpacked(pack.lists, pack.places, [term])
        return [PositionsRow(0, each.positions) for each in held if each.positions is not None]
    sql = """SELECT number, positions FROM keyword_blocks
             WHERE segment = ? AND term = ? AND number IN (SELECT value FROM json_each(?)) ORDER BY number"""
    return rows(transaction, PositionsRow, sql, (segment, term, json.dumps(list(numbers))))


def lengths(transaction: Reading, segment: int, numbers: Sequence[int]) -> list[LengthsRow]:
    sql = """SELECT block, lengths FROM keyword_lengths
             WHERE segment = ? AND block IN (SELECT value FROM json_each(?)) ORDER BY block"""
    return rows(transaction, LengthsRow, sql, (segment, json.dumps(list(numbers))))


def keyword_generation(transaction: Reading) -> int:
    (held,) = rows(transaction, Count, "SELECT keyword_changes FROM counts")
    return held.n
